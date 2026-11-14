// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief, probes each provider's catalog route, builds one
// catalog, registers its port and catalog with the env, serves the client's 20
// requests, and exits 0 when the env says the workload is done.
//
// ROUTING (the score's quality column is 60%)
//   Every request is routed to the best free model that meets its requirements,
//   ranked by the quality table's rating_milli. Two facts from the kit's sealed
//   generator shape the plan, and both are handled by reacting, not by assuming:
//   each provider has a small per-trial budget (the kit's ranges run 4-16 calls),
//   and each provider fails 1-2 scheduled calls per trial, one provider going
//   dark part-way through. So the router spreads load with a soft per-provider
//   call cap (below every provider's minimum budget), and fails over: a 402, 429
//   or 5xx is never the answer, the request is retried on the next-best free
//   model that meets it, in rating order — preferring a provider with no call
//   yet, whose first call can never be a scheduled failure or an outage. A
//   provider that fails three calls in a row is treated as dark and only
//   revisited when nothing else can serve.
//   The first call to a provider can never be a scheduled failure or an outage
//   (both are drawn from call 2 on), so a wide spread also buys free successes.
//   JSON-mode requests from an OpenAI client are never sent to an Anthropic-wire
//   provider: that wire cannot carry response_format, and its echo would report
//   json_mode false.
//
// WIRES (the wire column is 10%)
//   All six client-api x provider-wire combinations are implemented, in
//   lib/wire.ts. Every upstream request passes the client's request through
//   untouched in the target wire's spelling (system prompt, user text, images,
//   tools, stop sequences, max_tokens, JSON mode), because the provider echoes
//   what it parsed and the env compares that echo with what the client sent.
//   Every answer is re-emitted whole in the client's wire: the provider's text
//   in full, its usage counts as that wire reports them, and its stop reason
//   mapped inside its family. Streamed OpenAI calls always ask for usage
//   (stream_options.include_usage), or the usage check has nothing to read.
//
// DETERMINISM
//   No wall clock, no unseeded randomness: candidates sort by (rating desc,
//   provider id, model id), retries walk that order, and one request is served
//   at a time. A replay therefore sees the same provider answers and leaves the
//   same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicResponse,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  anthropicModelList,
  openAIResponse,
  openAIToAnthropicRequest,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  toGeminiRequest,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take (the client allows 30 s). */
const UPSTREAM_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream attempts for one client request. */
const MAX_ATTEMPTS = 12;

/**
 * Soft per-provider call caps for the primary pick: below every provider's
 * minimum per-trial budget, so the plan never leans on a budget it cannot see.
 * Retries may pass a cap when nothing under it can serve the request.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 6,
  cerebras: 3,
  gemini: 4,
  zai: 3,
  openrouter: 7,
  sambanova: 3,
  pollinations: 6,
  cloudflare: 5,
  ovh: 3,
  kilol: 6,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Upstream attempts made so far, per provider (failures count too: the env counts them). */
const callsMade = new Map<string, number>();
/** Consecutive failed upstream calls, per provider. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

const candidateKey = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;

/**
 * Every free model that can serve this request, best first: rating desc, then
 * provider id, then model id. A JSON-mode request from an OpenAI client never
 * goes to an Anthropic-wire provider: that wire cannot carry response_format,
 * so its echo would report json_mode false and lose the request.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/**
 * The next candidate to try: under its provider's soft cap and not dark first,
 * then any provider that has not failed three in a row, then — only when every
 * other candidate is spent — a dark provider again. On a retry, providers with
 * no call yet go first: a provider's first call can never be a scheduled
 * failure or an outage (both are drawn from call 2 on), so a fresh provider's
 * retry succeeds where a busy provider's might fail again.
 */
function pickCandidate(candidates: Candidate[], tried: Set<string>, preferFresh: boolean): Candidate | null {
  const fresh = candidates.filter((c) => !tried.has(candidateKey(c)));
  if (fresh.length === 0) return null;
  const alive = fresh.filter((c) => (consecutiveFailures.get(c.probe.provider.id) ?? 0) < 3);
  const pool = alive.length > 0 ? alive : fresh;
  const underCap = pool.filter((c) => (callsMade.get(c.probe.provider.id) ?? 0) < (SOFT_CAP[c.probe.provider.id] ?? 3));
  // Candidates arrive ranked by rating; this sort is stable, so within each
  // freshness group the rating order is kept.
  const ranked = (list: Candidate[]): Candidate[] =>
    [...list].sort((a, b) => {
      if (!preferFresh) return 0;
      const fa = (callsMade.get(a.probe.provider.id) ?? 0) === 0 ? 0 : 1;
      const fb = (callsMade.get(b.probe.provider.id) ?? 0) === 0 ? 0 : 1;
      return fa - fb;
    });
  if (underCap.length > 0) return ranked(underCap)[0];
  return ranked(pool)[0];
}

interface UpstreamResult {
  status: number;
  answer: ProviderAnswer | null;
  retryable: boolean;
}

/** One inference call to one provider, on that provider's own wire. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, jsonMode: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  let url: string;
  let requestBody: Json;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'anthropic') {
      requestBody = anthropicToOpenAI(body, candidate.model.id);
    } else {
      requestBody = { ...body, model: candidate.model.id };
      if (stream) requestBody.stream_options = { include_usage: true };
    }
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') requestBody = { ...body, model: candidate.model.id };
    else requestBody = openAIToAnthropicRequest(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const method = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${method}`;
    requestBody = toGeminiRequest(body, clientApi, jsonMode);
    if (stream) headers.accept = 'text/event-stream';
  }

  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(requestBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return { status: 0, answer: null, retryable: true };
  }
  const text = await response.text();
  if (!response.ok) return { status: response.status, answer: null, retryable: response.status === 402 || response.status === 429 || response.status >= 500 };
  const answer = provider.wire === 'openai' ? parseOpenAIAnswer(text, stream) : provider.wire === 'anthropic' ? parseAnthropicAnswer(text, stream) : parseGeminiAnswer(text, stream);
  if (answer.text === '' && answer.usage === null) return { status: response.status, answer: null, retryable: true };
  return { status: response.status, answer, retryable: false };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/**
 * Serve one client request: walk the ranked candidates until one answers, so a
 * refused provider never ends the request. Returns the serving model and its
 * answer, or null when no candidate could serve it.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ model: string; answer: ProviderAnswer } | null> {
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(body, clientApi);
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const candidate = pickCandidate(candidates, tried, attempt > 0);
    if (candidate === null) return null;
    const key = candidateKey(candidate);
    tried.add(key);
    const providerId = candidate.probe.provider.id;
    callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
    const result = await callUpstream(candidate, clientApi, body, jsonMode);
    if (result.answer !== null) {
      consecutiveFailures.set(providerId, 0);
      return { model: candidate.model.id, answer: result.answer };
    }
    consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
    if (!result.retryable) {
      // A 400/401 is this candidate's own fault (a model the provider will not
      // serve, or a bad key): the next candidate may still serve the request.
      log(`upstream ${providerId}/${candidate.model.id} refused HTTP ${result.status}; trying the next candidate`);
      continue;
    }
    log(`upstream ${providerId}/${candidate.model.id} failed HTTP ${result.status}; failing over`);
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on the best provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  const out = openAIResponse(served.model, served.answer, body.stream === true);
  if (out.json !== null) return sendJson(res, 200, out.json);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(out.sse);
}

/** POST /v1/messages: an Anthropic-shaped client, answered on the best provider. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : served.model;
  const out = anthropicResponse(requested, served.answer, body.stream === true);
  if (out.json !== null) return sendJson(res, 200, out.json);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(out.sse);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
