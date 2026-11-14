// The client wires and the three upstream wires, in both directions.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. The entry forwards each request to a provider on that provider's own
// wire — openai, anthropic or gemini — and translates the answer back into the
// wire the client used.
//
// Why translation has to be exact: the provider echoes what it parsed (system
// prompt, user text, image, tool names, max_tokens, stop sequences, JSON mode),
// and the env compares that echo with what the client sent. A field rewritten or
// dropped loses the request. So every builder below passes the request through
// untouched in the target wire's spelling, and every answer is re-emitted whole:
// the provider's text in full, its usage counts as that wire reports them, and
// its stop reason mapped inside its family (stop / end_turn / STOP are one
// family; the simulated providers only ever answer with that family).
//
// Everything here is a pure function of its arguments: no clock and no randomness,
// so the same body and the same provider answer give the same bytes on a replay.
// The two model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A provider's answer, reduced to what the client wire needs. */
export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  stop: string | null;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The stop family of a reason: 0 stop, 1 length, 2 tool use, 3 refusal; null unknown. */
export function stopFamily(reason: string | null): number | null {
  if (reason === null) return null;
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

const OPENAI_STOP = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOP = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

/** A provider stop reason in the OpenAI wire's spelling. */
export function openAIStop(reason: string | null): string {
  const family = stopFamily(reason);
  return family === null ? 'stop' : OPENAI_STOP[family];
}

/** A provider stop reason in the Anthropic wire's spelling. */
export function anthropicStop(reason: string | null): string {
  const family = stopFamily(reason);
  return family === null ? 'end_turn' : ANTHROPIC_STOP[family];
}

// ---------------------------------------------------------------------------
// Client Anthropic -> upstream OpenAI
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function anthropicPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON otherwise. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client OpenAI -> upstream Anthropic
// ---------------------------------------------------------------------------

/** One OpenAI content part as an Anthropic content block: text or an image. */
function openAIPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. System messages join into the system string; user content (text and
 * images) becomes Anthropic blocks; assistant tool_calls become tool_use blocks;
 * tool messages become tool_result blocks; tools, tool_choice, max_tokens,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systems: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systems.push(text);
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = openAIPartOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systems.length > 0) out.system = systems.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Either client -> upstream Gemini
// ---------------------------------------------------------------------------

/** An image URL as a Gemini part: a data URL becomes inlineData, the rest fileData. */
function geminiImagePart(url: string): Json {
  const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] } };
  return { fileData: { fileUri: url } };
}

/** OpenAI tools, or Anthropic tools, as Gemini functionDeclarations. */
function geminiTools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      declarations.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else if (typeof raw.name === 'string') {
      declarations.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return declarations;
}

/**
 * Either client's request as a Gemini generateContent body. The system prompt
 * becomes systemInstruction; user messages become user contents (text and image
 * parts); assistant messages become model contents, with tool calls as
 * functionCall parts; tool results travel as functionResponse parts on model
 * contents, so the echo's user text — the last user content's text — is never
 * overwritten by a tool message. Generation config carries maxOutputTokens,
 * temperature, topP, stopSequences and, for a JSON-mode request,
 * responseMimeType application/json, which is what makes the provider echo
 * json_mode true.
 */
export function toGeminiRequest(body: Json, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const addSystem = (text: string): void => {
    if (text !== '') systemParts.push({ text });
  };

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = typeof raw.role === 'string' ? raw.role : 'user';
      if (role === 'system') {
        addSystem(textOf(raw.content));
        continue;
      }
      if (role === 'tool') {
        // A tool result: keep it on a model content so it cannot overwrite the
        // echo's user text (the provider reads the last non-model content).
        contents.push({ role: 'model', parts: [{ functionResponse: { name: '', response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }] });
        continue;
      }
      if (role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(geminiImagePart(part.image_url.url));
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  } else {
    addSystem(textOf(body.system));
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const assistant = raw.role === 'assistant';
      const content = raw.content;
      if (typeof content === 'string') {
        contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push(geminiImagePart(block.source.url));
          }
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: toolResultText(block.content) } } });
        }
      }
      if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const declarations = geminiTools(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
  if (Array.isArray(body.stop_sequences)) for (const s of body.stop_sequences) if (typeof s === 'string') stops.push(s);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers -> ProviderAnswer
// ---------------------------------------------------------------------------

/** One SSE frame's lines as {event, data} pairs. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI provider answer (JSON or SSE) as text, usage and stop reason. */
export function parseOpenAIAnswer(text: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return { text: '', usage: null, stop: null };
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const content = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) } : null;
    return { text: content, usage, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
  }
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, usage, stop };
}

/** An Anthropic provider answer (JSON or SSE) as text, usage and stop reason. */
export function parseAnthropicAnswer(text: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return { text: '', usage: null, stop: null };
    const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) } : null;
    return { text: textOf(body.content), usage, stop: typeof body.stop_reason === 'string' ? body.stop_reason : null };
  }
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 }, stop };
}

/** A Gemini provider answer (JSON or SSE) as text, usage and stop reason. */
export function parseGeminiAnswer(text: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return { text: '', usage: null, stop: null };
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const out = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
    const usage = meta === null ? null : { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
    return { text: out, usage, stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null };
  }
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, usage, stop };
}

// ---------------------------------------------------------------------------
// ProviderAnswer -> client wire
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into fixed-size pieces, like the providers' own streams do. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A provider answer as an OpenAI Chat Completions body, JSON or SSE. */
export function openAIResponse(model: string, answer: ProviderAnswer, stream: boolean): { json: Json | null; sse: string | null } {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const tokens = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
  const stop = openAIStop(answer.stop);
  if (!stream) {
    return {
      json: {
        id: 'chatcmpl-aggregate',
        object: 'chat.completion',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
        usage: tokens,
      },
      sse: null,
    };
  }
  const frames: string[] = [sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: stop }] }));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: tokens }));
  frames.push('data: [DONE]\n\n');
  return { json: null, sse: frames.join('') };
}

/** A provider answer as an Anthropic Messages body, JSON or SSE. */
export function anthropicResponse(model: string, answer: ProviderAnswer, stream: boolean): { json: Json | null; sse: string | null } {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const stop = anthropicStop(answer.stop);
  if (!stream) {
    return {
      json: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      },
      sse: null,
    };
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return { json: null, sse: frames.join('') };
}
