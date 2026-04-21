/**
 * Ably client + message parser.
 *
 * No component imports `ably` directly — they use `useAbly()` hook.
 * Falls back to a deterministic mock simulator when VITE_ABLY_API_KEY is absent
 * so the demo always shows live activity.
 */

import * as Ably from "ably";
import type { ParsedAblyEvent } from "@/types";

export const ABLY_CHANNEL = "gunshot-detection";

export function getAblyApiKey(): string | undefined {
  const key = (import.meta as unknown as { env: Record<string, string | undefined> })
    .env?.VITE_ABLY_API_KEY;
  return key && key.length > 0 ? key : undefined;
}

let client: Ably.Realtime | null = null;

export function getAblyClient(): Ably.Realtime | null {
  const key = getAblyApiKey();
  if (!key) return null;
  if (client) return client;
  client = new Ably.Realtime({ key, clientId: `tacticaleye-${Math.random().toString(36).slice(2, 8)}` });
  return client;
}

/**
 * Parse messages of the form:
 *   audio:detected:{location}
 *   audio:snippet:{location}:{url}
 *   video:detected:{location}
 *   video:segment:{location}:{url}
 */
export function parseAblyMessage(name: string, data: unknown): ParsedAblyEvent | null {
  const raw = typeof data === "string" ? data : name;
  if (typeof raw !== "string") return null;
  const parts = raw.split(":");
  if (parts.length < 3) return null;
  const [media, action, ...rest] = parts;
  const kind = `${media}:${action}` as ParsedAblyEvent["kind"];
  const allowed = [
    "audio:detected",
    "audio:snippet",
    "video:detected",
    "video:segment",
  ];
  if (!allowed.includes(kind)) return null;

  if (kind === "audio:snippet" || kind === "video:segment") {
    // location is the next chunk; URL may itself contain ':' (e.g. https://)
    const location = rest[0];
    const url = rest.slice(1).join(":");
    if (!location || !url) return null;
    return { kind, location, url, raw };
  }
  const location = rest.join(":");
  if (!location) return null;
  return { kind, location, raw };
}
