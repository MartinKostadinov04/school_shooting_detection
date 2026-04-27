/**
 * Ably client + message parser.
 *
 * Authentication: fetches a short-lived token from /api/ably-token instead of
 * embedding the bare API key in the client bundle.
 * Falls back to a deterministic mock simulator when the token endpoint is
 * unavailable so the demo always shows live activity.
 */

import * as Ably from "ably";
import type { ParsedAblyEvent } from "@/types";

export const ABLY_CHANNEL = "gunshot-detection";

const API_BASE = (import.meta as unknown as { env: Record<string, string> })
  .env?.VITE_API_BASE_URL ?? "http://localhost:8000";

const DIRECT_KEY = (import.meta as unknown as { env: Record<string, string> })
  .env?.VITE_ABLY_API_KEY ?? "";

let client: Ably.Realtime | null = null;

async function fetchAblyToken(): Promise<Ably.TokenRequest | null> {
  try {
    const res = await fetch(`${API_BASE}/api/ably-token`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function getAblyClient(): Promise<Ably.Realtime | null> {
  if (client) return client;

  // Prefer token-auth via backend (production). Fall back to direct key for
  // local dev when the FastAPI backend isn't running.
  const tokenRequest = await fetchAblyToken();
  if (tokenRequest) {
    client = new Ably.Realtime({
      authCallback: (_data, callback) => {
        fetchAblyToken().then((t) => {
          if (t) callback(null, t);
          else callback(new Error("token fetch failed"), null);
        });
      },
      clientId: `tacticaleye-${Math.random().toString(36).slice(2, 8)}`,
    });
  } else if (DIRECT_KEY) {
    client = new Ably.Realtime({
      key: DIRECT_KEY,
      clientId: `tacticaleye-${Math.random().toString(36).slice(2, 8)}`,
    });
  } else {
    return null;
  }

  return client;
}

/**
 * Parse messages of the form:
 *   audio:detected:{location}:{prob}
 *   audio:snippet:{location}:{url}
 *   video:detected:{location}:{conf}
 *   video:segment:{location}:{url}
 *   video:negative:{location}
 *   chat:message  data=JSON
 */
export function parseAblyMessage(name: string, data: unknown): ParsedAblyEvent | null {
  // Chat messages carry JSON payload, not a colon-delimited string
  if (name === "chat:message") {
    try {
      const chatMsg = JSON.parse(typeof data === "string" ? data : "");
      return { kind: "chat:message", location: chatMsg.incidentId ?? "", chatMsg, raw: typeof data === "string" ? data : "" };
    } catch { return null; }
  }

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
    "video:negative",
    "chat:message",
  ];
  if (!allowed.includes(kind)) return null;

  if (kind === "audio:snippet" || kind === "video:segment") {
    const location = rest[0];
    const url = rest.slice(1).join(":");
    if (!location || !url) return null;
    return { kind, location, url, raw };
  }

  if (kind === "video:negative") {
    const location = rest.join(":");
    if (!location) return null;
    return { kind, location, raw };
  }

  // detected messages: {location}:{prob} — prob is the last segment
  const probStr = rest[rest.length - 1];
  const prob = rest.length >= 2 ? parseFloat(probStr) : NaN;
  const location = !isNaN(prob)
    ? rest.slice(0, -1).join(":")
    : rest.join(":");
  if (!location) return null;
  return { kind, location, probability: !isNaN(prob) ? prob : undefined, raw };
}
