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
  const tokenRequest = await fetchAblyToken();
  if (!tokenRequest) return null;
  client = new Ably.Realtime({
    authCallback: (_data, callback) => {
      // Re-fetch token on each renewal cycle
      fetchAblyToken().then((t) => {
        if (t) callback(null, t);
        else callback(new Error("token fetch failed"), null);
      });
    },
    clientId: `tacticaleye-${Math.random().toString(36).slice(2, 8)}`,
  });
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
    const location = rest[0];
    const url = rest.slice(1).join(":");
    if (!location || !url) return null;
    return { kind, location, url, raw };
  }
  const location = rest.join(":");
  if (!location) return null;
  return { kind, location, raw };
}
