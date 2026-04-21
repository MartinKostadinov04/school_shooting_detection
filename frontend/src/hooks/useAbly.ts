/**
 * Subscribes to the Ably channel `gunshot-detection` and forwards parsed
 * events to the in-memory store.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * TRANSFER NOTES (read before porting to the real codebase)
 * ─────────────────────────────────────────────────────────────────────────────
 * In the production repo (school_shooting_detection) the publishers of these
 * messages are the audio + video detection workers. This hook is the SUBSCRIBER
 * only. There is intentionally NO synthetic / simulated event loop here —
 * if `VITE_ABLY_API_KEY` is missing or the connection fails, the UI shows a
 * `disconnected` indicator and stays empty until real events arrive.
 *
 * Message contract (must match the publishers):
 *   audio:detected   data = `audio:detected:{location}`
 *   audio:snippet    data = `audio:snippet:{location}:{url}`
 *   video:detected   data = `video:detected:{location}`
 *   video:segment    data = `video:segment:{location}:{url}`
 *
 * When wiring the real backend:
 *   1. Set VITE_ABLY_API_KEY (or swap getAblyClient() for a token-auth flow
 *      that hits your /api/ably-token endpoint — recommended for production).
 *   2. Ensure publisher worker emits the message names above on channel
 *      ABLY_CHANNEL ("gunshot-detection").
 *   3. Replace seed data in src/lib/mockData.ts with real device inventory +
 *      historical incidents fetched from your backend on mount.
 * ─────────────────────────────────────────────────────────────────────────────
 */
import { useEffect, useRef } from "react";
import { ABLY_CHANNEL, getAblyApiKey, getAblyClient, parseAblyMessage } from "@/lib/ably";
import { useStore } from "@/lib/incidentStore";
import type { ParsedAblyEvent } from "@/types";

export function useAbly() {
  const setConnection = useStore((s) => s.setConnection);
  const ingestDetection = useStore((s) => s.ingestDetection);
  const attachMedia = useStore((s) => s.attachMedia);
  const markVideoConfirmed = useStore((s) => s.markVideoConfirmed);

  const handlerRef = useRef((evt: ParsedAblyEvent) => {
    if (evt.kind === "audio:detected") {
      ingestDetection({ location: evt.location, source: "AUDIO-AI" });
    } else if (evt.kind === "video:detected") {
      // If matching incident exists, mark video confirmed; else create one.
      const existing = useStore
        .getState()
        .incidents.find((i) => i.location === evt.location && i.status !== "RESOLVED");
      if (existing) markVideoConfirmed(evt.location);
      else ingestDetection({ location: evt.location, source: "VIDEO-AI" });
    } else if (evt.kind === "audio:snippet" && evt.url) {
      attachMedia({ location: evt.location, kind: "audio", url: evt.url });
    } else if (evt.kind === "video:segment" && evt.url) {
      attachMedia({ location: evt.location, kind: "video", url: evt.url });
    }
  });

  useEffect(() => {
    const apiKey = getAblyApiKey();

    if (!apiKey) {
      // No key configured. We deliberately do NOT simulate events here.
      // The dispatch / school UI will simply show "OFFLINE" until the
      // detection backend is wired up (see TRANSFER NOTES at top of file).
      setConnection("disconnected");
      return;
    }

    setConnection("connecting");
    const client = getAblyClient();
    if (!client) {
      setConnection("disconnected");
      return;
    }
    const channel = client.channels.get(ABLY_CHANNEL);

    const onConnected = () => setConnection("connected");
    const onDisconnected = () => setConnection("disconnected");
    client.connection.on("connected", onConnected);
    client.connection.on("disconnected", onDisconnected);
    client.connection.on("failed", onDisconnected);

    const onMessage = (msg: { name?: string; data?: unknown }) => {
      const parsed = parseAblyMessage(msg.name ?? "", msg.data);
      if (parsed) handlerRef.current(parsed);
    };
    channel.subscribe(onMessage);

    return () => {
      channel.unsubscribe(onMessage);
      client.connection.off("connected", onConnected);
      client.connection.off("disconnected", onDisconnected);
      client.connection.off("failed", onDisconnected);
    };
  }, [setConnection]);
}
