/**
 * Global incident / notification / comms store (Zustand).
 * Local state is updated optimistically; every mutation also fires an API call
 * so data persists in the FastAPI backend.
 */

import { create } from "zustand";
import type {
  ChatMessage,
  ConnectionState,
  Device,
  DeviceStatus,
  Incident,
  IncidentSeverity,
  IncidentSource,
  IncidentStatus,
  IncidentType,
  Notification,
} from "@/types";
import { seedDevices, seedIncidents, seedMessages } from "./mockData";
import { POLICE_SCHOOLS } from "./schools";

const API_BASE = (import.meta as unknown as { env: Record<string, string> })
  .env?.VITE_API_BASE_URL ?? "http://localhost:8000";

const SCHOOL_ID = "default";

function pad(n: number, width = 2) {
  return String(n).padStart(width, "0");
}

function todayPrefix() {
  const d = new Date();
  return `INC-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
}

// Fire-and-forget API helpers — errors are logged but never surface to the user
// because local state has already been updated optimistically.
async function apiPost(path: string, body: unknown): Promise<unknown> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) console.warn(`POST ${path} →`, res.status);
    return res.ok ? res.json() : null;
  } catch (e) {
    console.warn(`POST ${path} failed`, e);
    return null;
  }
}

async function apiPatch(path: string, body: unknown): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) console.warn(`PATCH ${path} →`, res.status);
  } catch (e) {
    console.warn(`PATCH ${path} failed`, e);
  }
}

interface StoreState {
  devices: Device[];
  incidents: Incident[];
  notifications: Notification[];
  messages: ChatMessage[];
  connection: ConnectionState;

  // selectors
  nextIncidentId: () => string;

  // actions
  setConnection: (s: ConnectionState) => void;
  loadFromApi: (schoolId?: string) => Promise<void>;

  addNotification: (n: Notification) => void;

  /** Audio-AI or Video-AI detection from Ably */
  ingestDetection: (params: {
    location: string;
    source: Extract<IncidentSource, "AUDIO-AI" | "VIDEO-AI">;
    probability?: number;
  }) => Incident;

  /** Attach a media URL (snippet or segment) to most recent matching incident */
  attachMedia: (params: {
    location: string;
    kind: "audio" | "video";
    url: string;
  }) => void;

  markVideoConfirmed: (location: string) => void;

  reportManualIncident: (params: {
    location: string;
    type: IncidentType;
    severity: IncidentSeverity;
    description: string;
    reportedBy: string;
  }) => Incident;

  setIncidentStatus: (id: string, status: IncidentStatus) => void;
  dispatchUnit: (id: string) => void;

  sendMessage: (msg: Omit<ChatMessage, "id" | "timestamp">) => void;
  setDeviceStatus: (id: string, status: Device["status"]) => void;
  setDevicePosition: (id: string, x: number, y: number) => void;
  resetDevicePositions: () => void;
}

const LS_POSITIONS_KEY = "te-device-positions";

function savedPositions(): Record<string, { x: number; y: number }> {
  try { return JSON.parse(localStorage.getItem(LS_POSITIONS_KEY) ?? "{}"); }
  catch { return {}; }
}

function applyPositions(devices: Device[]): Device[] {
  const saved = savedPositions();
  return devices.map((d) => (saved[d.id] ? { ...d, x: saved[d.id].x, y: saved[d.id].y } : d));
}

export const useStore = create<StoreState>((set, get) => ({
  devices: applyPositions(seedDevices),
  incidents: [...seedIncidents].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  ),
  notifications: [],
  messages: seedMessages,
  connection: "connecting",

  nextIncidentId: () => {
    const prefix = todayPrefix();
    const same = get().incidents.filter((i) => i.id.startsWith(prefix));
    return `${prefix}-${pad(same.length + 1, 3)}`;
  },

  setConnection: (s) => set({ connection: s }),

  loadFromApi: async (schoolId = SCHOOL_ID) => {
    try {
      const [devRes, incRes] = await Promise.all([
        fetch(`${API_BASE}/api/schools/${schoolId}/devices`),
        fetch(`${API_BASE}/api/schools/${schoolId}/incidents`),
      ]);
      if (devRes.ok) {
        const devs: Device[] = await devRes.json();
        set({ devices: applyPositions(devs) });
      }
      if (incRes.ok) {
        const incs: Incident[] = await incRes.json();
        set({ incidents: incs });
      }
    } catch (e) {
      console.warn("loadFromApi failed — using seed data", e);
    }
  },

  addNotification: (n) =>
    set((state) => ({ notifications: [n, ...state.notifications].slice(0, 50) })),

  ingestDetection: ({ location, source, probability }) => {
    const id = get().nextIncidentId();
    const now = new Date().toISOString();
    const incident: Incident = {
      id,
      createdAt: now,
      location,
      type: "Gunshot",
      source,
      status: "NEW",
      severity: "Critical",
      probability: probability ?? (source === "AUDIO-AI" ? 0.88 : 0.81),
      timeline: [
        {
          id: `${id}-t1`,
          timestamp: now,
          label: `${source} detection`,
          detail: `Detection at ${location}`,
        },
      ],
    };
    set((state) => ({ incidents: [incident, ...state.incidents] }));

    get().addNotification({
      id: `n-${id}`,
      timestamp: now,
      deviceType: source === "AUDIO-AI" ? "microphone" : "camera",
      location,
      probability: incident.probability,
      source,
      incidentId: id,
      message: `${source} detected ${incident.type.toLowerCase()} at ${location} — prob ${(incident.probability ?? 0).toFixed(2)}`,
    });

    // Persist to backend (fire-and-forget)
    apiPost("/api/incidents", {
      school_id: SCHOOL_ID,
      location,
      type: "Gunshot",
      source,
      severity: "Critical",
      probability: incident.probability,
    });

    // Trigger the matching sensor type at the detected location
    const sensorType = source === "AUDIO-AI" ? "microphone" : "camera";
    const eventLabel =
      source === "AUDIO-AI"
        ? `Gunshot probability ${(incident.probability ?? 0).toFixed(2)}`
        : "Gun detected by VIDEO-AI";
    const triggeredAt = now;
    set((state) => ({
      devices: state.devices.map((d) =>
        d.location === location && d.type === sensorType
          ? { ...d, status: "triggered" as DeviceStatus, lastEvent: eventLabel, lastSeen: triggeredAt }
          : d,
      ),
    }));
    get()
      .devices.filter((d) => d.location === location && d.type === sensorType)
      .forEach((d) => apiPatch(`/api/devices/${d.id}/status`, { status: "triggered" }));

    // Auto-alert to police channel on audio gunshot detection
    if (source === "AUDIO-AI") {
      const school = POLICE_SCHOOLS[0];
      const mic = get().devices.find(
        (d) => d.location === location && d.type === "microphone",
      );
      const timeStr = new Date(now).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
      get().sendMessage({
        sender: "system",
        incidentId: id,
        text: [
          "🔴 AUTOMATED ALERT — Gunshot Detected",
          `Time:        ${timeStr}`,
          `School:      ${school.name}`,
          `Address:     ${school.address}`,
          `Sensor:      ${mic ? `${mic.name} (${mic.id})` : `unknown — ${location}`}`,
          `Probability: ${(incident.probability ?? 0).toFixed(2)}`,
        ].join("\n"),
      });
    }

    return incident;
  },

  attachMedia: ({ location, kind, url }) => {
    set((state) => {
      const incidents = [...state.incidents];
      const idx = incidents.findIndex(
        (i) => i.location === location && i.status !== "RESOLVED",
      );
      if (idx === -1) return state;
      const inc = incidents[idx];
      const updated: Incident = {
        ...inc,
        audioUrl: kind === "audio" ? url : inc.audioUrl,
        videoUrl: kind === "video" ? url : inc.videoUrl,
        timeline: [
          ...inc.timeline,
          {
            id: `${inc.id}-media-${Date.now()}`,
            timestamp: new Date().toISOString(),
            label: kind === "audio" ? "Audio snippet attached" : "Video segment attached",
            detail: url,
          },
        ],
      };
      incidents[idx] = updated;

      // Persist to backend
      apiPatch(`/api/incidents/${inc.id}`, {
        [kind === "audio" ? "audio_url" : "video_url"]: url,
      });

      const notifications = state.notifications.map((n) =>
        n.incidentId === inc.id
          ? {
              ...n,
              audioUrl: kind === "audio" ? url : n.audioUrl,
              videoUrl: kind === "video" ? url : n.videoUrl,
            }
          : n,
      );
      return { incidents, notifications };
    });
  },

  markVideoConfirmed: (location) => {
    set((state) => {
      const incidents = [...state.incidents];
      const idx = incidents.findIndex(
        (i) => i.location === location && i.status !== "RESOLVED",
      );
      if (idx === -1) return state;
      const inc = incidents[idx];
      incidents[idx] = {
        ...inc,
        videoConfirmed: true,
        timeline: [
          ...inc.timeline,
          {
            id: `${inc.id}-vc-${Date.now()}`,
            timestamp: new Date().toISOString(),
            label: "Video AI confirmed",
            detail: `Visual confirmation at ${location}`,
          },
        ],
      };

      apiPatch(`/api/incidents/${inc.id}`, { video_confirmed: true });

      // Also trigger cameras at this location (video confirmation = gun sighted)
      const ts = new Date().toISOString();
      const updatedDevices = state.devices.map((d) =>
        d.location === location && d.type === "camera"
          ? { ...d, status: "triggered" as DeviceStatus, lastEvent: "Gun detected by VIDEO-AI", lastSeen: ts }
          : d,
      );
      state.devices
        .filter((d) => d.location === location && d.type === "camera")
        .forEach((d) => apiPatch(`/api/devices/${d.id}/status`, { status: "triggered" }));

      return { incidents, devices: updatedDevices };
    });
  },

  reportManualIncident: ({ location, type, severity, description, reportedBy }) => {
    const id = get().nextIncidentId();
    const now = new Date().toISOString();
    const incident: Incident = {
      id,
      createdAt: now,
      location,
      type,
      source: "MANUAL",
      status: "NEW",
      severity,
      description,
      reportedBy,
      timeline: [
        {
          id: `${id}-t1`,
          timestamp: now,
          label: "Manual report filed",
          detail: `${reportedBy} → ${type} (${severity})`,
        },
      ],
    };
    set((state) => ({ incidents: [incident, ...state.incidents] }));

    apiPost("/api/incidents", {
      school_id: SCHOOL_ID,
      location,
      type,
      source: "MANUAL",
      severity,
      description,
      reported_by: reportedBy,
    });

    return incident;
  },

  setIncidentStatus: (id, status) => {
    // Reset triggered devices when an incident is resolved
    if (status === "RESOLVED") {
      const incident = get().incidents.find((i) => i.id === id);
      if (incident) {
        const toReset = get().devices.filter(
          (d) => d.location === incident.location && d.status === "triggered",
        );
        if (toReset.length) {
          set((state) => ({
            devices: state.devices.map((d) =>
              d.location === incident.location && d.status === "triggered"
                ? { ...d, status: "online" as DeviceStatus }
                : d,
            ),
          }));
          toReset.forEach((d) => apiPatch(`/api/devices/${d.id}/status`, { status: "online" }));
        }
      }
    }

    set((state) => ({
      incidents: state.incidents.map((i) =>
        i.id === id
          ? {
              ...i,
              status,
              timeline: [
                ...i.timeline,
                {
                  id: `${i.id}-s-${Date.now()}`,
                  timestamp: new Date().toISOString(),
                  label: `Status → ${status}`,
                },
              ],
            }
          : i,
      ),
    }));
    apiPatch(`/api/incidents/${id}`, { status });
  },

  dispatchUnit: (id) => {
    set((state) => ({
      incidents: state.incidents.map((i) =>
        i.id === id
          ? {
              ...i,
              timeline: [
                ...i.timeline,
                {
                  id: `${i.id}-d-${Date.now()}`,
                  timestamp: new Date().toISOString(),
                  label: "Unit dispatched",
                  detail: "Patrol unit en route",
                },
              ],
            }
          : i,
      ),
    }));
    apiPost(`/api/incidents/${id}/dispatch`, {});
  },

  sendMessage: (msg) => {
    const incidentId = msg.incidentId;
    set((state) => ({
      messages: [
        ...state.messages,
        { ...msg, id: `m-${Date.now()}`, timestamp: new Date().toISOString() },
      ],
    }));
    if (incidentId) {
      apiPost(`/api/incidents/${incidentId}/messages`, {
        sender: msg.sender,
        text: msg.text,
        incidentReport: msg.incidentReport,
      });
    }
  },

  setDeviceStatus: (id, status) => {
    set((state) => ({
      devices: state.devices.map((d) => (d.id === id ? { ...d, status } : d)),
    }));
    apiPatch(`/api/devices/${id}/status`, { status });
  },

  setDevicePosition: (id, x, y) => {
    set((state) => ({
      devices: state.devices.map((d) => (d.id === id ? { ...d, x, y } : d)),
    }));
    const saved = savedPositions();
    saved[id] = { x, y };
    localStorage.setItem(LS_POSITIONS_KEY, JSON.stringify(saved));
  },

  resetDevicePositions: () => {
    localStorage.removeItem(LS_POSITIONS_KEY);
    set({ devices: seedDevices });
  },
}));
