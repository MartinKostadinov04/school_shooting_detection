/**
 * Global incident / notification / comms store (Zustand).
 * Framework-agnostic — easy to lift into another React project.
 */

import { create } from "zustand";
import type {
  ChatMessage,
  ConnectionState,
  Device,
  Incident,
  IncidentSeverity,
  IncidentSource,
  IncidentStatus,
  IncidentType,
  Notification,
} from "@/types";
import { seedDevices, seedIncidents, seedMessages } from "./mockData";

function pad(n: number, width = 2) {
  return String(n).padStart(width, "0");
}

function todayPrefix() {
  const d = new Date();
  return `INC-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
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
}

export const useStore = create<StoreState>((set, get) => ({
  devices: seedDevices,
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
      message: `${source} detected ${incident.type.toLowerCase()} at ${location}`,
    });
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
      return { incidents };
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
    return incident;
  },

  setIncidentStatus: (id, status) => {
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
  },

  sendMessage: (msg) => {
    set((state) => ({
      messages: [
        ...state.messages,
        { ...msg, id: `m-${Date.now()}`, timestamp: new Date().toISOString() },
      ],
    }));
  },

  setDeviceStatus: (id, status) => {
    set((state) => ({
      devices: state.devices.map((d) => (d.id === id ? { ...d, status } : d)),
    }));
  },
}));
