/**
 * TacticalEye — shared TypeScript types.
 * Every cross-component contract lives here.
 */

export type Role = "school" | "police";

export interface AuthUser {
  email: string;
  role: Role;
  displayName: string;
}

// ---------- Devices ----------
export type DeviceType = "camera" | "microphone";
export type DeviceStatus = "online" | "warning" | "triggered" | "offline";

export interface Device {
  id: string;
  name: string;
  type: DeviceType;
  location: string;
  status: DeviceStatus;
  /** Position on the SVG map, percent of viewBox 0-100 */
  x: number;
  y: number;
  lastEvent?: string;
  lastSeen: string; // ISO
  feedUrl?: string;
}

// ---------- Incidents ----------
export type IncidentType =
  | "Gunshot"
  | "Suspicious Activity"
  | "Fire"
  | "Medical"
  | "Other";

export type IncidentSource = "AUDIO-AI" | "VIDEO-AI" | "MANUAL";
export type IncidentStatus = "NEW" | "ACKNOWLEDGED" | "RESOLVED";
export type IncidentSeverity = "Low" | "Medium" | "High" | "Critical";

export interface IncidentTimelineEntry {
  id: string;
  timestamp: string; // ISO
  label: string;
  detail?: string;
}

export interface Incident {
  id: string; // INC-YYYYMMDD-NNN
  createdAt: string; // ISO
  location: string;
  type: IncidentType;
  source: IncidentSource;
  status: IncidentStatus;
  severity: IncidentSeverity;
  description?: string;
  probability?: number; // 0-1
  audioUrl?: string;
  videoUrl?: string;
  videoConfirmed?: boolean;
  reportedBy?: string;
  timeline: IncidentTimelineEntry[];
}

// ---------- Notifications ----------
export interface Notification {
  id: string;
  timestamp: string;
  deviceType: DeviceType;
  location: string;
  probability?: number;
  audioUrl?: string;
  videoUrl?: string;
  source: IncidentSource;
  incidentId?: string;
  message: string;
}

// ---------- Comms ----------
export interface ChatMessage {
  id: string;
  timestamp: string;
  sender: Role | "system";
  text?: string;
  /** When present, render as a structured incident report card */
  incidentReport?: {
    location: string;
    type: IncidentType;
    severity: IncidentSeverity;
    description: string;
  };
  incidentId?: string;
}

// ---------- Ably ----------
export type AblyMessageName =
  | "audio:detected"
  | "audio:snippet"
  | "video:detected"
  | "video:segment";

export interface ParsedAblyEvent {
  kind: AblyMessageName;
  location: string;
  url?: string;
  raw: string;
}

export type ConnectionState =
  | "connected"
  | "connecting"
  | "disconnected"
  | "mock";
