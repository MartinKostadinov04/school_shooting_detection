/**
 * Mock seed data for TacticalEye demo.
 * REPLACE: in production, devices come from a backend inventory API and
 * incidents come from your detection pipeline + persistent store.
 *
 * NOTE on coordinates: `x`/`y` are percentages (0-100) over the floor plan
 * image in `src/assets/school-floorplan.png`. When porting to a real school,
 * either re-measure pin positions against the new floor plan, or store
 * positions in the backend keyed by `deviceId`.
 */

import type { Device, Incident, ChatMessage } from "@/types";

export const LOCATIONS = [
  "Main Entrance",
  "1st Floor Hallway",
  "Cafeteria",
  "Gymnasium",
  "Stage",
  "Classroom Wing",
  "Science Lab",
  "Computer Lab",
  "Offices",
] as const;

export const seedDevices: Device[] = [
  // Main Entrance / front offices
  {
    id: "CAM-EN-01",
    name: "Entrance Cam",
    type: "camera",
    location: "Main Entrance",
    status: "online",
    x: 33,
    y: 92,
    lastSeen: new Date().toISOString(),
  },
  {
    id: "MIC-EN-01",
    name: "Entrance Mic",
    type: "microphone",
    location: "Main Entrance",
    status: "online",
    x: 33,
    y: 80,
    lastSeen: new Date().toISOString(),
  },

  // 1st Floor Hallway
  {
    id: "CAM-HW-01",
    name: "Hallway Cam West",
    type: "camera",
    location: "1st Floor Hallway",
    status: "online",
    x: 22,
    y: 64,
    lastSeen: new Date().toISOString(),
  },
  {
    id: "CAM-HW-02",
    name: "Hallway Cam East",
    type: "camera",
    location: "1st Floor Hallway",
    status: "warning",
    x: 65,
    y: 64,
    lastEvent: "Loud noise detected 11m ago",
    lastSeen: new Date().toISOString(),
  },

  // Cafeteria
  {
    id: "CAM-CF-01",
    name: "Cafeteria Cam",
    type: "camera",
    location: "Cafeteria",
    status: "online",
    x: 50,
    y: 10,
    lastSeen: new Date().toISOString(),
  },
  {
    id: "MIC-CF-01",
    name: "Cafeteria Mic",
    type: "microphone",
    location: "Cafeteria",
    status: "online",
    x: 58,
    y: 14,
    lastSeen: new Date().toISOString(),
  },

  // Gymnasium
  {
    id: "CAM-GY-01",
    name: "Gym Cam Court",
    type: "camera",
    location: "Gymnasium",
    status: "triggered",
    x: 28,
    y: 42,
    lastEvent: "Audio anomaly — possible gunshot",
    lastSeen: new Date().toISOString(),
  },
  {
    id: "MIC-GY-01",
    name: "Gym Mic",
    type: "microphone",
    location: "Gymnasium",
    status: "triggered",
    x: 35,
    y: 50,
    lastEvent: "Gunshot probability 0.92",
    lastSeen: new Date().toISOString(),
  },
  {
    id: "CAM-ST-01",
    name: "Stage Cam",
    type: "camera",
    location: "Stage",
    status: "online",
    x: 33,
    y: 27,
    lastSeen: new Date().toISOString(),
  },

  // Classroom Wing
  {
    id: "CAM-CR-01",
    name: "Classroom Cam N",
    type: "camera",
    location: "Classroom Wing",
    status: "online",
    x: 73,
    y: 28,
    lastSeen: new Date().toISOString(),
  },
  {
    id: "CAM-CR-02",
    name: "Classroom Cam S",
    type: "camera",
    location: "Classroom Wing",
    status: "online",
    x: 73,
    y: 52,
    lastSeen: new Date().toISOString(),
  },

  // Science / Computer Lab
  {
    id: "CAM-SL-01",
    name: "Science Lab Cam",
    type: "camera",
    location: "Science Lab",
    status: "online",
    x: 53,
    y: 78,
    lastSeen: new Date().toISOString(),
  },
  {
    id: "CAM-CL-01",
    name: "Computer Lab Cam",
    type: "camera",
    location: "Computer Lab",
    status: "online",
    x: 84,
    y: 78,
    lastSeen: new Date().toISOString(),
  },
  {
    id: "MIC-CL-01",
    name: "Computer Lab Mic",
    type: "microphone",
    location: "Computer Lab",
    status: "online",
    x: 88,
    y: 78,
    lastSeen: new Date().toISOString(),
  },

  // Offices
  {
    id: "CAM-OF-01",
    name: "Offices Cam",
    type: "camera",
    location: "Offices",
    status: "online",
    x: 17,
    y: 82,
    lastSeen: new Date().toISOString(),
  },
];

const now = Date.now();
const min = 60_000;
const hour = 60 * min;
const day = 24 * hour;

export const seedIncidents: Incident[] = [
  {
    id: "INC-20260421-004",
    createdAt: new Date(now - 4 * min).toISOString(),
    location: "Gymnasium",
    type: "Gunshot",
    source: "AUDIO-AI",
    status: "NEW",
    severity: "Critical",
    probability: 0.92,
    // TRANSFER: in production, `audioUrl` is populated by the `audio:snippet`
    // Ably event published by the detection worker (signed URL pointing at
    // the captured snippet in object storage). Left undefined here so the UI
    // shows the "Awaiting snippet…" placeholder instead of demo audio.
    timeline: [
      {
        id: "t1",
        timestamp: new Date(now - 4 * min).toISOString(),
        label: "AUDIO-AI detection",
        detail: "Gunshot probability 0.92 at Gymnasium MIC-GY-01",
      },
      {
        id: "t2",
        timestamp: new Date(now - 3 * min).toISOString(),
        label: "Snippet attached",
        detail: "12s audio snippet captured",
      },
    ],
  },
  {
    id: "INC-20260421-003",
    createdAt: new Date(now - 2 * hour).toISOString(),
    location: "Cafeteria",
    type: "Suspicious Activity",
    source: "MANUAL",
    status: "ACKNOWLEDGED",
    severity: "High",
    description: "Two unidentified individuals near west entrance.",
    reportedBy: "school@demo.com",
    timeline: [
      {
        id: "t1",
        timestamp: new Date(now - 2 * hour).toISOString(),
        label: "Manual report",
        detail: "School staff filed structured report",
      },
      {
        id: "t2",
        timestamp: new Date(now - 2 * hour + 4 * min).toISOString(),
        label: "Acknowledged",
        detail: "Dispatcher acknowledged",
      },
    ],
  },
  {
    id: "INC-20260420-002",
    createdAt: new Date(now - day).toISOString(),
    location: "Main Entrance",
    type: "Gunshot",
    source: "AUDIO-AI",
    status: "RESOLVED",
    severity: "Medium",
    probability: 0.61,
    timeline: [
      {
        id: "t1",
        timestamp: new Date(now - day).toISOString(),
        label: "AUDIO-AI detection",
        detail: "False positive — fireworks",
      },
      {
        id: "t2",
        timestamp: new Date(now - day + 8 * min).toISOString(),
        label: "Resolved",
        detail: "Marked false positive after on-site verification",
      },
    ],
  },
  {
    id: "INC-20260419-001",
    createdAt: new Date(now - 2 * day).toISOString(),
    location: "Cafeteria",
    type: "Fire",
    source: "MANUAL",
    status: "RESOLVED",
    severity: "Low",
    description: "Smoke from kitchen — false alarm.",
    reportedBy: "school@demo.com",
    timeline: [
      {
        id: "t1",
        timestamp: new Date(now - 2 * day).toISOString(),
        label: "Manual report",
      },
      {
        id: "t2",
        timestamp: new Date(now - 2 * day + 12 * min).toISOString(),
        label: "Resolved",
      },
    ],
  },
  {
    id: "INC-20260418-001",
    createdAt: new Date(now - 3 * day).toISOString(),
    location: "Gymnasium",
    type: "Medical",
    source: "MANUAL",
    status: "RESOLVED",
    severity: "Medium",
    description: "Student injury during practice.",
    reportedBy: "school@demo.com",
    timeline: [
      {
        id: "t1",
        timestamp: new Date(now - 3 * day).toISOString(),
        label: "Manual report",
      },
      {
        id: "t2",
        timestamp: new Date(now - 3 * day + 22 * min).toISOString(),
        label: "Resolved",
      },
    ],
  },
];

export const seedMessages: ChatMessage[] = [
  {
    id: "m1",
    timestamp: new Date(now - 6 * min).toISOString(),
    sender: "system",
    text: "Secure channel established. School ↔ Police.",
  },
  {
    id: "m2",
    timestamp: new Date(now - 4 * min).toISOString(),
    sender: "school",
    text: "Hearing loud bangs near the gym. Pulling up cameras now.",
  },
  {
    id: "m3",
    timestamp: new Date(now - 3 * min).toISOString(),
    sender: "police",
    text: "Acknowledged. Units en route. Keep students sheltered.",
  },
];
