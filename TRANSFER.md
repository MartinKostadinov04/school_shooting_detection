# TacticalEye — Transfer Guide

## 1. What this app is

**TacticalEye** is a school-safety command center frontend with two role-based
views:

- **School console** — live device map (cameras + microphones), real-time
  notification feed, device inventory table, and a floating police-comms window
  with both free-text chat and a structured incident-report form.
- **Police dispatch console** — incident feed (newest first) with auto-IDs
  (`INC-YYYYMMDD-NNN`), source/severity/status badges, plus an active-incident
  detail panel containing audio/video evidence, chronological timeline, and a
  per-incident comms thread back to the school.

Real-time detections arrive over an Ably channel and update both consoles
live. The whole UI is dark-tactical (amber / red / cyan / green signals,
monospace IDs and timestamps).

This is a **frontend-only prototype**. Auth, persistence, and detection are
mocked so the app always runs standalone, but every mock is isolated so you
can swap it for real infrastructure.

## 2. Run locally

```bash
npm install
npm run dev
```

### Environment variables

| Variable             | Required | Purpose                                            |
| -------------------- | -------- | -------------------------------------------------- |
| `VITE_ABLY_API_KEY`  | No       | Ably realtime key. If absent, app runs in mock mode and self-emits demo detections every ~25s. |

### Demo accounts

| Email              | Password    | Lands on   |
| ------------------ | ----------- | ---------- |
| `school@demo.com`  | `school123` | `/school`  |
| `police@demo.com`  | `police123` | `/police`  |

## 3. Connecting a real backend

Every mock/hardcoded piece is listed below with its file and what to swap in:

| Concern              | File                                | Replace with                                                                                  |
| -------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------- |
| Auth (login + session) | `src/hooks/useAuth.ts`            | Real auth provider (Supabase Auth, Auth0, Clerk, custom JWT). Keep the `useAuth()` shape: `{ user, ready, login, logout }`. |
| Demo accounts table  | `src/hooks/useAuth.ts` `ACCOUNTS`   | Delete entirely once real auth is wired.                                                      |
| Device inventory     | `src/lib/mockData.ts` `seedDevices` | Fetch from your devices API; keep the `Device` type from `src/types`.                          |
| Historical incidents | `src/lib/mockData.ts` `seedIncidents` | Fetch from your incidents API.                                                              |
| Seed messages        | `src/lib/mockData.ts` `seedMessages` | Replace with backend-persisted thread.                                                       |
| Mock Ably simulator  | `src/hooks/useAbly.ts` (the `if (!apiKey)` branch) | Delete once you ship with `VITE_ABLY_API_KEY`. |
| Audio snippet URL    | `src/hooks/useAbly.ts` `MOCK_AUDIO_URL` and `src/lib/mockData.ts` (initial incident `audioUrl`) | Real signed media URLs from your storage bucket. |
| Mock live device feed | `src/components/map/DevicePanel.tsx` (placeholder block) | Real WebRTC/HLS/MJPEG feed. |
| Device "View Feed" link | `src/components/devices/DeviceTable.tsx` (`href="#"`) | Real per-device feed URL. |
| Incident store       | `src/lib/incidentStore.ts`          | Either back the Zustand store with API calls (in `ingestDetection`, `setIncidentStatus`, etc.) or replace the store entirely with TanStack Query against your REST/GraphQL endpoints. The store API surface is intentionally narrow. |

The store actions (`ingestDetection`, `attachMedia`, `markVideoConfirmed`,
`reportManualIncident`, `setIncidentStatus`, `dispatchUnit`, `sendMessage`)
are the single integration boundary for real data.

## 4. Design system

`src/config/design.ts` is the **only file you need to edit to retheme** the
app. It exports a `design` object with:

- color references (mirror of CSS custom properties in `src/styles.css`)
- typography (sans + mono fonts)
- layout constants (sidebar width, panel widths, map height)
- motion tokens
- severity / status / source / device-status / connection style maps
- incident type & severity option lists

If you change a color value in `design.ts`, also change the matching token in
`src/styles.css` `:root` (these are the actual CSS values Tailwind reads). No
component hardcodes colors — they all consume tokens from `design.ts` or
Tailwind utilities like `text-tactical-amber` that resolve to those tokens.

## 5. Component map

```
src/
├── config/design.ts                       single source of truth for visuals
├── types/index.ts                         all shared TypeScript types
├── lib/
│   ├── ably.ts                            Ably singleton + parseAblyMessage()
│   ├── incidentStore.ts                   Zustand store: incidents, notifications, messages, devices
│   └── mockData.ts                        seed devices, incidents, messages
├── hooks/
│   ├── useAuth.ts                         localStorage-backed mock auth
│   └── useAbly.ts                         channel subscriber + mock simulator fallback
├── components/
│   ├── SchoolSidebar.tsx                  collapsible nav for the school console
│   ├── ui/
│   │   ├── ConnectionIndicator.tsx        live/offline/mock pill
│   │   └── StatusBadges.tsx               StatusPill, SourceBadge, SeverityBadge
│   ├── map/
│   │   ├── SchoolMap.tsx                  zoom/pan SVG map of the school
│   │   ├── DevicePin.tsx                  pin with status glow
│   │   └── DevicePanel.tsx                side panel: device info + mock feed
│   ├── devices/DeviceTable.tsx            inventory table with View Feed / Flag
│   ├── notifications/
│   │   ├── NotificationBar.tsx            scrollable live alert feed
│   │   └── NotificationItem.tsx           single alert card with audio/video expansion
│   ├── comms/
│   │   ├── CommunicationWindow.tsx        floating (school) / inline (police) chat panel
│   │   ├── MessageBubble.tsx              chat + structured incident-report rendering
│   │   └── IncidentReportForm.tsx         manual incident reporter
│   ├── incidents/IncidentCard.tsx         feed card with ACK / Resolve / Dispatch
│   └── police/
│       ├── PoliceIncidentFeed.tsx         filterable list (ALL / NEW / ACK / RESOLVED)
│       └── ActiveIncidentPanel.tsx        header, audio + video evidence, timeline, comms
└── routes/
    ├── __root.tsx                         html shell
    ├── index.tsx                          role-based redirect to /school or /police
    ├── login.tsx                          mock login screen
    ├── school.tsx                         school console layout
    └── police.tsx                         dispatch console layout
```

## 6. Ably channel contract

Subscribed channel: **`gunshot-detection`** (constant in `src/lib/ably.ts`).

| Message data format                       | Action                                                          |
| ----------------------------------------- | --------------------------------------------------------------- |
| `audio:detected:{location}`               | Create new incident (source = AUDIO-AI), fire notification.     |
| `audio:snippet:{location}:{url}`          | Attach audio player to most recent matching open incident.      |
| `video:detected:{location}`               | Mark matching open incident video-confirmed, or create a new VIDEO-AI incident if none. |
| `video:segment:{location}:{url}`          | Attach video player to most recent matching open incident.      |

Parsing is centralized in `parseAblyMessage(name, data)` — it accepts the
payload either as the message `name` or as a string `data`. URLs are allowed
to contain `:` (e.g. `https://`).
