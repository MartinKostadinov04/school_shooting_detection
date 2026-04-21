
# TacticalEye — School Safety Command Center

A dark, ops-room style dashboard with two role-based views (School & Police), real-time Ably integration, and a fully portable architecture designed to be lifted into the existing detection repo.

## Auth & Routing
- Mock login screen (localStorage session) with two demo accounts:
  - `school@demo.com / school123` → `/school`
  - `police@demo.com / police123` → `/police`
- Routes: `/login`, `/school`, `/police` — protected by a simple role guard.
- Logout button in top bar.

## School Dashboard (`/school`)
- **Collapsible left sidebar**: logo, Dashboard, Incidents, Communications, Settings.
- **Top-right notification bar**: live scrollable feed of detections — device-type icon, location, timestamp, probability score. Click to expand into audio player or camera thumbnail.
- **Center map**: zoom/pan SVG canvas of the school with 9 device pins (6 cameras, 3 mics) across Main Entrance, Cafeteria, Gymnasium. Pins glow by status (online / warning / triggered). Clicking a pin opens a side panel with device info, mock live feed, last detection log.
- **Device table** below map: Name | Type | Location | Status | Last Event | Last Seen | View Feed | Flag.
- **Floating Police Comms window** (bottom-right, toggleable):
  - Free-text chat thread
  - "Report Incident" form → Location (from devices), Type (Gunshot / Suspicious Activity / Fire / Medical / Other), Description, Severity (Low–Critical) → posts as a structured incident card into the thread.

## Police Dashboard (`/police`)
- **Two-column layout**.
- **Left — Incident Feed**: scrollable cards (newest first). Each shows auto-ID (`INC-YYYYMMDD-NNN`), timestamp, location, type badge, source badge (AUDIO-AI / VIDEO-AI / MANUAL), status pill (NEW / ACKNOWLEDGED / RESOLVED). Inline actions: Acknowledge, Resolve, Dispatch Unit.
- **Right — Active Incident Detail**:
  - Header with ID, severity, location
  - Audio section: waveform placeholder + player + probability score
  - Video section: video player / thumbnail placeholder
  - Chronological event timeline
  - Communication thread with the school for this specific incident
- AI detections render as structured notification cards matching the contract.

## Real-Time (Ably)
- Isolated in `src/lib/ably.ts` — singleton client + `parseAblyMessage()` parser.
- `useAbly` hook subscribes to channel `gunshot-detection` and dispatches to incident store.
- Message contract handled:
  - `audio:detected:{location}` → new incident + notification
  - `audio:snippet:{location}:{url}` → attach audio player
  - `video:detected:{location}` → mark video-confirmed
  - `video:segment:{location}:{url}` → attach video player
- **Connection status indicator** (top-right dot: green connected / red disconnected / amber connecting).
- API key via `VITE_ABLY_API_KEY` (graceful fallback to mock-event simulator if absent so the demo always works).

## Mock Data Seed
- 3 locations (Main Entrance, Cafeteria, Gymnasium), 6 cameras + 3 mics.
- 5 historical incidents (mixed AUDIO-AI / MANUAL, varied statuses).
- 1 active NEW incident with mock audio snippet URL.

## Design System (`src/config/design.ts`)
Single source of truth — no hardcoded design values anywhere else:
- **Palette**: deep charcoal/near-black backgrounds, tactical amber accents, alert red, online green, warning yellow, info cyan.
- **Typography**: monospace for IDs/timestamps/data, geometric sans for UI.
- **Tokens**: severity colors, status colors/labels, source-badge styles, layout constants (sidebar width, panel widths, map height), elevation, radii, motion timings.
- Mapped into Tailwind via CSS custom properties in `styles.css`.

## AI Model Note
The "Use only Claude Opus 4.7" instruction cannot be honored in this build — the Lovable AI Gateway only exposes Google Gemini and OpenAI GPT models; Anthropic Claude is not available. This frontend prototype does not actually call any LLM (it's UI + mock data + Ably wiring), so no AI provider is wired in. If an AI feature is added later, it will use a Gemini or GPT model via the Lovable AI Gateway.

## Architecture (portable)
```text
src/
├── components/
│   ├── map/           SchoolMap, DevicePin, DevicePanel
│   ├── incidents/     IncidentCard, IncidentFeed, IncidentDetail
│   ├── notifications/ NotificationBar, NotificationItem
│   ├── comms/         CommunicationWindow, MessageBubble, IncidentReportForm
│   ├── devices/       DeviceTable, DeviceRow, DeviceStatusBadge
│   ├── police/        PoliceIncidentFeed, ActiveIncidentPanel, MediaPlayer
│   └── ui/            Badge, StatusPill, Card, Modal, ConnectionIndicator
├── config/design.ts
├── hooks/             useAbly, useIncidents, useAuth
├── lib/               ably.ts, mockData.ts, incidentStore.ts
├── routes/            __root, index (redirect), login, school, police
└── types/index.ts     Incident, Device, AblyEvent, Message, Role, etc.
```
- Lightweight Zustand store for incidents, messages, notifications — kept framework-agnostic for easy port.

## Deliverable: `TRANSFER.md`
Root-level handoff doc covering: what the app is, run instructions + env vars, every mock/hardcoded piece with file paths and replacement guidance, retheming via `src/config/design.ts`, one-line component map, and the Ably channel contract table.
