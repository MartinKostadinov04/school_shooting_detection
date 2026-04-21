/**
 * TacticalEye — central design system.
 *
 * SINGLE SOURCE OF TRUTH for every visual / labeling decision.
 * All components import tokens from here; nothing is hardcoded elsewhere.
 *
 * Color tokens here mirror the CSS custom properties defined in src/styles.css.
 * To retheme the entire app, edit this file and the matching values in styles.css.
 */

import type {
  IncidentSeverity,
  IncidentSource,
  IncidentStatus,
  DeviceStatus,
  IncidentType,
} from "@/types";

export const design = {
  app: {
    name: "TacticalEye",
    tagline: "School Safety Command Center",
  },

  /** Reference palette — keep in sync with styles.css */
  palette: {
    background: "var(--background)",
    surface: "var(--surface)",
    surfaceRaised: "var(--surface-raised)",
    foreground: "var(--foreground)",
    muted: "var(--muted)",
    mutedForeground: "var(--muted-foreground)",
    border: "var(--border)",
    primary: "var(--primary)",
    amber: "var(--tactical-amber)",
    red: "var(--tactical-red)",
    green: "var(--tactical-green)",
    yellow: "var(--tactical-yellow)",
    cyan: "var(--tactical-cyan)",
    violet: "var(--tactical-violet)",
  },

  typography: {
    sans: "Inter, ui-sans-serif, system-ui, sans-serif",
    mono: "JetBrains Mono, ui-monospace, Menlo, Consolas, monospace",
  },

  layout: {
    sidebarWidth: "16rem",
    sidebarWidthCollapsed: "3.5rem",
    notificationBarWidth: "22rem",
    incidentFeedWidth: "26rem",
    commsWindowWidth: "24rem",
    commsWindowHeight: "30rem",
    mapHeight: "32rem",
    headerHeight: "3.25rem",
  },

  motion: {
    fast: "120ms",
    base: "200ms",
    slow: "320ms",
    easing: "cubic-bezier(0.22, 1, 0.36, 1)",
  },

  /** Maps + classes for severity */
  severity: {
    Low: {
      label: "LOW",
      textClass: "text-tactical-cyan",
      bgClass: "bg-tactical-cyan/15",
      borderClass: "border-tactical-cyan/40",
      dotClass: "bg-tactical-cyan",
    },
    Medium: {
      label: "MED",
      textClass: "text-tactical-yellow",
      bgClass: "bg-tactical-yellow/15",
      borderClass: "border-tactical-yellow/40",
      dotClass: "bg-tactical-yellow",
    },
    High: {
      label: "HIGH",
      textClass: "text-tactical-amber",
      bgClass: "bg-tactical-amber/15",
      borderClass: "border-tactical-amber/40",
      dotClass: "bg-tactical-amber",
    },
    Critical: {
      label: "CRIT",
      textClass: "text-tactical-red",
      bgClass: "bg-tactical-red/15",
      borderClass: "border-tactical-red/50",
      dotClass: "bg-tactical-red",
    },
  } satisfies Record<
    IncidentSeverity,
    {
      label: string;
      textClass: string;
      bgClass: string;
      borderClass: string;
      dotClass: string;
    }
  >,

  status: {
    NEW: {
      label: "NEW",
      textClass: "text-tactical-red",
      bgClass: "bg-tactical-red/15",
      borderClass: "border-tactical-red/50",
      pulse: true,
    },
    ACKNOWLEDGED: {
      label: "ACK",
      textClass: "text-tactical-amber",
      bgClass: "bg-tactical-amber/15",
      borderClass: "border-tactical-amber/40",
      pulse: false,
    },
    RESOLVED: {
      label: "RESOLVED",
      textClass: "text-tactical-green",
      bgClass: "bg-tactical-green/15",
      borderClass: "border-tactical-green/40",
      pulse: false,
    },
  } satisfies Record<
    IncidentStatus,
    {
      label: string;
      textClass: string;
      bgClass: string;
      borderClass: string;
      pulse: boolean;
    }
  >,

  source: {
    "AUDIO-AI": {
      label: "AUDIO-AI",
      textClass: "text-tactical-cyan",
      bgClass: "bg-tactical-cyan/15",
      borderClass: "border-tactical-cyan/40",
    },
    "VIDEO-AI": {
      label: "VIDEO-AI",
      textClass: "text-tactical-violet",
      bgClass: "bg-tactical-violet/15",
      borderClass: "border-tactical-violet/40",
    },
    MANUAL: {
      label: "MANUAL",
      textClass: "text-foreground",
      bgClass: "bg-muted",
      borderClass: "border-border",
    },
  } satisfies Record<
    IncidentSource,
    { label: string; textClass: string; bgClass: string; borderClass: string }
  >,

  deviceStatus: {
    online: {
      label: "ONLINE",
      textClass: "text-tactical-green",
      dotClass: "bg-tactical-green",
      glowClass: "shadow-[0_0_12px_var(--tactical-green)]",
    },
    warning: {
      label: "WARNING",
      textClass: "text-tactical-yellow",
      dotClass: "bg-tactical-yellow",
      glowClass: "shadow-[0_0_14px_var(--tactical-yellow)]",
    },
    triggered: {
      label: "TRIGGERED",
      textClass: "text-tactical-red",
      dotClass: "bg-tactical-red",
      glowClass: "shadow-[0_0_18px_var(--tactical-red)]",
    },
    offline: {
      label: "OFFLINE",
      textClass: "text-muted-foreground",
      dotClass: "bg-muted-foreground",
      glowClass: "",
    },
  } satisfies Record<
    DeviceStatus,
    { label: string; textClass: string; dotClass: string; glowClass: string }
  >,

  incidentTypes: [
    "Gunshot",
    "Suspicious Activity",
    "Fire",
    "Medical",
    "Other",
  ] satisfies IncidentType[],

  severities: ["Low", "Medium", "High", "Critical"] satisfies IncidentSeverity[],

  connection: {
    connected: {
      label: "LIVE",
      dotClass: "bg-tactical-green",
      textClass: "text-tactical-green",
    },
    connecting: {
      label: "LINKING",
      dotClass: "bg-tactical-yellow animate-tactical-blink",
      textClass: "text-tactical-yellow",
    },
    disconnected: {
      label: "OFFLINE",
      dotClass: "bg-tactical-red",
      textClass: "text-tactical-red",
    },
    mock: {
      label: "DEMO",
      dotClass: "bg-tactical-cyan animate-tactical-blink",
      textClass: "text-tactical-cyan",
    },
  },
} as const;

export type Design = typeof design;
