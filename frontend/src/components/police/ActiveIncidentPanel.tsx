/**
 * Police-side active incident panel.
 *
 * The actual rendering lives in the shared IncidentDetail component so the
 * school-side dashboard can present the same view (audio waveform, annotated
 * video, timeline, comms, gun count) with role-appropriate affordances.
 * Keeping this thin wrapper preserves the existing import path used by the
 * police route and any other callers.
 */

import type { Incident } from "@/types";
import { IncidentDetail } from "@/components/incidents/IncidentDetail";

export function ActiveIncidentPanel({ incident }: { incident: Incident | null }) {
  return <IncidentDetail incident={incident} viewerRole="police" />;
}
