import { format } from "date-fns";
import { Volume2, Video } from "lucide-react";
import type { Incident } from "@/types";
import { SeverityBadge, SourceBadge, StatusPill } from "@/components/ui/StatusBadges";
import { CommunicationWindow } from "@/components/comms/CommunicationWindow";

export function ActiveIncidentPanel({ incident }: { incident: Incident | null }) {
  if (!incident) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed border-border bg-surface">
        <div className="text-center">
          <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Select an incident
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            Detection details, media, and timeline appear here.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto">
      {/* Header */}
      <div className="rounded-md border border-border bg-surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="font-mono text-xs font-bold tracking-widest text-tactical-amber">
              {incident.id}
            </div>
            <h2 className="mt-1 text-xl font-semibold text-foreground">
              {incident.type} · {incident.location}
            </h2>
            <div className="mt-1 font-mono text-[11px] text-muted-foreground">
              {format(new Date(incident.createdAt), "PPpp")}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <StatusPill status={incident.status} />
            <SeverityBadge severity={incident.severity} />
            <SourceBadge source={incident.source} />
          </div>
        </div>
        {incident.description && (
          <div className="mt-3 rounded-sm border border-border bg-background/40 p-3 text-sm text-foreground">
            {incident.description}
          </div>
        )}
      </div>

      {/* Media row */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {/* Audio */}
        <div className="rounded-md border border-border bg-surface p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Volume2 className="h-3.5 w-3.5 text-tactical-cyan" />
              <h3 className="font-mono text-[11px] uppercase tracking-widest">
                Audio Evidence
              </h3>
            </div>
            {typeof incident.probability === "number" && (
              <span className="font-mono text-[10px] uppercase tracking-widest text-tactical-amber">
                P {(incident.probability * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <Waveform />
          {incident.audioUrl ? (
            <audio src={incident.audioUrl} controls className="mt-2 w-full" />
          ) : (
            <div className="mt-2 rounded-sm border border-dashed border-border p-3 text-center font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Awaiting snippet…
            </div>
          )}
        </div>

        {/* Video */}
        <div className="rounded-md border border-border bg-surface p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Video className="h-3.5 w-3.5 text-tactical-violet" />
              <h3 className="font-mono text-[11px] uppercase tracking-widest">
                Video Evidence
              </h3>
            </div>
            {incident.videoConfirmed && (
              <span className="font-mono text-[10px] uppercase tracking-widest text-tactical-violet">
                VIDEO-AI CONFIRMED
              </span>
            )}
          </div>
          {incident.videoUrl ? (
            <video src={incident.videoUrl} controls className="aspect-video w-full rounded-sm bg-black" />
          ) : (
            <div className="flex aspect-video items-center justify-center rounded-sm border border-dashed border-border bg-background/40 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              No video segment
            </div>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div className="rounded-md border border-border bg-surface p-4">
        <h3 className="mb-3 font-mono text-[11px] uppercase tracking-widest">
          Event Timeline
        </h3>
        <ol className="space-y-2.5">
          {incident.timeline.map((t) => (
            <li key={t.id} className="flex gap-3">
              <div className="flex flex-col items-center">
                <span className="h-2 w-2 rounded-full bg-tactical-amber" />
                <span className="mt-1 w-px flex-1 bg-border" />
              </div>
              <div className="flex-1 pb-1">
                <div className="text-xs font-medium text-foreground">{t.label}</div>
                {t.detail && (
                  <div className="text-[11px] text-muted-foreground">{t.detail}</div>
                )}
                <div className="mt-0.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  {format(new Date(t.timestamp), "MMM d · HH:mm:ss")}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* Comms */}
      <div className="h-80">
        <CommunicationWindow viewerRole="police" filterIncidentId={incident.id} />
      </div>
    </div>
  );
}

function Waveform() {
  // Decorative waveform — does not represent real audio.
  const bars = Array.from({ length: 56 }, (_, i) => i);
  return (
    <div className="flex h-16 items-center gap-[2px] rounded-sm border border-border bg-background/40 px-2">
      {bars.map((i) => {
        const h = 18 + Math.abs(Math.sin(i * 0.6) + Math.cos(i * 0.27)) * 28;
        return (
          <span
            key={i}
            className="block w-[3px] rounded-full bg-tactical-amber/60"
            style={{ height: `${h}px` }}
          />
        );
      })}
    </div>
  );
}
