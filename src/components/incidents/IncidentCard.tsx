import { format, formatDistanceToNow } from "date-fns";
import type { Incident } from "@/types";
import { SeverityBadge, SourceBadge, StatusPill } from "@/components/ui/StatusBadges";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/incidentStore";
import { Check, Send, ShieldCheck } from "lucide-react";

export function IncidentCard({
  incident,
  selected,
  onSelect,
}: {
  incident: Incident;
  selected: boolean;
  onSelect: (i: Incident) => void;
}) {
  const setIncidentStatus = useStore((s) => s.setIncidentStatus);
  const dispatchUnit = useStore((s) => s.dispatchUnit);

  return (
    <button
      onClick={() => onSelect(incident)}
      className={cn(
        "w-full rounded-md border bg-surface p-3 text-left transition-colors hover:border-tactical-amber/50",
        selected ? "border-tactical-amber" : "border-border",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="font-mono text-[11px] font-bold tracking-widest text-foreground">
            {incident.id}
          </div>
          <div className="mt-0.5 text-[11px] text-muted-foreground">
            {format(new Date(incident.createdAt), "MMM d · HH:mm:ss")} ·{" "}
            {formatDistanceToNow(new Date(incident.createdAt), { addSuffix: true })}
          </div>
        </div>
        <StatusPill status={incident.status} />
      </div>

      <div className="mt-2 flex items-center gap-2">
        <SourceBadge source={incident.source} />
        <SeverityBadge severity={incident.severity} />
        {incident.videoConfirmed && (
          <span className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-widest text-tactical-violet">
            <ShieldCheck className="h-3 w-3" /> VIDEO CONF
          </span>
        )}
      </div>

      <div className="mt-2 text-sm font-medium text-foreground">
        {incident.type} · {incident.location}
      </div>
      {typeof incident.probability === "number" && (
        <div className="mt-1 font-mono text-[11px] text-tactical-amber">
          P = {(incident.probability * 100).toFixed(0)}%
        </div>
      )}

      <div
        className="mt-3 flex items-center gap-1.5"
        onClick={(e) => e.stopPropagation()}
      >
        {incident.status === "NEW" && (
          <ActionBtn
            onClick={() => setIncidentStatus(incident.id, "ACKNOWLEDGED")}
            tone="amber"
          >
            <Check className="h-3 w-3" /> ACK
          </ActionBtn>
        )}
        {incident.status !== "RESOLVED" && (
          <ActionBtn
            onClick={() => setIncidentStatus(incident.id, "RESOLVED")}
            tone="green"
          >
            <Check className="h-3 w-3" /> Resolve
          </ActionBtn>
        )}
        <ActionBtn onClick={() => dispatchUnit(incident.id)} tone="red">
          <Send className="h-3 w-3" /> Dispatch
        </ActionBtn>
      </div>
    </button>
  );
}

function ActionBtn({
  children,
  onClick,
  tone,
}: {
  children: React.ReactNode;
  onClick: () => void;
  tone: "amber" | "green" | "red";
}) {
  const c =
    tone === "amber"
      ? "border-tactical-amber/40 text-tactical-amber hover:bg-tactical-amber/10"
      : tone === "green"
        ? "border-tactical-green/40 text-tactical-green hover:bg-tactical-green/10"
        : "border-tactical-red/40 text-tactical-red hover:bg-tactical-red/10";
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-2 py-1 font-mono text-[10px] uppercase tracking-widest",
        c,
      )}
    >
      {children}
    </button>
  );
}
