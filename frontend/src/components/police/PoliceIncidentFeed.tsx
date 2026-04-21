import { useEffect, useState } from "react";
import { useStore } from "@/lib/incidentStore";
import { IncidentCard } from "@/components/incidents/IncidentCard";
import type { Incident } from "@/types";

export function PoliceIncidentFeed({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (i: Incident) => void;
}) {
  const incidents = useStore((s) => s.incidents);
  const [filter, setFilter] = useState<"ALL" | "NEW" | "ACK" | "RESOLVED">("ALL");

  // Auto-select newest NEW if nothing chosen
  useEffect(() => {
    if (!selectedId && incidents.length) {
      const newest = incidents.find((i) => i.status === "NEW") ?? incidents[0];
      onSelect(newest);
    }
  }, [incidents, selectedId, onSelect]);

  const visible = incidents.filter((i) => {
    if (filter === "ALL") return true;
    if (filter === "NEW") return i.status === "NEW";
    if (filter === "ACK") return i.status === "ACKNOWLEDGED";
    return i.status === "RESOLVED";
  });

  const counts = {
    ALL: incidents.length,
    NEW: incidents.filter((i) => i.status === "NEW").length,
    ACK: incidents.filter((i) => i.status === "ACKNOWLEDGED").length,
    RESOLVED: incidents.filter((i) => i.status === "RESOLVED").length,
  };

  return (
    <div className="flex h-full flex-col rounded-md border border-border bg-surface">
      <header className="border-b border-border p-3">
        <div className="flex items-center justify-between">
          <h2 className="font-mono text-xs uppercase tracking-widest">Incident Feed</h2>
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            {visible.length} of {incidents.length}
          </span>
        </div>
        <div className="mt-2 flex gap-1">
          {(["ALL", "NEW", "ACK", "RESOLVED"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`flex-1 rounded-sm border px-1.5 py-1 font-mono text-[10px] uppercase tracking-widest ${
                filter === f
                  ? "border-tactical-amber bg-tactical-amber/15 text-tactical-amber"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {f} · {counts[f]}
            </button>
          ))}
        </div>
      </header>
      <div className="flex-1 space-y-2 overflow-y-auto p-2">
        {visible.map((i) => (
          <IncidentCard
            key={i.id}
            incident={i}
            selected={selectedId === i.id}
            onSelect={onSelect}
          />
        ))}
        {visible.length === 0 && (
          <div className="rounded-md border border-dashed border-border p-6 text-center font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
            No incidents in this filter
          </div>
        )}
      </div>
    </div>
  );
}
