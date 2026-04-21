import { useState } from "react";
import { useStore } from "@/lib/incidentStore";
import { design } from "@/config/design";
import type { IncidentSeverity, IncidentType } from "@/types";

export function IncidentReportForm({
  onSubmitted,
  reportedBy,
}: {
  onSubmitted: (incidentId: string) => void;
  reportedBy: string;
}) {
  const devices = useStore((s) => s.devices);
  const reportManualIncident = useStore((s) => s.reportManualIncident);
  const sendMessage = useStore((s) => s.sendMessage);

  const locations = Array.from(new Set(devices.map((d) => d.location)));

  const [location, setLocation] = useState(locations[0] ?? "");
  const [type, setType] = useState<IncidentType>("Suspicious Activity");
  const [severity, setSeverity] = useState<IncidentSeverity>("High");
  const [description, setDescription] = useState("");

  const handle = (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim()) return;
    const inc = reportManualIncident({
      location,
      type,
      severity,
      description,
      reportedBy,
    });
    sendMessage({
      sender: "school",
      incidentId: inc.id,
      incidentReport: { location, type, severity, description },
    });
    setDescription("");
    onSubmitted(inc.id);
  };

  return (
    <form onSubmit={handle} className="space-y-2.5 p-3">
      <div className="grid grid-cols-2 gap-2">
        <Field label="Location">
          <select
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="w-full rounded-sm border border-border bg-input px-2 py-1.5 text-xs"
          >
            {locations.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Type">
          <select
            value={type}
            onChange={(e) => setType(e.target.value as IncidentType)}
            className="w-full rounded-sm border border-border bg-input px-2 py-1.5 text-xs"
          >
            {design.incidentTypes.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <Field label="Severity">
        <div className="flex gap-1">
          {design.severities.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSeverity(s)}
              className={`flex-1 rounded-sm border px-2 py-1 font-mono text-[10px] uppercase tracking-widest ${
                severity === s
                  ? `${design.severity[s].borderClass} ${design.severity[s].bgClass} ${design.severity[s].textClass}`
                  : "border-border text-muted-foreground hover:border-foreground/30"
              }`}
            >
              {design.severity[s].label}
            </button>
          ))}
        </div>
      </Field>
      <Field label="Description">
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
          placeholder="Brief situation report…"
          className="w-full resize-none rounded-sm border border-border bg-input px-2 py-1.5 text-xs"
        />
      </Field>
      <button
        type="submit"
        className="w-full rounded-sm bg-tactical-red px-3 py-2 font-mono text-[11px] font-bold uppercase tracking-widest text-white hover:opacity-90"
      >
        Dispatch Report
      </button>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1 font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      {children}
    </label>
  );
}
