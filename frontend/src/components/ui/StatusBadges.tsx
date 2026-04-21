import { design } from "@/config/design";
import { cn } from "@/lib/utils";
import type {
  IncidentSeverity,
  IncidentSource,
  IncidentStatus,
} from "@/types";

export function StatusPill({
  status,
  className,
}: {
  status: IncidentStatus;
  className?: string;
}) {
  const c = design.status[status];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-widest",
        c.textClass,
        c.bgClass,
        c.borderClass,
        className,
      )}
    >
      {c.pulse && (
        <span className="h-1.5 w-1.5 rounded-full bg-current animate-tactical-blink" />
      )}
      {c.label}
    </span>
  );
}

export function SourceBadge({
  source,
  className,
}: {
  source: IncidentSource;
  className?: string;
}) {
  const c = design.source[source];
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-widest",
        c.textClass,
        c.bgClass,
        c.borderClass,
        className,
      )}
    >
      {c.label}
    </span>
  );
}

export function SeverityBadge({
  severity,
  className,
}: {
  severity: IncidentSeverity;
  className?: string;
}) {
  const c = design.severity[severity];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-widest",
        c.textClass,
        c.bgClass,
        c.borderClass,
        className,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", c.dotClass)} />
      {c.label}
    </span>
  );
}
