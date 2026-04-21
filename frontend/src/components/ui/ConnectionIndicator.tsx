import { design } from "@/config/design";
import type { ConnectionState } from "@/types";
import { cn } from "@/lib/utils";

export function ConnectionIndicator({
  state,
  className,
}: {
  state: ConnectionState;
  className?: string;
}) {
  const c = design.connection[state];
  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full border border-border bg-surface px-2.5 py-1 font-mono text-[10px] uppercase tracking-widest",
        c.textClass,
        className,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", c.dotClass)} />
      {c.label}
    </div>
  );
}
