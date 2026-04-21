import { createFileRoute, Link } from "@tanstack/react-router";
import { Eye, ShieldAlert } from "lucide-react";

export const Route = createFileRoute("/")({
  component: RolePicker,
});

function RolePicker() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background bg-tactical-grid">
      <div className="w-full max-w-sm space-y-4 p-6">
        <div className="mb-8 flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-tactical-amber text-background">
            <Eye className="h-5 w-5" />
          </span>
          <div className="font-mono text-sm font-bold uppercase tracking-widest text-foreground">
            TacticalEye
          </div>
        </div>

        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Select console
        </p>

        <Link
          to="/school"
          className="flex w-full items-center gap-3 rounded-sm border border-border bg-surface px-4 py-3 text-sm font-medium text-foreground hover:border-tactical-amber/60 hover:shadow-[0_0_18px_-6px_var(--tactical-amber)]"
        >
          <Eye className="h-4 w-4 text-tactical-amber" />
          School Operations
        </Link>

        <Link
          to="/police"
          className="flex w-full items-center gap-3 rounded-sm border border-border bg-surface px-4 py-3 text-sm font-medium text-foreground hover:border-tactical-cyan/60 hover:shadow-[0_0_18px_-6px_var(--tactical-cyan)]"
        >
          <ShieldAlert className="h-4 w-4 text-tactical-cyan" />
          Police Dispatch
        </Link>
      </div>
    </div>
  );
}
