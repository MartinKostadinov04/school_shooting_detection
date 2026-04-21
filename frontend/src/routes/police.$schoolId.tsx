import { createFileRoute, useNavigate, Link, notFound } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowLeft, Eye, LogOut } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useAbly } from "@/hooks/useAbly";
import { useStore } from "@/lib/incidentStore";
import { PoliceIncidentFeed } from "@/components/police/PoliceIncidentFeed";
import { ActiveIncidentPanel } from "@/components/police/ActiveIncidentPanel";
import { SchoolMap } from "@/components/map/SchoolMap";
import { ConnectionIndicator } from "@/components/ui/ConnectionIndicator";
import { design } from "@/config/design";
import { POLICE_SCHOOLS } from "@/lib/schools";
import type { Incident } from "@/types";

export const Route = createFileRoute("/police/$schoolId")({
  loader: ({ params }) => {
    const school = POLICE_SCHOOLS.find((s) => s.id === params.schoolId);
    if (!school) throw notFound();
    return { school };
  },
  head: ({ loaderData }) => ({
    meta: [
      {
        title: loaderData?.school
          ? `${loaderData.school.name} — Dispatch`
          : "Dispatch — TacticalEye",
      },
      {
        name: "description",
        content: "Live incident feed, evidence, and direct comms with the school.",
      },
    ],
  }),
  errorComponent: ({ error, reset }) => (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <div className="max-w-md text-center">
        <div className="font-mono text-xs uppercase tracking-widest text-tactical-red">
          Console error
        </div>
        <p className="mt-2 text-sm text-muted-foreground">{error.message}</p>
        <button
          onClick={reset}
          className="mt-4 rounded-sm border border-border px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest hover:border-tactical-amber/60"
        >
          Retry
        </button>
      </div>
    </div>
  ),
  notFoundComponent: () => (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <div className="text-center">
        <div className="font-mono text-xs uppercase tracking-widest text-tactical-amber">
          School not found
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          This school is not assigned to your department.
        </p>
        <Link
          to="/police"
          className="mt-4 inline-flex items-center gap-1 rounded-sm border border-border px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest hover:border-tactical-amber/60"
        >
          <ArrowLeft className="h-3 w-3" /> Back to schools
        </Link>
      </div>
    </div>
  ),
  component: SchoolDispatchPage,
});

function SchoolDispatchPage() {
  const { school } = Route.useLoaderData();
  const { user, ready, logout } = useAuth();
  const navigate = useNavigate();
  // TRANSFER: pass schoolId into useAbly so each school subscribes to its
  // own channel (e.g. `gunshot-detection.{schoolId}`). Demo is single-tenant.
  useAbly();

  const incidents = useStore((s) => s.incidents);
  const connection = useStore((s) => s.connection);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    if (!user) navigate({ to: "/login" });
    else if (user.role !== "police") navigate({ to: "/school" });
  }, [user, ready, navigate]);

  if (!ready || !user) return null;

  const selected = incidents.find((i) => i.id === selectedId) ?? null;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background text-foreground">
      <header
        className="flex items-center justify-between border-b border-border bg-sidebar px-4"
        style={{ height: design.layout.headerHeight }}
      >
        <div className="flex items-center gap-3">
          <Link
            to="/police"
            className="flex h-7 w-7 items-center justify-center rounded-sm border border-border text-muted-foreground transition-colors hover:border-tactical-amber/60 hover:text-tactical-amber"
            aria-label="Back to schools"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
          </Link>
          <span className="flex h-7 w-7 items-center justify-center rounded-sm bg-tactical-amber text-background">
            <Eye className="h-4 w-4" />
          </span>
          <div>
            <div className="font-mono text-xs font-bold uppercase tracking-widest text-foreground">
              {school.name}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {school.district} · Operator: {user.displayName}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ConnectionIndicator state={connection} />
          <button
            onClick={logout}
            className="inline-flex items-center gap-1.5 rounded-sm border border-border px-2.5 py-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground hover:border-tactical-red/50 hover:text-tactical-red"
          >
            <LogOut className="h-3 w-3" /> Sign out
          </button>
        </div>
      </header>

      <div className="flex flex-1 gap-3 overflow-hidden p-3">
        <div className="shrink-0" style={{ width: design.layout.incidentFeedWidth }}>
          <PoliceIncidentFeed
            selectedId={selectedId}
            onSelect={(i: Incident) => setSelectedId(i.id)}
          />
        </div>

        <div className="flex flex-1 flex-col gap-3 overflow-y-auto">
          {/*
            Live floor plan gives dispatch the same situational awareness as
            the school operator. TRANSFER: scope devices by schoolId once the
            incident store is multi-tenant.
          */}
          <SchoolMap />
          <ActiveIncidentPanel incident={selected} />
        </div>
      </div>
    </div>
  );
}
