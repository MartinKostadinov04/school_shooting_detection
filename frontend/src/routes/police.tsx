import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useEffect } from "react";
import { Eye, LogOut, Building2, ChevronRight, Radio, Users, Cpu } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useStore } from "@/lib/incidentStore";
import { ConnectionIndicator } from "@/components/ui/ConnectionIndicator";
import { design } from "@/config/design";
import { POLICE_SCHOOLS } from "@/lib/schools";

export const Route = createFileRoute("/police")({
  head: () => ({
    meta: [
      { title: "Police Dispatch — TacticalEye" },
      {
        name: "description",
        content: "Schools under your department's jurisdiction.",
      },
    ],
  }),
  component: PoliceSchoolsPage,
});

function PoliceSchoolsPage() {
  const { user, ready, logout } = useAuth();
  const navigate = useNavigate();
  const incidents = useStore((s) => s.incidents);
  const connection = useStore((s) => s.connection);

  useEffect(() => {
    if (!ready) return;
    if (!user) navigate({ to: "/login" });
    else if (user.role !== "police") navigate({ to: "/school" });
  }, [user, ready, navigate]);

  if (!ready || !user) return null;

  // TRANSFER: in real backend, active count is per-school. Demo is single-tenant.
  const activeCount = incidents.filter((i) => i.status !== "RESOLVED").length;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background text-foreground">
      <header
        className="flex items-center justify-between border-b border-border bg-sidebar px-4"
        style={{ height: design.layout.headerHeight }}
      >
        <div className="flex items-center gap-3">
          <span className="flex h-7 w-7 items-center justify-center rounded-sm bg-tactical-amber text-background">
            <Eye className="h-4 w-4" />
          </span>
          <div>
            <div className="font-mono text-xs font-bold uppercase tracking-widest text-foreground">
              {design.app.name} · Dispatch
            </div>
            <div className="text-[10px] text-muted-foreground">
              Operator: {user.displayName} ({user.email})
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

      <main className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-6xl">
          <div className="mb-6 flex items-end justify-between">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-tactical-amber">
                Department Overview
              </div>
              <h1 className="mt-1 text-2xl font-semibold text-foreground">
                Schools Under Jurisdiction
              </h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Select a school to open its live dispatch console.
              </p>
            </div>
            <div className="flex gap-3">
              <Stat label="Schools" value={POLICE_SCHOOLS.length} />
              <Stat
                label="Active Incidents"
                value={activeCount}
                tone={activeCount > 0 ? "alert" : "default"}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {POLICE_SCHOOLS.map((school) => {
              // TRANSFER: per-school active count when store is multi-tenant.
              const schoolActive = activeCount;
              return (
                <Link
                  key={school.id}
                  to="/police/$schoolId"
                  params={{ schoolId: school.id }}
                  className="group relative flex flex-col rounded-md border border-border bg-surface p-4 transition-all hover:border-tactical-amber/60 hover:shadow-[0_0_22px_-8px_var(--tactical-amber)]"
                >
                  {schoolActive > 0 && (
                    <span className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-sm border border-tactical-red/50 bg-tactical-red/15 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-tactical-red">
                      <span className="h-1.5 w-1.5 animate-tactical-blink rounded-full bg-tactical-red" />
                      {schoolActive} ACTIVE
                    </span>
                  )}

                  <div className="flex items-start gap-3">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-sm border border-border bg-background/40 text-tactical-amber">
                      <Building2 className="h-5 w-5" />
                    </span>
                    <div className="min-w-0">
                      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                        {school.district}
                      </div>
                      <h2 className="truncate text-base font-semibold text-foreground">
                        {school.name}
                      </h2>
                      <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                        {school.address}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-2 border-t border-border pt-3">
                    <Meta icon={<Cpu className="h-3 w-3" />} label="Devices" value={school.deviceCount} />
                    <Meta
                      icon={<Users className="h-3 w-3" />}
                      label="Students"
                      value={school.studentCount.toLocaleString()}
                    />
                  </div>

                  <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
                    <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest text-tactical-cyan">
                      <Radio className="h-3 w-3" /> Open console
                    </span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-tactical-amber" />
                  </div>
                </Link>
              );
            })}

            {/* Placeholder slot — TRANSFER: remove when real list is wired. */}
            <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-border/60 bg-surface/40 p-6 text-center">
              <Building2 className="mb-2 h-6 w-6 text-muted-foreground" />
              <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                No additional schools
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground">
                Additional schools assigned to your department will appear here.
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function Stat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: number | string;
  tone?: "default" | "alert";
}) {
  const toneClass =
    tone === "alert"
      ? "border-tactical-red/50 text-tactical-red"
      : "border-border text-foreground";
  return (
    <div className={`rounded-sm border ${toneClass} bg-surface px-3 py-2 text-right`}>
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="font-mono text-lg font-semibold leading-none">{value}</div>
    </div>
  );
}

function Meta({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <div>
      <div className="flex items-center gap-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className="mt-0.5 font-mono text-sm text-foreground">{value}</div>
    </div>
  );
}
