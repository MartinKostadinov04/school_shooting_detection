import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect } from "react";
import { Eye, ArrowLeft } from "lucide-react";
import { useAbly } from "@/hooks/useAbly";
import { NotificationBar } from "@/components/notifications/NotificationBar";
import { SchoolMap } from "@/components/map/SchoolMap";
import { DeviceTable } from "@/components/devices/DeviceTable";
import { CommunicationWindow } from "@/components/comms/CommunicationWindow";
import { useStore } from "@/lib/incidentStore";
import { design } from "@/config/design";
import { ConnectionIndicator } from "@/components/ui/ConnectionIndicator";

export const Route = createFileRoute("/school")({
  head: () => ({
    meta: [
      { title: "School Console — TacticalEye" },
      {
        name: "description",
        content: "School-side dashboard: live device map, alerts, and police comms.",
      },
    ],
  }),
  component: SchoolPage,
});

function SchoolPage() {
  useAbly();
  const loadFromApi = useStore((s) => s.loadFromApi);
  const incidents   = useStore((s) => s.incidents);
  const connection  = useStore((s) => s.connection);

  useEffect(() => { loadFromApi(); }, []);

  const activeCount = incidents.filter((i) => i.status !== "RESOLVED").length;
  const newCount    = incidents.filter((i) => i.status === "NEW").length;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background text-foreground">
      <header
        className="flex items-center justify-between border-b border-border bg-sidebar px-4"
        style={{ height: design.layout.headerHeight }}
      >
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="flex h-7 w-7 items-center justify-center rounded-sm border border-border text-muted-foreground transition-colors hover:border-tactical-amber/60 hover:text-tactical-amber"
            aria-label="Back to role select"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
          </Link>
          <span className="flex h-7 w-7 items-center justify-center rounded-sm bg-tactical-amber text-background">
            <Eye className="h-4 w-4" />
          </span>
          <div>
            <div className="font-mono text-xs font-bold uppercase tracking-widest text-foreground">
              {design.app.name} · School
            </div>
            <div className="text-[10px] text-muted-foreground">Live Situation Console</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Stat label="Active" value={activeCount} tone="amber" />
          <Stat label="NEW" value={newCount} tone="red" />
          <ConnectionIndicator state={connection} />
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-1 flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            <SchoolMap />
            <div className="mt-4">
              <DeviceTable />
            </div>
          </div>
        </main>
        <NotificationBar />
      </div>

      <CommunicationWindow viewerRole="school" />
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "amber" | "red";
}) {
  const c = tone === "amber" ? "text-tactical-amber" : "text-tactical-red";
  return (
    <div className="flex items-center gap-2 rounded-sm border border-border bg-background px-2.5 py-1">
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span className={`font-mono text-sm font-bold ${c}`}>{value}</span>
    </div>
  );
}
