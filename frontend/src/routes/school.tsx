import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { useAbly } from "@/hooks/useAbly";
import { SchoolSidebar } from "@/components/SchoolSidebar";
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
  const loadFromApi  = useStore((s) => s.loadFromApi);
  const incidents    = useStore((s) => s.incidents);
  const connection   = useStore((s) => s.connection);

  useEffect(() => { loadFromApi(); }, []);

  const activeCount = incidents.filter((i) => i.status !== "RESOLVED").length;
  const newCount = incidents.filter((i) => i.status === "NEW").length;

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      <SchoolSidebar />

      <main className="flex flex-1 flex-col overflow-hidden">
        <div
          className="flex items-center justify-between border-b border-border bg-surface px-4"
          style={{ height: design.layout.headerHeight }}
        >
          <div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              School Operations
            </div>
            <div className="text-sm font-semibold">Live Situation Console</div>
          </div>
          <div className="flex items-center gap-2">
            <Stat label="Active" value={activeCount} tone="amber" />
            <Stat label="NEW" value={newCount} tone="red" />
            <ConnectionIndicator state={connection} />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <SchoolMap />
          <div className="mt-4">
            <DeviceTable />
          </div>
        </div>
      </main>

      <NotificationBar />
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
