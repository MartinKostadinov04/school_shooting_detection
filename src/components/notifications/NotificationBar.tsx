import { Bell } from "lucide-react";
import { useStore } from "@/lib/incidentStore";
import { ConnectionIndicator } from "@/components/ui/ConnectionIndicator";
import { NotificationItem } from "./NotificationItem";
import { design } from "@/config/design";

export function NotificationBar() {
  const notifications = useStore((s) => s.notifications);
  const connection = useStore((s) => s.connection);

  return (
    <aside
      className="flex h-full flex-col border-l border-border bg-sidebar"
      style={{ width: design.layout.notificationBarWidth }}
    >
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Bell className="h-4 w-4 text-tactical-amber" />
          <h2 className="font-mono text-xs uppercase tracking-widest text-foreground">
            Live Feed
          </h2>
        </div>
        <ConnectionIndicator state={connection} />
      </header>

      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        {notifications.length === 0 && (
          <div className="rounded-md border border-dashed border-border p-6 text-center font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
            Awaiting signals…
          </div>
        )}
        {notifications.map((n) => (
          <NotificationItem key={n.id} notification={n} />
        ))}
      </div>
    </aside>
  );
}
