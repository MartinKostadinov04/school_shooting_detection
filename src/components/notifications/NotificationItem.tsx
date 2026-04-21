import { format } from "date-fns";
import { Mic, Camera, Video } from "lucide-react";
import type { Notification } from "@/types";
import { cn } from "@/lib/utils";
import { SourceBadge } from "@/components/ui/StatusBadges";

export function NotificationItem({
  notification,
  onSelect,
}: {
  notification: Notification;
  onSelect?: (n: Notification) => void;
}) {
  const Icon = notification.deviceType === "camera" ? Camera : Mic;
  return (
    <button
      onClick={() => onSelect?.(notification)}
      className={cn(
        "group w-full rounded-md border border-border bg-surface p-3 text-left transition-colors hover:border-tactical-amber/60 hover:bg-surface-raised",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-sm bg-muted text-tactical-amber">
            <Icon className="h-3.5 w-3.5" />
          </span>
          <div>
            <div className="text-xs font-semibold text-foreground">
              {notification.location}
            </div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              {format(new Date(notification.timestamp), "HH:mm:ss")}
            </div>
          </div>
        </div>
        <SourceBadge source={notification.source} />
      </div>
      <div className="mt-2 text-xs text-muted-foreground">{notification.message}</div>
      {typeof notification.probability === "number" && (
        <div className="mt-2 flex items-center gap-2">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-tactical-amber"
              style={{ width: `${Math.round(notification.probability * 100)}%` }}
            />
          </div>
          <span className="font-mono text-[10px] text-tactical-amber">
            {(notification.probability * 100).toFixed(0)}%
          </span>
        </div>
      )}
      {notification.audioUrl && (
        <audio
          src={notification.audioUrl}
          controls
          className="mt-3 h-8 w-full"
          onClick={(e) => e.stopPropagation()}
        />
      )}
      {notification.videoUrl && (
        <div className="mt-3 flex items-center gap-2 rounded-sm border border-border bg-background/40 p-2 text-xs text-muted-foreground">
          <Video className="h-3.5 w-3.5 text-tactical-violet" />
          Video segment attached
        </div>
      )}
    </button>
  );
}
