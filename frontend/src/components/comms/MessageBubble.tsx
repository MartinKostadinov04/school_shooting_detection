import { format } from "date-fns";
import type { ChatMessage } from "@/types";
import { SeverityBadge } from "@/components/ui/StatusBadges";
import { cn } from "@/lib/utils";

export function MessageBubble({ msg, viewerRole }: { msg: ChatMessage; viewerRole: "school" | "police" }) {
  if (msg.sender === "system") {
    return (
      <div className="my-2 text-center font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        — {msg.text} —
      </div>
    );
  }

  const isMine = msg.sender === viewerRole;

  if (msg.incidentReport) {
    const r = msg.incidentReport;
    return (
      <div className={cn("flex", isMine ? "justify-end" : "justify-start")}>
        <div className="max-w-[85%] rounded-md border border-tactical-red/40 bg-tactical-red/10 p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="font-mono text-[10px] uppercase tracking-widest text-tactical-red">
              ⚠ Incident Report · {msg.incidentId}
            </div>
            <SeverityBadge severity={r.severity} />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-1.5 text-[11px]">
            <div className="text-muted-foreground">Location</div>
            <div className="text-foreground">{r.location}</div>
            <div className="text-muted-foreground">Type</div>
            <div className="text-foreground">{r.type}</div>
          </div>
          <div className="mt-2 rounded-sm border border-border bg-background/40 p-2 text-xs text-foreground">
            {r.description}
          </div>
          <div className="mt-1.5 text-right font-mono text-[10px] text-muted-foreground">
            {format(new Date(msg.timestamp), "HH:mm:ss")}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex", isMine ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-md px-3 py-2 text-xs",
          isMine
            ? "bg-tactical-amber/15 text-foreground"
            : "border border-border bg-surface text-foreground",
        )}
      >
        <div className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
          {msg.sender} · {format(new Date(msg.timestamp), "HH:mm:ss")}
        </div>
        <div className="mt-0.5">{msg.text}</div>
      </div>
    </div>
  );
}
