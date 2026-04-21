import { format } from "date-fns";
import { Camera, Mic, ExternalLink, Flag } from "lucide-react";
import { useStore } from "@/lib/incidentStore";
import { design } from "@/config/design";
import { cn } from "@/lib/utils";

export function DeviceTable() {
  const devices = useStore((s) => s.devices);

  return (
    <div className="overflow-hidden rounded-md border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <h3 className="font-mono text-xs uppercase tracking-widest text-foreground">
          Device Inventory
        </h3>
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {devices.length} units
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-background/40">
            <tr className="text-left font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              <Th>Device Name</Th>
              <Th>Type</Th>
              <Th>Location</Th>
              <Th>Status</Th>
              <Th>Last Event</Th>
              <Th>Last Seen</Th>
              <Th className="text-right">Actions</Th>
            </tr>
          </thead>
          <tbody>
            {devices.map((d) => {
              const s = design.deviceStatus[d.status];
              const Icon = d.type === "camera" ? Camera : Mic;
              return (
                <tr
                  key={d.id}
                  className="border-b border-border/60 last:border-b-0 hover:bg-surface-raised"
                >
                  <Td>
                    <div className="flex items-center gap-2">
                      <Icon className="h-3.5 w-3.5 text-muted-foreground" />
                      <div>
                        <div className="text-xs font-medium text-foreground">
                          {d.name}
                        </div>
                        <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                          {d.id}
                        </div>
                      </div>
                    </div>
                  </Td>
                  <Td className="capitalize text-muted-foreground">{d.type}</Td>
                  <Td>{d.location}</Td>
                  <Td>
                    <span className={cn("inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest", s.textClass)}>
                      <span className={cn("h-1.5 w-1.5 rounded-full", s.dotClass, d.status === "triggered" && "animate-tactical-blink")} />
                      {s.label}
                    </span>
                  </Td>
                  <Td className="max-w-[18rem] truncate text-xs text-muted-foreground">
                    {d.lastEvent ?? "—"}
                  </Td>
                  <Td className="font-mono text-[11px] text-muted-foreground">
                    {format(new Date(d.lastSeen), "HH:mm:ss")}
                  </Td>
                  <Td className="text-right">
                    <div className="inline-flex items-center gap-1">
                      <a
                        href="#"
                        onClick={(e) => e.preventDefault()}
                        className="inline-flex items-center gap-1 rounded-sm border border-border px-2 py-1 text-[10px] font-mono uppercase tracking-widest text-foreground hover:border-tactical-amber/50"
                      >
                        <ExternalLink className="h-3 w-3" /> View
                      </a>
                      <button className="inline-flex items-center gap-1 rounded-sm border border-border px-2 py-1 text-[10px] font-mono uppercase tracking-widest text-tactical-red hover:border-tactical-red/50">
                        <Flag className="h-3 w-3" /> Flag
                      </button>
                    </div>
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return <th className={cn("px-4 py-2 font-medium", className)}>{children}</th>;
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={cn("px-4 py-2.5 align-middle", className)}>{children}</td>;
}
