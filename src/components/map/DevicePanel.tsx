import { format } from "date-fns";
import { Camera, Mic, X, Flag, ExternalLink } from "lucide-react";
import type { Device } from "@/types";
import { design } from "@/config/design";
import { cn } from "@/lib/utils";

export function DevicePanel({
  device,
  onClose,
}: {
  device: Device;
  onClose: () => void;
}) {
  const status = design.deviceStatus[device.status];
  const Icon = device.type === "camera" ? Camera : Mic;

  return (
    <div className="absolute right-3 top-3 z-20 w-72 rounded-md border border-border bg-popover/95 p-4 shadow-xl backdrop-blur">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-9 w-9 items-center justify-center rounded-sm bg-muted">
            <Icon className={cn("h-4 w-4", status.textClass)} />
          </span>
          <div>
            <div className="text-sm font-semibold">{device.name}</div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              {device.id}
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="rounded-sm p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
        <Field label="Location" value={device.location} />
        <Field label="Type" value={device.type} />
        <Field
          label="Status"
          value={<span className={status.textClass}>{status.label}</span>}
        />
        <Field
          label="Last seen"
          value={format(new Date(device.lastSeen), "HH:mm:ss")}
        />
      </div>

      <div className="mt-3 flex aspect-video items-center justify-center rounded-sm border border-dashed border-border bg-background/60 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {device.type === "camera" ? "LIVE FEED · MOCK" : "AUDIO STREAM · MOCK"}
      </div>

      {device.lastEvent && (
        <div className="mt-3 rounded-sm border border-border bg-background/60 p-2 text-[11px] text-muted-foreground">
          <div className="font-mono text-[9px] uppercase tracking-widest text-tactical-amber">
            Last event
          </div>
          <div className="mt-1">{device.lastEvent}</div>
        </div>
      )}

      <div className="mt-3 flex gap-2">
        <button className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-sm border border-border bg-surface px-2 py-1.5 text-[11px] font-medium text-foreground hover:border-tactical-amber/50">
          <ExternalLink className="h-3 w-3" /> View Feed
        </button>
        <button className="inline-flex items-center justify-center gap-1.5 rounded-sm border border-border bg-surface px-2 py-1.5 text-[11px] font-medium text-tactical-red hover:border-tactical-red/50">
          <Flag className="h-3 w-3" /> Flag
        </button>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 text-foreground">{value}</div>
    </div>
  );
}
