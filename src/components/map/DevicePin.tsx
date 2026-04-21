import { Camera, Mic } from "lucide-react";
import { cn } from "@/lib/utils";
import { design } from "@/config/design";
import type { Device } from "@/types";

export function DevicePin({
  device,
  selected,
  onClick,
}: {
  device: Device;
  selected: boolean;
  onClick: (d: Device) => void;
}) {
  const status = design.deviceStatus[device.status];
  const Icon = device.type === "camera" ? Camera : Mic;

  return (
    <button
      onClick={() => onClick(device)}
      className="absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${device.x}%`, top: `${device.y}%` }}
      aria-label={device.name}
    >
      <span
        className={cn(
          "relative flex h-9 w-9 items-center justify-center rounded-full border-2 border-background bg-surface-raised transition-transform hover:scale-110",
          status.glowClass,
          selected && "ring-2 ring-tactical-amber ring-offset-2 ring-offset-background",
          device.status === "triggered" && "animate-tactical-pulse",
        )}
        style={{ color: `var(--tactical-${device.status === "online" ? "green" : device.status === "warning" ? "yellow" : device.status === "triggered" ? "red" : ""})` }}
      >
        <Icon className={cn("h-4 w-4", status.textClass)} />
        <span
          className={cn(
            "absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full border border-background",
            status.dotClass,
          )}
        />
      </span>
    </button>
  );
}
