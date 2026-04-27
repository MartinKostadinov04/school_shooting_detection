import { useRef } from "react";
import { Camera, Mic } from "lucide-react";
import { cn } from "@/lib/utils";
import { design } from "@/config/design";
import type { Device } from "@/types";

export function DevicePin({
  device,
  selected,
  onClick,
  editMode = false,
  mapRef,
  onMove,
}: {
  device: Device;
  selected: boolean;
  onClick: (d: Device) => void;
  editMode?: boolean;
  mapRef?: React.RefObject<HTMLDivElement | null>;
  onMove?: (device: Device, x: number, y: number) => void;
}) {
  const status = design.deviceStatus[device.status];
  const Icon = device.type === "camera" ? Camera : Mic;
  const didDrag = useRef(false);

  const handlePointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (!editMode) return;
    e.preventDefault();
    didDrag.current = false;
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (!editMode || !(e.buttons & 1) || !mapRef?.current) return;
    didDrag.current = true;
    const rect = mapRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
    onMove?.(device, x, y);
  };

  const handleClick = () => {
    if (editMode || didDrag.current) return;
    onClick(device);
  };

  return (
    <button
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onClick={handleClick}
      className={cn(
        "absolute -translate-x-1/2 -translate-y-1/2 touch-none",
        editMode ? "cursor-move" : "cursor-pointer",
      )}
      style={{ left: `${device.x}%`, top: `${device.y}%` }}
      aria-label={device.name}
    >
      <span
        className={cn(
          "relative flex h-9 w-9 items-center justify-center rounded-full border-2 border-background bg-surface-raised transition-transform hover:scale-110",
          status.glowClass,
          selected && "ring-2 ring-tactical-amber ring-offset-2 ring-offset-background",
          device.status === "triggered" && "animate-tactical-pulse",
          editMode && "ring-2 ring-tactical-amber/40",
        )}
        style={{
          color: `var(--tactical-${
            device.status === "online" ? "green"
            : device.status === "warning" ? "yellow"
            : device.status === "triggered" ? "red"
            : ""
          })`,
        }}
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
