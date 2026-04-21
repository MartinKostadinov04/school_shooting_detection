import { useState } from "react";
import { ZoomIn, ZoomOut, Locate } from "lucide-react";
import { useStore } from "@/lib/incidentStore";
import { DevicePin } from "./DevicePin";
import { DevicePanel } from "./DevicePanel";
import type { Device } from "@/types";
import { design } from "@/config/design";
import floorplan from "@/assets/school-floorplan.png";

export function SchoolMap() {
  const devices = useStore((s) => s.devices);
  const [selected, setSelected] = useState<Device | null>(null);
  const [zoom, setZoom] = useState(1);

  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-border bg-surface bg-tactical-grid"
      style={{ height: design.layout.mapHeight }}
    >
      {/* zoom controls */}
      <div className="absolute left-3 top-3 z-20 flex flex-col gap-1 rounded-md border border-border bg-popover/95 p-1 shadow backdrop-blur">
        <button
          onClick={() => setZoom((z) => Math.min(z + 0.2, 2))}
          className="rounded-sm p-1.5 text-foreground hover:bg-muted"
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(z - 0.2, 0.6))}
          className="rounded-sm p-1.5 text-foreground hover:bg-muted"
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => setZoom(1)}
          className="rounded-sm p-1.5 text-foreground hover:bg-muted"
        >
          <Locate className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="absolute right-3 top-3 z-10 rounded-sm border border-border bg-popover/90 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground backdrop-blur">
        SCHOOL · LIVE TOPOLOGY
      </div>

      <div
        className="relative h-full w-full origin-center transition-transform duration-300"
        style={{ transform: `scale(${zoom})` }}
      >
        {/*
          Floor plan background.
          TRANSFER: swap `school-floorplan.png` with the real school's plan.
          Keep the image's intrinsic aspect; pin coordinates are stored as
          percentages, so any plan works as long as devices are re-measured.
          The `invert` filter makes a black-on-white plan read on dark UI —
          drop it (or replace with a styled SVG plan) for production.
        */}
        <img
          src={floorplan}
          alt="School floor plan"
          className="absolute inset-0 h-full w-full object-contain opacity-70 [filter:invert(1)_hue-rotate(180deg)_brightness(1.1)_contrast(0.9)]"
          draggable={false}
        />

        {devices.map((d) => (
          <DevicePin
            key={d.id}
            device={d}
            selected={selected?.id === d.id}
            onClick={(dev) => setSelected((s) => (s?.id === dev.id ? null : dev))}
          />
        ))}
      </div>

      {selected && (
        <DevicePanel device={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
