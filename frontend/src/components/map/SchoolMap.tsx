import { useRef, useState } from "react";
import { ZoomIn, ZoomOut, Locate, Pencil, Check, RotateCcw } from "lucide-react";
import { useStore } from "@/lib/incidentStore";
import { DevicePin } from "./DevicePin";
import { DevicePanel } from "./DevicePanel";
import type { Device } from "@/types";
import { design } from "@/config/design";
import floorplan from "@/assets/school-floorplan.png";

export function SchoolMap({ readOnly = false }: { readOnly?: boolean }) {
  const devices             = useStore((s) => s.devices);
  const setDevicePosition   = useStore((s) => s.setDevicePosition);
  const resetDevicePositions = useStore((s) => s.resetDevicePositions);

  const [selected, setSelected] = useState<Device | null>(null);
  const [zoom, setZoom]         = useState(1);
  const [editMode, setEditMode] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);

  const exitEdit = () => {
    setEditMode(false);
    setSelected(null);
  };

  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-border bg-surface bg-tactical-grid"
      style={{ height: design.layout.mapHeight }}
    >
      {/* controls */}
      <div className="absolute left-3 top-3 z-20 flex flex-col gap-1 rounded-md border border-border bg-popover/95 p-1 shadow backdrop-blur">
        {!editMode ? (
          <>
            <CtrlBtn onClick={() => setZoom((z) => Math.min(z + 0.2, 2))}><ZoomIn className="h-3.5 w-3.5" /></CtrlBtn>
            <CtrlBtn onClick={() => setZoom((z) => Math.max(z - 0.2, 0.6))}><ZoomOut className="h-3.5 w-3.5" /></CtrlBtn>
            <CtrlBtn onClick={() => setZoom(1)}><Locate className="h-3.5 w-3.5" /></CtrlBtn>
            {!readOnly && (
              <>
                <div className="my-0.5 h-px bg-border" />
                <CtrlBtn onClick={() => { setEditMode(true); setSelected(null); }}>
                  <Pencil className="h-3.5 w-3.5" />
                </CtrlBtn>
              </>
            )}
          </>
        ) : (
          <>
            <CtrlBtn onClick={exitEdit} title="Done">
              <Check className="h-3.5 w-3.5 text-tactical-green" />
            </CtrlBtn>
            <CtrlBtn onClick={() => { resetDevicePositions(); }} title="Reset all positions">
              <RotateCcw className="h-3.5 w-3.5 text-tactical-amber" />
            </CtrlBtn>
          </>
        )}
      </div>

      <div className="absolute right-3 top-3 z-10 rounded-sm border border-border bg-popover/90 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground backdrop-blur">
        {editMode ? "✏ drag pins to reposition" : "SCHOOL · LIVE TOPOLOGY"}
      </div>

      <div
        ref={mapRef}
        className="relative h-full w-full origin-center transition-transform duration-300"
        style={{ transform: `scale(${zoom})` }}
      >
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
            editMode={editMode}
            mapRef={mapRef}
            onMove={(dev, x, y) => setDevicePosition(dev.id, x, y)}
          />
        ))}
      </div>

      {selected && (
        <DevicePanel device={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function CtrlBtn({
  onClick,
  title,
  children,
}: {
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="rounded-sm p-1.5 text-foreground hover:bg-muted"
    >
      {children}
    </button>
  );
}
