import { Eye } from "lucide-react";
import { design } from "@/config/design";

export function SchoolSidebar() {
  return (
    <aside
      className="flex h-full flex-col border-r border-border bg-sidebar"
      style={{ width: design.layout.sidebarWidth }}
    >
      <div className="flex items-center gap-2 border-b border-border px-3 py-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-sm bg-tactical-amber text-background">
          <Eye className="h-4 w-4" />
        </span>
        <div>
          <div className="font-mono text-xs font-bold uppercase tracking-widest text-foreground">
            {design.app.name}
          </div>
          <div className="text-[10px] text-muted-foreground">{design.app.tagline}</div>
        </div>
      </div>
    </aside>
  );
}
