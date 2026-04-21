import { Eye, LayoutDashboard, ShieldAlert, MessageSquare, Settings, Menu } from "lucide-react";
import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { design } from "@/config/design";
import { cn } from "@/lib/utils";

const NAV = [
  { label: "Dashboard", icon: LayoutDashboard, href: "/school" as const },
  { label: "Incidents", icon: ShieldAlert, href: "/school" as const },
  { label: "Communications", icon: MessageSquare, href: "/school" as const },
  { label: "Settings", icon: Settings, href: "/school" as const },
];

export function SchoolSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className="flex h-full flex-col border-r border-border bg-sidebar transition-[width] duration-200"
      style={{
        width: collapsed ? design.layout.sidebarWidthCollapsed : design.layout.sidebarWidth,
      }}
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-3">
        <div className="flex items-center gap-2 overflow-hidden">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-sm bg-tactical-amber text-background">
            <Eye className="h-4 w-4" />
          </span>
          {!collapsed && (
            <div>
              <div className="font-mono text-xs font-bold uppercase tracking-widest text-foreground">
                {design.app.name}
              </div>
              <div className="text-[10px] text-muted-foreground">{design.app.tagline}</div>
            </div>
          )}
        </div>
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="rounded-sm p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="Toggle sidebar"
        >
          <Menu className="h-4 w-4" />
        </button>
      </div>

      <nav className="flex-1 space-y-1 p-2">
        {NAV.map((n) => (
          <Link
            key={n.label}
            to={n.href}
            className={cn(
              "flex items-center gap-3 rounded-sm px-2.5 py-2 text-sm text-sidebar-foreground transition-colors hover:bg-sidebar-accent",
              collapsed && "justify-center",
            )}
            activeProps={{ className: "bg-sidebar-accent text-tactical-amber" }}
          >
            <n.icon className="h-4 w-4 shrink-0" />
            {!collapsed && <span className="text-xs">{n.label}</span>}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
