import { useEffect, useRef, useState } from "react";
import { Send, MessageSquare, X, ClipboardList } from "lucide-react";
import { useStore } from "@/lib/incidentStore";
import { MessageBubble } from "./MessageBubble";
import { IncidentReportForm } from "./IncidentReportForm";
import { design } from "@/config/design";
import { cn } from "@/lib/utils";

type Tab = "chat" | "report";

export function CommunicationWindow({
  viewerRole,
  filterIncidentId,
}: {
  viewerRole: "school" | "police";
  filterIncidentId?: string;
}) {
  const messages = useStore((s) => s.messages);
  const sendMessage = useStore((s) => s.sendMessage);

  const [open, setOpen] = useState(viewerRole === "police");
  const [tab, setTab] = useState<Tab>("chat");
  const [text, setText] = useState("");
  const scrollerRef = useRef<HTMLDivElement>(null);

  const visible = filterIncidentId
    ? messages.filter((m) => m.incidentId === filterIncidentId || m.sender === "system")
    : messages;

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight });
  }, [visible.length, open]);

  const send = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    sendMessage({ sender: viewerRole, text, incidentId: filterIncidentId });
    setText("");
  };

  // Police-mode: render inline (no toggle)
  const inline = viewerRole === "police";

  if (inline) {
    return (
      <div className="flex h-full flex-col rounded-md border border-border bg-surface">
        <Header title="School Comms" tab={tab} setTab={setTab} showReportTab={false} />
        <Body
          scrollerRef={scrollerRef}
          messages={visible}
          viewerRole={viewerRole}
          tab="chat"
          reportedBy="police"
          onReportSubmitted={() => setTab("chat")}
        />
        <Composer text={text} setText={setText} onSubmit={send} />
      </div>
    );
  }

  // School-mode: floating toggleable panel
  return (
    <>
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-4 right-4 z-30 flex items-center gap-2 rounded-full bg-tactical-amber px-4 py-2.5 font-mono text-xs font-bold uppercase tracking-widest text-background shadow-lg hover:opacity-90"
        >
          <MessageSquare className="h-4 w-4" /> Police Comms
        </button>
      )}

      {open && (
        <div
          className="fixed bottom-4 right-4 z-30 flex flex-col rounded-md border border-border bg-popover shadow-2xl"
          style={{
            width: design.layout.commsWindowWidth,
            height: design.layout.commsWindowHeight,
          }}
        >
          <Header
            title="Police Comms"
            tab={tab}
            setTab={setTab}
            showReportTab
            onClose={() => setOpen(false)}
          />
          <Body
            scrollerRef={scrollerRef}
            messages={visible}
            viewerRole={viewerRole}
            tab={tab}
            reportedBy="school"
            onReportSubmitted={() => setTab("chat")}
          />
          {tab === "chat" && <Composer text={text} setText={setText} onSubmit={send} />}
        </div>
      )}
    </>
  );
}

function Header({
  title,
  tab,
  setTab,
  showReportTab,
  onClose,
}: {
  title: string;
  tab: Tab;
  setTab: (t: Tab) => void;
  showReportTab: boolean;
  onClose?: () => void;
}) {
  return (
    <header className="flex items-center justify-between border-b border-border px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-tactical-green animate-tactical-pulse" />
        <h3 className="font-mono text-[11px] uppercase tracking-widest">{title}</h3>
      </div>
      <div className="flex items-center gap-1">
        {showReportTab && (
          <>
            <TabBtn active={tab === "chat"} onClick={() => setTab("chat")}>
              <MessageSquare className="h-3 w-3" /> Chat
            </TabBtn>
            <TabBtn active={tab === "report"} onClick={() => setTab("report")}>
              <ClipboardList className="h-3 w-3" /> Report
            </TabBtn>
          </>
        )}
        {onClose && (
          <button
            onClick={onClose}
            className="ml-1 rounded-sm p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </header>
  );
}

function TabBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1 rounded-sm px-2 py-1 font-mono text-[10px] uppercase tracking-widest",
        active
          ? "bg-tactical-amber/20 text-tactical-amber"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function Body({
  scrollerRef,
  messages,
  viewerRole,
  tab,
  reportedBy,
  onReportSubmitted,
}: {
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  messages: ReturnType<typeof useStore.getState>["messages"];
  viewerRole: "school" | "police";
  tab: Tab;
  reportedBy: string;
  onReportSubmitted: (id: string) => void;
}) {
  if (tab === "report") {
    return (
      <div className="flex-1 overflow-y-auto">
        <IncidentReportForm onSubmitted={onReportSubmitted} reportedBy={reportedBy} />
      </div>
    );
  }
  return (
    <div ref={scrollerRef} className="flex-1 space-y-2 overflow-y-auto p-3">
      {messages.map((m) => (
        <MessageBubble key={m.id} msg={m} viewerRole={viewerRole} />
      ))}
    </div>
  );
}

function Composer({
  text,
  setText,
  onSubmit,
}: {
  text: string;
  setText: (s: string) => void;
  onSubmit: (e: React.FormEvent) => void;
}) {
  return (
    <form
      onSubmit={onSubmit}
      className="flex items-center gap-2 border-t border-border p-2"
    >
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a message…"
        className="flex-1 rounded-sm border border-border bg-input px-2.5 py-1.5 text-xs"
      />
      <button
        type="submit"
        className="inline-flex items-center justify-center rounded-sm bg-tactical-amber px-3 py-1.5 text-background hover:opacity-90"
      >
        <Send className="h-3.5 w-3.5" />
      </button>
    </form>
  );
}
