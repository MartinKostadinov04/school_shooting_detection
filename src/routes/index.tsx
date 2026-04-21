import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { readSession } from "@/hooks/useAuth";

export const Route = createFileRoute("/")({
  component: IndexRedirect,
});

function IndexRedirect() {
  const navigate = useNavigate();

  useEffect(() => {
    const session = readSession();
    if (!session) {
      navigate({ to: "/login", replace: true });
    } else if (session.role === "school") {
      navigate({ to: "/school", replace: true });
    } else {
      navigate({ to: "/police", replace: true });
    }
  }, [navigate]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="text-sm font-mono text-muted-foreground">
        TACTICALEYE // initializing...
      </div>
    </div>
  );
}
