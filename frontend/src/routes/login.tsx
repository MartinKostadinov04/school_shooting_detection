import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Eye, ShieldAlert } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { design } from "@/config/design";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign In — TacticalEye" },
      { name: "description", content: "Secure access to the TacticalEye command center." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const u = login(email, password);
      navigate({ to: u.role === "school" ? "/school" : "/police" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    }
  };

  const fillDemo = (role: "school" | "police") => {
    if (role === "school") {
      setEmail("school@demo.com");
      setPassword("school123");
    } else {
      setEmail("police@demo.com");
      setPassword("police123");
    }
  };

  return (
    <div className="grid min-h-screen grid-cols-1 bg-background bg-tactical-grid lg:grid-cols-2">
      {/* Left brand panel */}
      <div className="hidden flex-col justify-between border-r border-border p-10 lg:flex">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-tactical-amber text-background">
            <Eye className="h-5 w-5" />
          </span>
          <div>
            <div className="font-mono text-sm font-bold uppercase tracking-widest">
              {design.app.name}
            </div>
            <div className="text-[11px] text-muted-foreground">{design.app.tagline}</div>
          </div>
        </div>

        <div>
          <h1 className="text-4xl font-bold leading-tight text-foreground">
            Eyes on every<br />
            <span className="text-tactical-amber">corridor.</span>
          </h1>
          <p className="mt-4 max-w-sm text-sm text-muted-foreground">
            Real-time gunshot, video, and human-in-the-loop incident telemetry.
            One channel between school staff and dispatch.
          </p>
        </div>

        <div className="space-y-2 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-tactical-green" /> Secure channel
          </div>
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-tactical-amber animate-tactical-blink" />
            Listening · gunshot-detection
          </div>
        </div>
      </div>

      {/* Right form */}
      <div className="flex items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <div className="mb-6 flex items-center gap-2 lg:hidden">
            <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-tactical-amber text-background">
              <Eye className="h-5 w-5" />
            </span>
            <div className="font-mono text-sm font-bold uppercase tracking-widest">
              {design.app.name}
            </div>
          </div>

          <h2 className="font-mono text-xs uppercase tracking-widest text-tactical-amber">
            Authenticate
          </h2>
          <p className="mt-1 text-2xl font-semibold text-foreground">Sign in to console</p>

          <form onSubmit={submit} className="mt-6 space-y-3">
            <Field label="Email">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-sm border border-border bg-input px-3 py-2 text-sm"
                placeholder="operator@school.gov"
                required
              />
            </Field>
            <Field label="Password">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-sm border border-border bg-input px-3 py-2 text-sm"
                placeholder="••••••••"
                required
              />
            </Field>

            {error && (
              <div className="rounded-sm border border-tactical-red/40 bg-tactical-red/10 px-3 py-2 text-xs text-tactical-red">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full rounded-sm bg-tactical-amber px-4 py-2.5 font-mono text-xs font-bold uppercase tracking-widest text-background hover:opacity-90"
            >
              Enter Console
            </button>
          </form>

          <div className="mt-6 rounded-md border border-border bg-surface p-3">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Demo accounts
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => fillDemo("school")}
                className="flex items-center gap-2 rounded-sm border border-border bg-background px-2 py-1.5 text-left text-[11px] hover:border-tactical-amber/50"
              >
                <Eye className="h-3.5 w-3.5 text-tactical-amber" />
                <div>
                  <div className="font-medium">School</div>
                  <div className="font-mono text-[9px] text-muted-foreground">
                    school@demo.com
                  </div>
                </div>
              </button>
              <button
                onClick={() => fillDemo("police")}
                className="flex items-center gap-2 rounded-sm border border-border bg-background px-2 py-1.5 text-left text-[11px] hover:border-tactical-amber/50"
              >
                <ShieldAlert className="h-3.5 w-3.5 text-tactical-cyan" />
                <div>
                  <div className="font-medium">Police</div>
                  <div className="font-mono text-[9px] text-muted-foreground">
                    police@demo.com
                  </div>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      {children}
    </label>
  );
}
