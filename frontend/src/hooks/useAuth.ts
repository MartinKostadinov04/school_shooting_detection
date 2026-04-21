/**
 * Auth hook backed by the FastAPI /api/auth/* endpoints.
 * JWT is stored in localStorage under the same key as the previous mock
 * so existing session handling in the rest of the app is unchanged.
 *
 * Demo accounts (seeded on first API startup):
 *   school@demo.com / school123
 *   police@demo.com / police123
 */
import { useEffect, useState, useCallback } from "react";
import type { AuthUser } from "@/types";

const STORAGE_KEY = "tacticaleye.session";
const API_BASE    = (import.meta as unknown as { env: Record<string, string> })
  .env?.VITE_API_BASE_URL ?? "http://localhost:8000";

function _getToken(): string | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed?.token ?? null;
  } catch {
    return null;
  }
}

export function readSession(): AuthUser | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed?.user ?? null;
  } catch {
    return null;
  }
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? "Invalid email or password");
  }
  const data = await res.json();
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ token: data.access_token, user: data.user }),
  );
  return data.user as AuthUser;
}

export function logout(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}

export function useAuth() {
  const [user, setUser]   = useState<AuthUser | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setUser(readSession());
    setReady(true);
    const onStorage = () => setUser(readSession());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const doLogin = useCallback(async (email: string, password: string) => {
    const u = await login(email, password);
    setUser(u);
    return u;
  }, []);

  const doLogout = useCallback(() => {
    logout();
    setUser(null);
  }, []);

  return { user, ready, login: doLogin, logout: doLogout };
}
