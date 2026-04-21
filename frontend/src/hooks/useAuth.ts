/**
 * Mock auth using localStorage. REPLACE with real auth provider.
 * Demo accounts:
 *   school@demo.com / school123 -> School dashboard
 *   police@demo.com / police123 -> Police dashboard
 */
import { useEffect, useState, useCallback } from "react";
import type { AuthUser, Role } from "@/types";

const STORAGE_KEY = "tacticaleye.session";

const ACCOUNTS: Record<
  string,
  { password: string; role: Role; displayName: string }
> = {
  "school@demo.com": { password: "school123", role: "school", displayName: "School Operator" },
  "police@demo.com": { password: "police123", role: "police", displayName: "Dispatch Officer" },
};

export function readSession(): AuthUser | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function login(email: string, password: string): AuthUser {
  const acct = ACCOUNTS[email.toLowerCase().trim()];
  if (!acct || acct.password !== password) {
    throw new Error("Invalid email or password");
  }
  const user: AuthUser = {
    email: email.toLowerCase().trim(),
    role: acct.role,
    displayName: acct.displayName,
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
  return user;
}

export function logout() {
  window.localStorage.removeItem(STORAGE_KEY);
}

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setUser(readSession());
    setReady(true);
    const onStorage = () => setUser(readSession());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const doLogin = useCallback((email: string, password: string) => {
    const u = login(email, password);
    setUser(u);
    return u;
  }, []);

  const doLogout = useCallback(() => {
    logout();
    setUser(null);
  }, []);

  return { user, ready, login: doLogin, logout: doLogout };
}
