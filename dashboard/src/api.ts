import type { AlertEvent, DecisionEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchDecisions(): Promise<DecisionEvent[]> {
  try {
    const response = await fetch(`${API_BASE}/predictions`);
    if (!response.ok) return [];
    return response.json();
  } catch {
    return [];
  }
}

export async function fetchAlerts(): Promise<AlertEvent[]> {
  try {
    const response = await fetch(`${API_BASE}/alerts`);
    if (!response.ok) return [];
    return response.json();
  } catch {
    return [];
  }
}
