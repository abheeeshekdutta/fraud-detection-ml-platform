import type { AlertEvent, DecisionEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`API returned ${response.status}`);
  return response.json();
}

export const fetchDecisions = () => get<DecisionEvent[]>("/predictions");
export const fetchAlerts = () => get<AlertEvent[]>("/alerts");
