import { useEffect, useState } from "react";

import { fetchAlerts, fetchDecisions } from "./api";
import { AlertPanel } from "./components/AlertPanel";
import { DecisionFeed } from "./components/DecisionFeed";
import { KpiStrip } from "./components/KpiStrip";
import { TransactionDrawer } from "./components/TransactionDrawer";
import type { AlertEvent, DecisionEvent } from "./types";

export default function App() {
  const [decisions, setDecisions] = useState<DecisionEvent[]>([]);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [selected, setSelected] = useState<DecisionEvent | null>(null);

  const [status, setStatus] = useState<"loading" | "live" | "error">("loading");
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timer: number;
    const refresh = async () => {
      try {
        const [nextDecisions, nextAlerts] = await Promise.all([fetchDecisions(), fetchAlerts()]);
        if (!active) return;
        setDecisions(nextDecisions);
        setAlerts(nextAlerts);
        setStatus("live");
        setUpdatedAt(new Date().toLocaleTimeString());
      } catch {
        if (active) setStatus("error");
      } finally {
        if (active) timer = window.setTimeout(refresh, 5000);
      }
    };
    void refresh();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Fraud operations</p>
          <h1>Live transaction decisions</h1>
        </div>
        <div className="model-pill">Model {decisions[0]?.model_version ?? "awaiting transactions"}</div>
      </section>
      <p className={`connection-status ${status}`} role={status === "error" ? "alert" : "status"}>
        {status === "loading" && "Connecting to the scoring service…"}
        {status === "live" && `Connected · Last updated ${updatedAt} · Latest ${decisions.length} transactions`}
        {status === "error" && `Unable to refresh the feed. Retrying automatically.${updatedAt ? ` Showing data from ${updatedAt}.` : ""}`}
      </p>
      {status === "live" && decisions.length === 0 && (
        <p className="empty-state">No scored transactions yet. Start a replay or submit a transaction through the API.</p>
      )}
      <KpiStrip decisions={decisions} />
      <section className="workspace">
        <DecisionFeed decisions={decisions} onSelect={setSelected} />
        <AlertPanel alerts={alerts} />
      </section>
      <TransactionDrawer decision={selected} onClose={() => setSelected(null)} />
    </main>
  );
}
