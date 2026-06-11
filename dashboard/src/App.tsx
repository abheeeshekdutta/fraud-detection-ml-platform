import { useEffect, useMemo, useState } from "react";

import { fetchAlerts, fetchDecisions } from "./api";
import { AlertPanel } from "./components/AlertPanel";
import { DecisionFeed } from "./components/DecisionFeed";
import { KpiStrip } from "./components/KpiStrip";
import { TransactionDrawer } from "./components/TransactionDrawer";
import type { AlertEvent, DecisionEvent } from "./types";

const fallbackDecision: DecisionEvent = {
  event_id: "sample-1",
  transaction_id: 2987000,
  scored_at: new Date().toISOString(),
  model_version: "synthetic-fraud-model:1",
  feature_schema_version: "v1",
  decision_policy_version: "v1",
  fraud_probability: 0.82,
  calibrated_probability: 0.76,
  conformal_prediction_set: ["fraud"],
  uncertainty: "low",
  decision: "block",
  reason_codes: [{ feature: "TransactionAmt", direction: "increases_risk" }],
  latency_ms: 42,
};

export default function App() {
  const [decisions, setDecisions] = useState<DecisionEvent[]>([]);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [selected, setSelected] = useState<DecisionEvent | null>(null);

  useEffect(() => {
    void fetchDecisions().then(setDecisions);
    void fetchAlerts().then(setAlerts);
    const timer = window.setInterval(() => {
      void fetchDecisions().then(setDecisions);
      void fetchAlerts().then(setAlerts);
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const visibleDecisions = useMemo(
    () => (decisions.length ? decisions : [fallbackDecision]),
    [decisions],
  );

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Fraud operations</p>
          <h1>Live transaction decisions</h1>
        </div>
        <div className="model-pill">Model {visibleDecisions[0]?.model_version ?? "loading"}</div>
      </section>
      <KpiStrip decisions={visibleDecisions} />
      <section className="workspace">
        <DecisionFeed decisions={visibleDecisions} onSelect={setSelected} />
        <AlertPanel alerts={alerts} />
      </section>
      <TransactionDrawer decision={selected} onClose={() => setSelected(null)} />
    </main>
  );
}
