import type { Decision, DecisionEvent } from "../types";

interface KpiStripProps {
  decisions: DecisionEvent[];
}

function rate(decisions: DecisionEvent[], decision: Decision) {
  if (!decisions.length) return "0%";
  const count = decisions.filter((item) => item.decision === decision).length;
  return `${Math.round((count / decisions.length) * 100)}%`;
}

function p95Latency(decisions: DecisionEvent[]) {
  if (!decisions.length) return "0 ms";
  const sorted = [...decisions].map((item) => item.latency_ms).sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1);
  return `${Math.round(sorted[index])} ms`;
}

export function KpiStrip({ decisions }: KpiStripProps) {
  const items = [
    ["Approve rate", rate(decisions, "approve")],
    ["Review rate", rate(decisions, "review")],
    ["Block rate", rate(decisions, "block")],
    ["p95 latency", p95Latency(decisions)],
  ];

  return (
    <section className="kpi-strip">
      {items.map(([label, value]) => (
        <div className="kpi" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}
