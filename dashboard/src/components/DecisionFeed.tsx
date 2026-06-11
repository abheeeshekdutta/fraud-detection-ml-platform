import type { DecisionEvent } from "../types";

interface DecisionFeedProps {
  decisions: DecisionEvent[];
  onSelect: (decision: DecisionEvent) => void;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function DecisionFeed({ decisions, onSelect }: DecisionFeedProps) {
  return (
    <section className="panel decision-feed">
      <div className="panel-header">
        <h2>Decision feed</h2>
        <span>{decisions.length} transactions</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Transaction</th>
            <th>Decision</th>
            <th>Probability</th>
            <th>Uncertainty</th>
            <th>Latency</th>
            <th>Model</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((decision) => (
            <tr key={decision.event_id} onClick={() => onSelect(decision)}>
              <td>{decision.transaction_id}</td>
              <td>
                <span className={`status ${decision.decision}`}>{decision.decision}</span>
              </td>
              <td>{formatPercent(decision.calibrated_probability)}</td>
              <td>{decision.uncertainty}</td>
              <td>{Math.round(decision.latency_ms)} ms</td>
              <td>{decision.model_version}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
