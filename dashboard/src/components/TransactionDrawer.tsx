import { X } from "lucide-react";

import type { DecisionEvent } from "../types";

interface TransactionDrawerProps {
  decision: DecisionEvent | null;
  onClose: () => void;
}

export function TransactionDrawer({ decision, onClose }: TransactionDrawerProps) {
  if (!decision) return null;

  return (
    <aside className="drawer" aria-label="Transaction detail">
      <button className="icon-button" onClick={onClose} aria-label="Close transaction detail">
        <X size={18} />
      </button>
      <p className="eyebrow">Transaction {decision.transaction_id}</p>
      <h2>{decision.decision}</h2>
      <dl className="detail-grid">
        <div>
          <dt>Calibrated probability</dt>
          <dd>{Math.round(decision.calibrated_probability * 100)}%</dd>
        </div>
        <div>
          <dt>Conformal set</dt>
          <dd>{decision.conformal_prediction_set.join(", ")}</dd>
        </div>
        <div>
          <dt>Policy</dt>
          <dd>{decision.decision_policy_version}</dd>
        </div>
        <div>
          <dt>Feature schema</dt>
          <dd>{decision.feature_schema_version}</dd>
        </div>
      </dl>
      <h3>Reason codes</h3>
      <ul className="reason-list">
        {decision.reason_codes.map((reason) => (
          <li key={`${reason.feature}-${reason.direction}`}>
            <span>{reason.feature}</span>
            <span>{reason.direction.replace("_", " ")}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
