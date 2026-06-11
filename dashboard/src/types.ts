export type Decision = "approve" | "review" | "block";
export type Severity = "info" | "warning" | "critical";

export interface ReasonCode {
  feature: string;
  direction: "increases_risk" | "decreases_risk";
  contribution?: number | null;
}

export interface DecisionEvent {
  event_id: string;
  transaction_id: number;
  scored_at: string;
  model_version: string;
  feature_schema_version: string;
  decision_policy_version: string;
  fraud_probability: number;
  calibrated_probability: number;
  conformal_prediction_set: string[];
  uncertainty: "low" | "medium" | "high";
  decision: Decision;
  reason_codes: ReasonCode[];
  latency_ms: number;
}

export interface AlertEvent {
  alert_id: string;
  created_at: string;
  severity: Severity;
  alert_type: string;
  message: string;
  metadata: Record<string, unknown>;
}
