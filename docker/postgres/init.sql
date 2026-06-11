CREATE TABLE IF NOT EXISTS predictions (
  event_id TEXT PRIMARY KEY,
  transaction_id INTEGER NOT NULL,
  scored_at TIMESTAMPTZ NOT NULL,
  model_version TEXT NOT NULL,
  feature_schema_version TEXT NOT NULL,
  decision_policy_version TEXT NOT NULL,
  fraud_probability DOUBLE PRECISION NOT NULL,
  calibrated_probability DOUBLE PRECISION NOT NULL,
  conformal_prediction_set JSONB NOT NULL,
  uncertainty TEXT NOT NULL,
  decision TEXT NOT NULL,
  reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
  latency_ms DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_scored_at ON predictions (scored_at);
CREATE INDEX IF NOT EXISTS idx_predictions_decision ON predictions (decision);

CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL,
  severity TEXT NOT NULL,
  alert_type TEXT NOT NULL,
  message TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts (created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity);
CREATE INDEX IF NOT EXISTS idx_alerts_alert_type ON alerts (alert_type);
