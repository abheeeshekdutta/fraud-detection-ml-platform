import type { AlertEvent } from "../types";

interface AlertPanelProps {
  alerts: AlertEvent[];
}

export function AlertPanel({ alerts }: AlertPanelProps) {
  return (
    <section className="panel alert-panel">
      <div className="panel-header">
        <h2>Alerts</h2>
        <span>{alerts.length} active</span>
      </div>
      {alerts.length === 0 ? (
        <p className="empty-state">No active model or service alerts.</p>
      ) : (
        <ul>
          {alerts.map((alert) => (
            <li key={alert.alert_id} className={`alert ${alert.severity}`}>
              <strong>{alert.alert_type}</strong>
              <span>{alert.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
