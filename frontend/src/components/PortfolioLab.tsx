import { useState } from "react";

const API = "http://127.0.0.1:8000/api/v1";

const eventTypes = [
  "Geopolitical",
  "Macroeconomic",
  "Credit Event",
  "M&A",
  "Earnings",
  "Regulatory/Legal",
  "Product/Tech",
  "Cyber/Operational",
  "Supply Chain/Commodity",
  "Other",
] as const;

type RebalancePosition = {
  ticker: string;
  company: string;
  base_weight: number;
  current_weight: number;
  proposed_weight: number;
  change: number;
  signal_score: number;
  event_override: boolean;
  rationale: string;
};

type RebalanceResult = {
  positions: RebalancePosition[];
  signals_used: number;
  one_way_turnover: number;
  mandatory_turnover: number;
  discretionary_turnover: number;
  max_turnover: number;
  turnover_cap_applied: boolean;
  summary: string;
};

type StressPosition = {
  position_id: string;
  name: string;
  asset_type: string;
  market_value: number;
  pnl: number;
  loss_contribution: number;
  explanation: string;
};

type StressResult = {
  event_type: string;
  impact_score: number;
  starting_nav: number;
  total_pnl: number;
  net_loss: number;
  loss_pct: number;
  gross_losses: number;
  gross_gains: number;
  positions: StressPosition[];
  assumptions: Record<string, number>;
  summary: string;
};

function number(value: number) {
  return value.toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function Card({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <article
      style={{
        padding: 18,
        borderRadius: 12,
        background: "#ffffff",
        border: "1px solid #dfe7f0",
        minWidth: 0,
      }}
    >
      <div style={{ color: "#64748b", fontSize: 12, fontWeight: 700 }}>
        {label}
      </div>
      <div
        style={{
          fontSize: 24,
          fontWeight: 750,
          marginTop: 8,
          overflowWrap: "anywhere",
        }}
      >
        {value}
      </div>
      {detail && (
        <div style={{ color: "#64748b", fontSize: 12, marginTop: 6 }}>
          {detail}
        </div>
      )}
    </article>
  );
}

export default function PortfolioLab() {
  const [rebalance, setRebalance] = useState<RebalanceResult | null>(null);
  const [stress, setStress] = useState<StressResult | null>(null);

  const [eventType, setEventType] =
    useState<(typeof eventTypes)[number]>("Credit Event");
  const [impact, setImpact] = useState(9);

  const [loadingRebalance, setLoadingRebalance] = useState(false);
  const [loadingStress, setLoadingStress] = useState(false);
  const [error, setError] = useState("");

  async function requestJson<T>(url: string): Promise<T> {
    const response = await fetch(url);

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`HTTP ${response.status}: ${body}`);
    }

    return response.json() as Promise<T>;
  }

  async function runRebalance() {
    setLoadingRebalance(true);
    setError("");

    try {
      const result = await requestJson<RebalanceResult>(
        `${API}/rebalancer/demo`
      );
      setRebalance(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Rebalancing request failed."
      );
    } finally {
      setLoadingRebalance(false);
    }
  }

  async function runStressTest() {
    setLoadingStress(true);
    setError("");

    try {
      const result = await requestJson<StressResult>(
        `${API}/stress-test/demo?event_type=${encodeURIComponent(eventType)}&impact_score=${impact}`
      );
      setStress(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Stress-test request failed."
      );
    } finally {
      setLoadingStress(false);
    }
  }

  const sectionStyle = {
    marginTop: 24,
    padding: 20,
    borderRadius: 16,
    background: "#f8fafc",
    border: "1px solid #dfe7f0",
  };

  const buttonStyle = {
    border: 0,
    borderRadius: 9,
    padding: "11px 16px",
    background: "#194a78",
    color: "#ffffff",
    fontWeight: 700,
    cursor: "pointer",
  } as const;

  const tableStyle = {
    width: "100%",
    borderCollapse: "collapse" as const,
    fontSize: 13,
    marginTop: 16,
  };

  const cellStyle = {
    textAlign: "left" as const,
    padding: "12px 10px",
    borderBottom: "1px solid #e2e8f0",
    verticalAlign: "top" as const,
  };

  return (
    <section style={{ marginTop: 32 }}>
      <div style={{ marginBottom: 20 }}>
        <div
          style={{
            color: "#2563a6",
            fontSize: 12,
            fontWeight: 800,
            letterSpacing: "0.14em",
          }}
        >
          PORTFOLIO ANALYTICS
        </div>
        <h2 style={{ fontSize: 25, margin: "8px 0" }}>Portfolio Lab</h2>
        <p style={{ color: "#64748b", lineHeight: 1.6, margin: 0 }}>
          Explore signal-driven allocation changes and event-based portfolio
          stress scenarios using a synthetic portfolio.
        </p>
      </div>

      {error && (
        <div
          role="alert"
          style={{
            marginBottom: 16,
            padding: 14,
            color: "#991b1b",
            background: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: 10,
            overflowWrap: "anywhere",
          }}
        >
          {error}
        </div>
      )}

      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0, fontSize: 19 }}>
          A. Event-driven portfolio rebalancer
        </h3>
        <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.6 }}>
          Calculate constrained target weights using available stored risk
          signals. This example uses the backend's predefined synthetic
          portfolio.
        </p>

        <button
          style={buttonStyle}
          onClick={() => void runRebalance()}
          disabled={loadingRebalance}
        >
          {loadingRebalance ? "Calculating..." : "Run rebalancer"}
        </button>

        {rebalance && (
          <>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))",
                gap: 12,
                marginTop: 18,
              }}
            >
              <Card
                label="SIGNALS USED"
                value={String(rebalance.signals_used)}
              />
              <Card
                label="ONE-WAY TURNOVER"
                value={percent(rebalance.one_way_turnover)}
                detail="Actual portfolio turnover"
              />
              <Card
                label="DISCRETIONARY TURNOVER"
                value={percent(rebalance.discretionary_turnover)}
                detail={`Budget ${percent(rebalance.max_turnover)}`}
              />
            </div>

            <div
              style={{
                marginTop: 14,
                overflowX: "auto",
                background: "#fff",
                borderRadius: 10,
              }}
            >
              <table style={tableStyle}>
                <thead>
                  <tr>
                    {[
                      "Holding",
                      "Current",
                      "Proposed",
                      "Change",
                      "Signal",
                      "Override",
                    ].map((label) => (
                      <th key={label} style={cellStyle}>{label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rebalance.positions.map((position) => (
                    <tr key={position.ticker}>
                      <td style={cellStyle}>
                        <strong>{position.ticker}</strong>
                        <div style={{ color: "#64748b", marginTop: 4 }}>
                          {position.company}
                        </div>
                      </td>
                      <td style={cellStyle}>
                        {percent(position.current_weight)}
                      </td>
                      <td style={cellStyle}>
                        <strong>{percent(position.proposed_weight)}</strong>
                      </td>
                      <td
                        style={{
                          ...cellStyle,
                          color:
                            position.change < 0
                              ? "#b91c1c"
                              : position.change > 0
                                ? "#047857"
                                : "#64748b",
                        }}
                      >
                        {position.change > 0 ? "+" : ""}
                        {percent(position.change)}
                      </td>
                      <td style={cellStyle}>
                        {position.signal_score.toFixed(2)}
                      </td>
                      <td style={cellStyle}>
                        {position.event_override ? "Yes" : "No"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p style={{ fontSize: 12, color: "#64748b", lineHeight: 1.6 }}>
              {rebalance.summary}
            </p>

            <details>
              <summary style={{ cursor: "pointer", fontSize: 13 }}>
                View allocation rationale
              </summary>
              {rebalance.positions.map((position) => (
                <p
                  key={position.ticker}
                  style={{ fontSize: 12, lineHeight: 1.6 }}
                >
                  <strong>{position.ticker}:</strong> {position.rationale}
                </p>
              ))}
            </details>
          </>
        )}
      </section>

      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0, fontSize: 19 }}>
          B. Portfolio stress tester
        </h3>
        <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.6 }}>
          Apply an illustrative adverse-event scenario to a synthetic multi-asset
          portfolio. Shocks are assumptions, not market forecasts.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 14,
            alignItems: "end",
          }}
        >
          <label style={{ display: "grid", gap: 7, fontSize: 13 }}>
            Event scenario
            <select
              value={eventType}
              onChange={(event) =>
                setEventType(event.target.value as (typeof eventTypes)[number])
              }
              style={{
                padding: 11,
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                background: "#fff",
              }}
            >
              {eventTypes.map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </label>

          <label style={{ display: "grid", gap: 7, fontSize: 13 }}>
            Impact score: <strong>{impact}/10</strong>
            <input
              type="range"
              min="1"
              max="10"
              value={impact}
              onChange={(event) => setImpact(Number(event.target.value))}
            />
          </label>

          <button
            style={buttonStyle}
            onClick={() => void runStressTest()}
            disabled={loadingStress}
          >
            {loadingStress ? "Simulating..." : "Run stress test"}
          </button>
        </div>

        {stress && (
          <>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))",
                gap: 12,
                marginTop: 18,
              }}
            >
              <Card
                label="STARTING NAV"
                value={number(stress.starting_nav)}
              />
              <Card
                label="NET P&L"
                value={number(stress.total_pnl)}
                detail={`Net loss: ${number(stress.net_loss)}`}
              />
              <Card
                label="PORTFOLIO LOSS"
                value={percent(stress.loss_pct)}
                detail="Net loss divided by starting NAV"
              />
              <Card
                label="GROSS LOSSES"
                value={number(stress.gross_losses)}
                detail={`Gross gains: ${number(stress.gross_gains)}`}
              />
            </div>

            <h4 style={{ margin: "22px 0 8px" }}>
              Largest position losses
            </h4>

            <div style={{ overflowX: "auto", background: "#fff", borderRadius: 10 }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    {["Position", "Asset type", "P&L", "Loss contribution"].map(
                      (label) => (
                        <th key={label} style={cellStyle}>{label}</th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {stress.positions.map((position) => (
                    <tr key={position.position_id}>
                      <td style={cellStyle}>
                        <strong>{position.name}</strong>
                        <div style={{ color: "#64748b", marginTop: 4 }}>
                          {position.position_id}
                        </div>
                      </td>
                      <td style={cellStyle}>{position.asset_type}</td>
                      <td
                        style={{
                          ...cellStyle,
                          color: position.pnl < 0 ? "#b91c1c" : "#047857",
                        }}
                      >
                        {number(position.pnl)}
                      </td>
                      <td style={cellStyle}>
                        {number(position.loss_contribution)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <details style={{ marginTop: 16 }}>
              <summary style={{ cursor: "pointer", fontSize: 13 }}>
                View scenario assumptions and explanations
              </summary>
              <pre
                style={{
                  marginTop: 12,
                  padding: 14,
                  background: "#fff",
                  borderRadius: 8,
                  fontSize: 12,
                  overflowX: "auto",
                  whiteSpace: "pre-wrap",
                }}
              >
                {JSON.stringify(stress.assumptions, null, 2)}
              </pre>

              {stress.positions.map((position) => (
                <p
                  key={position.position_id}
                  style={{ fontSize: 12, lineHeight: 1.6 }}
                >
                  <strong>{position.name}:</strong> {position.explanation}
                </p>
              ))}
            </details>

            <p style={{ fontSize: 12, color: "#64748b", lineHeight: 1.6 }}>
              {stress.summary}
            </p>
          </>
        )}
      </section>
    </section>
  );
}
