import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Clock3,
  Filter,
  Gauge,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  Wifi,
} from "lucide-react";

import { getSignals } from "./api/signals";
import type { RiskSignal } from "./types/signal";

type SentimentTone = "positive" | "negative" | "neutral";

function getSentimentTone(score: number): SentimentTone {
  if (score < -0.15) return "negative";
  if (score > 0.15) return "positive";
  return "neutral";
}

function formatScore(score: number): string {
  return `${score > 0 ? "+" : ""}${score.toFixed(2)}`;
}

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function StatCard({
  label,
  value,
  description,
  icon: Icon,
  variant,
}: {
  label: string;
  value: string | number;
  description: string;
  icon: typeof Activity;
  variant: string;
}) {
  return (
    <article className="stat-card">
      <div className={`stat-icon ${variant}`}>
        <Icon size={20} />
      </div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-description">{description}</div>
    </article>
  );
}

function App() {
  const [signals, setSignals] = useState<RiskSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
const [demoLoading, setDemoLoading] = useState(false);
const [notice, setNotice] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [eventFilter, setEventFilter] = useState("All events");
  const [sentimentFilter, setSentimentFilter] = useState("All sentiment");

  const loadSignals = useCallback(async (showLoader = false) => {
    if (showLoader) setLoading(true);

    try {
      const data = await getSignals();

      setSignals(data);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to the backend."
      );
    } finally {
      if (showLoader) setLoading(false);
    }
  }, []);
const runDemoAnalysis = useCallback(async () => {
  setDemoLoading(true);
  setError(null);
  setNotice(null);

  const documents = [
    {
      document_id: "demo-credit-001",
      source_type: "news",
      source_name: "Synthetic Demo",
      title: "JPMorgan reports losses after a debt default",
      text: "JPMorgan reports increasing losses following a debt default.",
    },
    {
      document_id: "demo-earnings-001",
      source_type: "news",
      source_name: "Synthetic Demo",
      title: "NVIDIA reports record revenue growth",
      text: "NVIDIA reports record revenue growth and strong earnings.",
    },
    {
      document_id: "demo-macro-001",
      source_type: "news",
      source_name: "Synthetic Demo",
      title: "Inflation accelerates amid higher interest rates",
      text: "Inflation rises, increasing pressure on the economy and interest rates.",
    },
    {
      document_id: "demo-cyber-001",
      source_type: "news",
      source_name: "Synthetic Demo",
      title: "Company reports data breach after ransomware attack",
      text: "A company confirms a data breach and operational disruption.",
    },
    {
      document_id: "demo-supply-001",
      source_type: "news",
      source_name: "Synthetic Demo",
      title: "Oil prices plunge amid supply chain disruption",
      text: "Oil prices plunge after shipping delays disrupt commodity supply.",
    },
  ];

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/api/v1/signals/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents }),
      }
    );

    if (!response.ok) {
      const details = await response.text();
      throw new Error(`Analysis failed (${response.status}): ${details}`);
    }

    const generated = (await response.json()) as RiskSignal[];

    setSearch("");
    setEventFilter("All events");
    setSentimentFilter("All sentiment");

    await loadSignals(false);

    setNotice(
      `Demo analysis completed: ${generated.length} documents processed.`
    );
  } catch (err) {
    setError(
      err instanceof Error ? err.message : "Demo analysis failed."
    );
  } finally {
    setDemoLoading(false);
  }
}, [loadSignals]);
  useEffect(() => {
    void loadSignals(true);

    const timer = window.setInterval(() => {
      void loadSignals(false);
    }, 5000);

    return () => window.clearInterval(timer);
  }, [loadSignals]);

  const eventOptions = useMemo(
    () => [
      "All events",
      ...Array.from(new Set(signals.map((s) => s.event.type))).sort(),
    ],
    [signals]
  );

  const filteredSignals = useMemo(() => {
    const term = search.toLowerCase().trim();

    return signals.filter((signal) => {
      const sentiment = getSentimentTone(signal.sentiment.score);

      if (eventFilter !== "All events" && signal.event.type !== eventFilter) {
        return false;
      }

      if (
        sentimentFilter !== "All sentiment" &&
        sentiment !== sentimentFilter.toLowerCase()
      ) {
        return false;
      }

      const searchableText = [
        signal.signal_id,
        signal.source.name,
        signal.event.type,
        signal.rationale,
        ...signal.entities.map((e) => `${e.ticker} ${e.company} ${e.sector ?? ""}`),
      ]
        .join(" ")
        .toLowerCase();

      return searchableText.includes(term);
    });
  }, [signals, search, eventFilter, sentimentFilter]);

  const negativeCount = signals.filter(
    (s) => getSentimentTone(s.sentiment.score) === "negative"
  ).length;

  const highImpactCount = signals.filter(
    (s) => s.impact.score >= 7
  ).length;

  const averageImpact = signals.length
    ? (
        signals.reduce((sum, s) => sum + s.impact.score, 0) / signals.length
      ).toFixed(1)
    : "—";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={23} />
          </div>
          <div>
            <div className="brand-name">FINRISK</div>
            <div className="brand-subtitle">Financial Risk Intelligence</div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>API MONITOR</span>
          <span className="status-divider" />
          <Wifi size={16} />
          <span>5s refresh</span>
        </div>
      </header>

      <section className="hero">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-line" />
            REAL-TIME RISK MONITORING
          </div>
          <h1>See the signal behind the noise.</h1>
          <p>
            Transform unstructured financial information into
            structured, actionable risk intelligence.
          </p>
        </div>
        
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
  <button
    className="refresh-button"
    onClick={() => void loadSignals(false)}
    disabled={loading || demoLoading}
  >
    <RefreshCw size={16} />
    Refresh signals
  </button>

  <button
    className="refresh-button"
    onClick={() => void runDemoAnalysis()}
    disabled={demoLoading}
  >
    <Play size={16} />
    {demoLoading ? "Analyzing..." : "Run demo analysis"}
  </button>
</div>
      </section>

      {error && (
        <div className="error-banner">
          <ShieldAlert size={19} />
          <div>
            <strong>Connection issue</strong>
            <p>{error}</p>
            <span>Check that the FastAPI backend is running on port 8000.</span>
          </div>
        </div>
      )}
  {notice && (
  <div
    role="status"
    style={{
      marginTop: 16,
      padding: "12px 16px",
      borderRadius: 12,
      background: "#e8f7ef",
      color: "#136c43",
      border: "1px solid #b7e4cd",
    }}
  >
    {notice}
  </div>
)}
      <section className="metrics-grid">
        <StatCard
          label="TOTAL SIGNALS"
          value={signals.length}
          description="Signals received by the API"
          icon={Activity}
          variant="blue"
        />
        <StatCard
          label="NEGATIVE SENTIMENT"
          value={negativeCount}
          description="Signals with scores below −0.15"
          icon={ArrowDownRight}
          variant="red"
        />
        <StatCard
          label="HIGH-IMPACT EVENTS"
          value={highImpactCount}
          description="Impact score of 7 or higher"
          icon={ShieldAlert}
          variant="amber"
        />
        <StatCard
          label="AVERAGE IMPACT"
          value={averageImpact}
          description="Average score on a 1–10 scale"
          icon={Gauge}
          variant="green"
        />
      </section>

      <section className="signals-panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Activity size={19} />
              Risk signal feed
            </div>
            <p>Validated signals received from the risk engine API.</p>
          </div>

          <div className="feed-meta">
            <span className="live-indicator">
              <span className="status-dot" />
              AUTO-REFRESH
            </span>
            <span className="update-time">
              <Clock3 size={14} />
              {lastUpdated
                ? `Updated ${lastUpdated.toLocaleTimeString("en-IN")}`
                : "Waiting for data"}
            </span>
          </div>
        </div>

        <div className="toolbar">
          <label className="search-field">
            <Search size={17} />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search companies, tickers or events..."
            />
          </label>

          <label className="select-field">
            <Filter size={16} />
            <select
              value={eventFilter}
              onChange={(e) => setEventFilter(e.target.value)}
            >
              {eventOptions.map((event) => (
                <option key={event}>{event}</option>
              ))}
            </select>
          </label>

          <select
            className="sentiment-select"
            value={sentimentFilter}
            onChange={(e) => setSentimentFilter(e.target.value)}
            aria-label="Filter by sentiment"
          >
            <option>All sentiment</option>
            <option>Positive</option>
            <option>Negative</option>
            <option>Neutral</option>
          </select>
        </div>

        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>EVENT CLASSIFICATION</th>
                <th>ENTITY</th>
                <th>SENTIMENT</th>
                <th>IMPACT</th>
                <th>SOURCE / TIME</th>
              </tr>
            </thead>
            <tbody>
              {loading && signals.length === 0 ? (
                <tr>
                  <td colSpan={5} className="empty-state">
                    <RefreshCw className="spinning" size={22} />
                    <strong>Connecting to the risk engine...</strong>
                    <span>Waiting for signals from the backend.</span>
                  </td>
                </tr>
              ) : filteredSignals.length === 0 ? (
                <tr>
                  <td colSpan={5} className="empty-state">
                    <Activity size={26} />
                    <strong>
                      {signals.length === 0
                        ? "No signals received yet"
                        : "No matching signals"}
                    </strong>
                    <span>
                      {signals.length === 0
                        ? "Submit a test signal through the FastAPI documentation at /docs."
                        : "Try changing your search or filters."}
                    </span>
                  </td>
                </tr>
              ) : (
                filteredSignals.map((signal) => {
                  const tone = getSentimentTone(signal.sentiment.score);
                  const primaryEntity = signal.entities[0];

                  return (
                    <tr key={signal.signal_id}>
                      <td>
                        <div className="event-cell">
                          <span className={`event-dot event-${tone}`} />
                          <div>
                            <strong>{signal.event.type}</strong>
                            <span>
                              Event confidence:{" "}
                              {Math.round(signal.event.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                      </td>

                      <td>
                        <div className="entity-cell">
                          <strong>
                            {primaryEntity?.company ?? "Market-wide"}
                          </strong>
                          <span>
                            {signal.entities.length
                              ? signal.entities.map((e) => e.ticker).join(" · ")
                              : "Cross-market"}
                          </span>
                        </div>
                      </td>

                      <td>
                        <div className="sentiment-cell">
                          <span className={`sentiment-pill ${tone}`}>
                            {tone === "negative" ? (
                              <ArrowDownRight size={14} />
                            ) : tone === "positive" ? (
                              <ArrowUpRight size={14} />
                            ) : (
                              <Activity size={14} />
                            )}
                            {tone}
                          </span>
                          <strong className={`score-${tone}`}>
                            {formatScore(signal.sentiment.score)}
                          </strong>
                          <span>
                            {Math.round(signal.sentiment.confidence * 100)}%
                            confidence
                          </span>
                        </div>
                      </td>

                      <td>
                        <div className="impact-cell">
                          <div className="impact-number">
                            <strong>{signal.impact.score}</strong>
                            <span>/10</span>
                          </div>
                          <div className="impact-track">
                            <div
                              className={`impact-fill ${
                                signal.impact.score >= 7
                                  ? "high"
                                  : signal.impact.score >= 4
                                    ? "medium"
                                    : "low"
                              }`}
                              style={{
                                width: `${signal.impact.score * 10}%`,
                              }}
                            />
                          </div>
                          <span className="horizon">
                            Horizon: {signal.impact.horizon}
                          </span>
                        </div>
                      </td>

                      <td>
                        <div className="source-cell">
                          <span className="source-name">
                            {signal.source.url ? (
                              <a
                                href={signal.source.url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                {signal.source.name}
                              </a>
                            ) : (
                              signal.source.name
                            )}
                          </span>
                          <span>{formatDate(signal.event_time)}</span>
                          <span className="source-type">
                            {signal.source.type}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        <footer className="panel-footer">
          <span>
            Showing {filteredSignals.length} of {signals.length} signals
          </span>
          <span>Sorted by API insertion order</span>
        </footer>
      </section>

      <footer className="app-footer">
        <span>FINRISK · Financial Risk Intelligence</span>
        <span>
          <span className="status-dot" />
          Risk engine interface v0.1
        </span>
      </footer>
    </main>
  );
}

export default App;