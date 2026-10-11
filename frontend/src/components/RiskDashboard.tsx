import { useEffect, useMemo, useState } from "react";

const API = "http://127.0.0.1:8000/api/v1";

type RiskSignal = {
  signal_id: string;
  event_time: string;
  headline?: string | null;
  source: { name: string; url?: string | null };
  event: { type: string; confidence: number };
  sentiment: { score: number; confidence: number };
  impact: { score: number; horizon: string };
  entities: { ticker: string; company: string; sector?: string | null }[];
  rationale: string;
};

const demoDocuments = [
  {
    document_id: "synthetic-credit-001",
    source_type: "news",
    source_name: "Synthetic Demo",
    title: "JPMorgan reports losses after a debt default",
    text: "JPMorgan reports increasing losses following a debt default."
  },
  {
    document_id: "synthetic-earnings-001",
    source_type: "news",
    source_name: "Synthetic Demo",
    title: "NVIDIA reports record revenue growth",
    text: "NVIDIA reports record revenue growth and strong earnings."
  },
  {
    document_id: "synthetic-macro-001",
    source_type: "news",
    source_name: "Synthetic Demo",
    title: "Inflation accelerates as interest rates rise",
    text: "Inflation rises, increasing pressure on the economy and interest rates."
  },
  {
    document_id: "synthetic-cyber-001",
    source_type: "news",
    source_name: "Synthetic Demo",
    title: "Company confirms data breach after ransomware attack",
    text: "A company confirms a data breach and operational disruption."
  },
  {
    document_id: "synthetic-supply-001",
    source_type: "news",
    source_name: "Synthetic Demo",
    title: "Oil prices plunge amid supply chain disruption",
    text: "Oil prices plunge after shipping delays disrupt commodity supply."
  }
];

export default function RiskDashboard() {
  const [signals, setSignals] = useState<RiskSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [eventFilter, setEventFilter] = useState("All");
  const [minimumImpact, setMinimumImpact] = useState("0");

  async function loadSignals() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/signals?limit=200`);

      if (!response.ok) {
        throw new Error(`API returned HTTP ${response.status}`);
      }

      const result: RiskSignal[] = await response.json();
      setSignals(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load signals."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadSignals();
  }, []);

  async function runDemo() {
    setRunning(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API}/signals/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: demoDocuments })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed with HTTP ${response.status}`);
      }

      const result: RiskSignal[] = await response.json();
      await loadSignals();

      setNotice(`Processed ${result.length} synthetic documents.`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Demo analysis failed."
      );
    } finally {
      setRunning(false);
    }
  }

  const eventTypes = useMemo(
    () => [...new Set(signals.map((signal) => signal.event.type))].sort(),
    [signals]
  );

  const filteredSignals = useMemo(() => {
    const term = search.toLowerCase().trim();

    return signals.filter((signal) => {
      const text = [
        signal.headline,
        signal.source?.name,
        signal.event?.type,
        signal.rationale,
        ...((signal.entities ?? []).flatMap((entity) => [
          entity.ticker,
          entity.company
        ]))
      ].filter(Boolean).join(" ").toLowerCase();

      return (
        text.includes(term) &&
        (eventFilter === "All" || signal.event.type === eventFilter) &&
        signal.impact.score >= Number(minimumImpact)
      );
    });
  }, [signals, search, eventFilter, minimumImpact]);

  const highImpact = signals.filter(
    (signal) => signal.impact.score >= 7
  ).length;

  const negative = signals.filter(
    (signal) => signal.sentiment.score < -0.2
  ).length;

  const averageImpact = signals.length
    ? (
        signals.reduce((sum, signal) => sum + signal.impact.score, 0) /
        signals.length
      ).toFixed(1)
    : "0.0";

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <header className="flex flex-col justify-between gap-5 border-b border-slate-800 pb-7 sm:flex-row sm:items-center">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Financial Risk Intelligence
            </p>
            <h1 className="mt-3 text-3xl font-bold sm:text-4xl">
              Risk Signal Dashboard
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Financial event detection, sentiment and impact monitoring.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => void loadSignals()}
              disabled={loading}
              className="rounded-lg border border-slate-700 px-4 py-2.5 text-sm hover:bg-slate-800 disabled:opacity-50"
            >
              Refresh
            </button>
            <button
              onClick={() => void runDemo()}
              disabled={running}
              className="rounded-lg bg-cyan-400 px-4 py-2.5 text-sm font-bold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
            >
              {running ? "Analyzing..." : "Run demo analysis"}
            </button>
          </div>
        </header>

        <div className="mt-5 flex flex-wrap gap-2 text-xs">
          <span className="rounded-full bg-cyan-950 px-3 py-1.5 text-cyan-300">
            Rule-based NLP baseline
          </span>
          <span className="rounded-full bg-slate-800 px-3 py-1.5 text-slate-300">
            SQLite persistence
          </span>
          <span className="rounded-full bg-slate-800 px-3 py-1.5 text-slate-300">
            Synthetic demo data
          </span>
        </div>

        {error && (
          <div role="alert" className="mt-5 rounded-lg border border-red-800 bg-red-950 p-4 text-sm text-red-300">
            {error}. Confirm that FastAPI is running on port 8000.
          </div>
        )}

        {notice && (
          <div role="status" className="mt-5 rounded-lg border border-emerald-800 bg-emerald-950 p-4 text-sm text-emerald-300">
            {notice}
          </div>
        )}

        <section className="mt-7 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[
            ["Stored signals", signals.length, "Persisted records"],
            ["High-impact events", highImpact, "Impact score 7–10"],
            ["Negative sentiment", negative, "Heuristic score below −0.2"],
            ["Average impact", `${averageImpact}/10`, "Across loaded signals"]
          ].map(([label, value, detail]) => (
            <article key={String(label)} className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <p className="text-sm text-slate-400">{label}</p>
              <p className="mt-3 text-3xl font-bold">
                {loading ? "—" : value}
              </p>
              <p className="mt-2 text-xs text-slate-500">{detail}</p>
            </article>
          ))}
        </section>

        <section className="mt-9">
          <div className="flex flex-col justify-between gap-4 xl:flex-row xl:items-end">
            <div>
              <h2 className="text-xl font-bold">Detected events</h2>
              <p className="mt-1 text-sm text-slate-400">
                Showing {filteredSignals.length} of {signals.length} signals
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search ticker, company, event..."
                aria-label="Search risk signals"
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-sm outline-none focus:border-cyan-400"
              />

              <select
                value={eventFilter}
                onChange={(event) => setEventFilter(event.target.value)}
                aria-label="Filter by event type"
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-sm"
              >
                <option value="All">All event types</option>
                {eventTypes.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>

              <select
                value={minimumImpact}
                onChange={(event) => setMinimumImpact(event.target.value)}
                aria-label="Minimum impact score"
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-sm"
              >
                <option value="0">All impact scores</option>
                <option value="4">Impact 4+</option>
                <option value="7">High impact 7+</option>
                <option value="9">Critical impact 9+</option>
              </select>
            </div>
          </div>

          {loading ? (
            <p className="mt-6 rounded-xl bg-slate-900 p-10 text-center text-slate-400">
              Loading signals...
            </p>
          ) : filteredSignals.length === 0 ? (
            <div className="mt-5 rounded-2xl border border-dashed border-slate-700 p-10 text-center">
              <h3 className="text-lg font-semibold">No matching signals</h3>
              <p className="mt-2 text-sm text-slate-400">
                Run the demo to generate sample financial risk events.
              </p>
            </div>
          ) : (
            <div className="mt-5 grid gap-4 xl:grid-cols-2">
              {filteredSignals.map((signal) => {
                const score = signal.sentiment.score;
                const sentiment = score < -0.2
                  ? "Negative"
                  : score > 0.2 ? "Positive" : "Neutral";

                const tone = sentiment === "Negative"
                  ? "text-red-300 bg-red-950"
                  : sentiment === "Positive"
                    ? "text-emerald-300 bg-emerald-950"
                    : "text-slate-300 bg-slate-800";

                return (
                  <article key={signal.signal_id} className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="rounded-full bg-cyan-950 px-3 py-1 text-xs font-semibold text-cyan-300">
                        {signal.event.type}
                      </span>
                      <span className="text-xs text-slate-500">
                        {new Date(signal.event_time).toLocaleString()}
                      </span>
                    </div>

                    <h3 className="mt-4 text-lg font-semibold leading-7">
                      {signal.headline || "Financial event detected"}
                    </h3>

                    <p className="mt-1 text-sm text-slate-400">
                      {signal.source?.name || "Unknown source"}
                      {signal.source?.url && (
                        <>
                          {" · "}
                          <a
                            href={signal.source.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-cyan-400 hover:underline"
                          >
                            Source
                          </a>
                        </>
                      )}
                    </p>

                    <div className="mt-4 flex flex-wrap gap-2 text-xs">
                      <span className={`rounded-full px-3 py-1.5 ${tone}`}>
                        {sentiment} sentiment · {score.toFixed(2)}
                      </span>
                      <span className="rounded-full bg-slate-800 px-3 py-1.5">
                        Impact {signal.impact.score}/10
                      </span>
                      <span className="rounded-full bg-slate-800 px-3 py-1.5">
                        Horizon {signal.impact.horizon}
                      </span>
                    </div>

                    <div className="mt-4 h-1.5 rounded-full bg-slate-800">
                      <div
                        className={`h-1.5 rounded-full ${
                          signal.impact.score >= 7
                            ? "bg-red-400"
                            : "bg-cyan-400"
                        }`}
                        style={{
                          width: `${Math.max(0, Math.min(10, signal.impact.score)) * 10}%`
                        }}
                      />
                    </div>

                    <div className="mt-4 flex flex-wrap gap-2">
                      {signal.entities.length ? signal.entities.map((entity) => (
                        <span key={entity.ticker} className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-300">
                          <strong>{entity.ticker}</strong> · {entity.company}
                        </span>
                      )) : (
                        <span className="text-xs text-slate-500">No recognized entities</span>
                      )}
                    </div>

                    <details className="mt-4 border-t border-slate-800 pt-4">
                      <summary className="cursor-pointer text-sm font-semibold text-slate-300">
                        Reasoning and confidence
                      </summary>
                      <p className="mt-3 text-sm leading-6 text-slate-400">
                        {signal.rationale}
                      </p>
                      <p className="mt-2 text-xs text-slate-500">
                        Event confidence: {signal.event.confidence.toFixed(2)}
                        {" · "}Sentiment confidence: {signal.sentiment.confidence.toFixed(2)}
                      </p>
                    </details>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <footer className="mt-10 border-t border-slate-800 pt-5 text-xs leading-5 text-slate-500">
          Experimental, rule-based indicators for demonstration and research.
          Scores are not calibrated probabilities, investment advice, or validated forecasts.
        </footer>
      </div>
    </main>
  );
}
