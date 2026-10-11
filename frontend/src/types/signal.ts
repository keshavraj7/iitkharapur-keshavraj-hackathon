export type EventType =
  | "Geopolitical"
  | "Macroeconomic"
  | "Credit Event"
  | "M&A"
  | "Earnings"
  | "Regulatory/Legal"
  | "Product/Tech"
  | "Cyber/Operational"
  | "Supply Chain/Commodity"
  | "Other";

export interface SourceInfo {
  type: "news" | "social" | "replay" | "manual";
  name: string;
  url: string | null;
}

export interface EntityMention {
  ticker: string;
  company: string;
  sector: string | null;
  confidence: number;
}

export interface SentimentResult {
  score: number;
  confidence: number;
}

export interface EventResult {
  type: EventType;
  confidence: number;
}

export interface ImpactResult {
  score: number;
  horizon: "1h" | "1d" | "1w";
}

export interface RiskSignal {
  signal_id: string;
  event_time: string;
  source: SourceInfo;
  entities: EntityMention[];
  sentiment: SentimentResult;
  event: EventResult;
  impact: ImpactResult;
  novelty: number;
  rationale: string;
  model_versions: Record<string, string>;
}