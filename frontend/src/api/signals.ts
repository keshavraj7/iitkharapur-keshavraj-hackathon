import { apiGet } from "./client";
import type { RiskSignal } from "../types/signal";

export function getSignals(): Promise<RiskSignal[]> {
  return apiGet<RiskSignal[]>("/api/v1/signals");
}