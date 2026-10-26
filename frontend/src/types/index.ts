export interface ModelInfo {
  id: string; implementation: "A" | "B" | "C"; implementation_label: string; model_name: string;
  feature_set: string; classes: string[]; supports_proba: boolean; trained_on: string;
  provenance: { created_utc: string; seed: number; dataset_sha256: string; n_rows: number; cleaning_protocol: string };
}
export interface FeatureSetInfo {
  id: string; label: string; notebook: string; gases: string[]; positive_required: string[];
  uses_epsilon: boolean; formulas: Record<string, string>; note: string; feature_names: string[];
}
export interface ModelResult {
  model_id: string; implementation: string; model_name: string; feature_set: string;
  status: "ok" | "not_computable"; predicted_class: string | null;
  probabilities: Record<string, number> | null; probability_note: string | null;
  features: Record<string, number> | null; message: string | null;
}
export interface DiagnosisResponse {
  input: Record<string, number | null>; results: ModelResult[];
  agreement: { n_models: number; votes: Record<string, number>; unanimous: boolean }; disclaimer: string;
}
export interface MetricRow {
  experiment: string; implementation: string; feature_set: string; model: string; stage: string; n: number;
  headline_average: string; accuracy?: number; accuracy_std?: number;
  macro_precision?: number; macro_recall?: number; macro_f1?: number;
  weighted_precision?: number; weighted_recall?: number; weighted_f1?: number;
  specificity_weighted?: number; mcc?: number;
}
export interface Confusion { labels: string[]; matrix: number[][] }
export interface ExperimentRow {
  id: string; implementation: string; title: string; kind: string; status: string; created_utc: string;
  seed: number; dataset_sha256: string; n_rows: number; seconds: number; split: Record<string, unknown>;
  provenance: Record<string, any>; warnings: { case?: string; model?: string; warning: string }[]; n_records: number;
}
export interface BatchRow {
  row: number; status: "ok" | "invalid_input" | "not_computable"; predictions: Record<string, string>;
  notes: string[]; [gas: string]: unknown;
}
export interface BatchResponse {
  n_rows: number; n_valid_rows: number; n_invalid_rows: number; issues: { row: number; column: string; issue: string }[];
  models: Record<string, { model_name: string; feature_set: string; implementation: string; class_counts: Record<string, number>; n_not_computable: number }>;
  rows: BatchRow[]; disclaimer: string;
}
export const GASES = [
  { key: "H2", name: "Hydrogen", sub: "H₂", desc: "Key gas for partial discharge" },
  { key: "CH4", name: "Methane", sub: "CH₄", desc: "Low-temperature thermal faults" },
  { key: "C2H6", name: "Ethane", sub: "C₂H₆", desc: "Thermal faults, lower range" },
  { key: "C2H4", name: "Ethylene", sub: "C₂H₄", desc: "High-temperature thermal faults" },
  { key: "C2H2", name: "Acetylene", sub: "C₂H₂", desc: "Arcing / high-energy discharge" },
] as const;
export const CO_FIELD = { key: "CO", name: "Carbon monoxide", sub: "CO", desc: "Recorded only; not used by any model" };
export const IMPL_NAMES: Record<string, string> = { A: "RF comparative ML", B: "DGA feature cases", C: "PCA + ANN" };
export const FS_LABEL: Record<string, string> = {
  rf15: "Engineered 15 (A)", duval3: "Duval %", doern4: "Doernenburg ratios", rogers3: "Rogers ratios",
  pca15: "Engineered 15 (C)", raw5: "Raw 5 gases", iec3: "IEC 3 ratios",
};
export const FAULT_NAMES: Record<string, string> = { Normal: "Normal", PD: "Partial discharge", OH: "Overheating", Arcing: "Arcing" };
