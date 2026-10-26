export const pct = (v: number | undefined | null, d = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | undefined | null, d = 3) =>
  v == null || Number.isNaN(v) ? "–" : Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : v.toFixed(d);

export function toCsv(rows: Record<string, unknown>[], columns?: string[]): string {
  const cols = columns ?? Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
  const esc = (x: unknown) => {
    const s = x == null ? "" : String(x);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
}

export function downloadText(filename: string, text: string, mime = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export type FieldCheck = { value?: number; error?: string };
/** Validate one gas input exactly as the API does: required, numeric, finite, non-negative. Never defaults to 0. */
export function checkGas(raw: string, required: boolean): FieldCheck {
  const s = raw.trim();
  if (s === "") return required ? { error: "Required – missing values are not replaced with zero" } : {};
  if (!/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return { error: "Must be a number" };
  const v = Number(s);
  if (!Number.isFinite(v)) return { error: "Must be finite" };
  if (v < 0) return { error: "Concentration cannot be negative" };
  return { value: v };
}
