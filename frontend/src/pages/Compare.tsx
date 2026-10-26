import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Download } from "lucide-react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Confusion, ExperimentRow, FS_LABEL, MetricRow } from "../types";
import { Badge, Card, CHART_COLORS, ErrorBox, Loading, Notice, PageHeader } from "../components/ui";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { num, pct } from "../utils/format";

const EXPS = [
  { id: "dga_main", label: "B · DGA feature cases × 6 classifiers" },
  { id: "rfrep_main", label: "A · RF comparative ML (SMOTE + grid search)" },
  { id: "pcaann_main", label: "C · PCA + ANN" },
];
const METRICS: { key: keyof MetricRow; label: string; avg: string }[] = [
  { key: "accuracy", label: "Accuracy", avg: "" },
  { key: "macro_f1", label: "F1", avg: "macro" }, { key: "macro_precision", label: "Precision", avg: "macro" }, { key: "macro_recall", label: "Recall", avg: "macro" },
  { key: "weighted_f1", label: "F1", avg: "weighted" }, { key: "weighted_precision", label: "Precision", avg: "weighted" }, { key: "weighted_recall", label: "Recall", avg: "weighted" },
  { key: "specificity_weighted", label: "Specificity", avg: "weighted" }, { key: "mcc", label: "MCC", avg: "" },
];
const STAGE_LABEL: Record<string, string> = {
  test: "Held-out test (20%)", cv10_train: "10-fold CV on training 80%", gridsearch_cv: "Grid-search CV (3×3 repeated) on training 80%",
  mean_10_random_splits: "Mean of 10 random 80/20 splits",
};
const PROTOCOLS = [
  { nb: "A", data: "Dedup + conflicting-label removal", split: "Stratified 80/20, seed 42", val: "3×3 repeated stratified CV grid search (train only)", feats: "15 engineered (log1p gases, 6 ratios, TCG, Duval %)", avg: "macro P/R/F1 + accuracy", imb: "SMOTE (k=1) inside pipeline" },
  { nb: "B", data: "Dedup only", split: "Stratified 80/20, seed 42", val: "10-fold CV predictions on train; fixed Table-4 params", feats: "Duval % (3) · Doernenburg ratios (4) · Rogers ratios (3)", avg: "weighted P/R/F1, specificity, MCC, accuracy", imb: "none" },
  { nb: "C", data: "Dedup only", split: "Stratified 80/20, seed 42 (+10 random splits)", val: "5-fold CV to choose PCA components k", feats: "15 engineered (5 gases, TCH, 9 ratios)", avg: "accuracy (+ per-class report)", imb: "none" },
];

export default function Compare() {
  const [exp, setExp] = useState("dga_main");
  const metrics = useApi<MetricRow[]>(`/experiments/${exp}/metrics`);
  const full = useApi<any>(`/experiments/${exp}`);
  const reg = useApi<ExperimentRow[]>("/experiments");
  const [stage, setStage] = useState("test");
  const [metric, setMetric] = useState<keyof MetricRow>("accuracy");
  const [fsel, setFsel] = useState<string>("all");
  const [cmKey, setCmKey] = useState("");
  const [cm, setCm] = useState<{ confusion: Confusion; per_class: Record<string, any> | null } | null>(null);
  const [cmErr, setCmErr] = useState<string | null>(null);

  const rows = metrics.data ?? [];
  const stages = useMemo(() => Array.from(new Set(rows.map((r) => r.stage))), [rows]);
  useEffect(() => { if (stages.length && !stages.includes(stage)) setStage(stages.includes("test") ? "test" : stages[0]); }, [stages]); // eslint-disable-line
  const fsets = useMemo(() => Array.from(new Set(rows.filter((r) => r.stage === stage).map((r) => r.feature_set))), [rows, stage]);
  useEffect(() => { setFsel("all"); }, [exp]);
  const available = METRICS.filter((m) => rows.some((r) => r.stage === stage && r[m.key] != null));
  useEffect(() => { if (available.length && !available.some((m) => m.key === metric)) setMetric(available[0].key); }, [available.length, stage, exp]); // eslint-disable-line

  const shown = rows.filter((r) => r.stage === stage && (fsel === "all" || r.feature_set === fsel));
  const metricMeta = METRICS.find((m) => m.key === metric);
  const models = Array.from(new Set(shown.map((r) => r.model)));
  const series = Array.from(new Set(shown.map((r) => r.feature_set)));
  const chart = models.map((m) => ({ model: m, ...Object.fromEntries(series.map((f) => { const r = shown.find((x) => x.model === m && x.feature_set === f); return [f, r && r[metric] != null ? +((r[metric] as number) * 100).toFixed(1) : null]; })) }));

  const cmOptions = useMemo(() => {
    const t = rows.filter((r) => r.stage === "test" || (exp === "pcaann_main" && r.model.startsWith("PCA + ANN (k=")));
    return t.filter((r) => exp !== "pcaann_main" || r.stage === "test").map((r) => ({
      key: exp === "dga_main" ? `${r.feature_set}|${r.model}` : exp === "pcaann_main" ? "PCA + ANN" : r.model,
      label: exp === "dga_main" ? `${FS_LABEL[r.feature_set]} · ${r.model}` : r.model }));
  }, [rows, exp]);
  useEffect(() => { setCmKey(cmOptions[0]?.key ?? ""); }, [cmOptions]);
  useEffect(() => {
    if (!cmKey) return; setCmErr(null);
    api.get<any>(`/experiments/${exp}/confusion?model=${encodeURIComponent(cmKey)}`).then(setCm).catch((e) => { setCm(null); setCmErr(e.message); });
  }, [cmKey, exp]);

  // ---- computed research summary (no hard-coded winners) ----
  const summary = useMemo(() => {
    const out: string[] = [];
    const t = rows.filter((r) => r.stage === stage && r[metric] != null && r.n > 0);
    if (!t.length) return out;
    const sorted = [...t].sort((a, b) => (b[metric] as number) - (a[metric] as number));
    const best = sorted[0]; const label = (r: MetricRow) => `${r.model}${r.feature_set && r.feature_set !== "see model name" ? ` on ${FS_LABEL[r.feature_set] ?? r.feature_set}` : ""}`;
    const unit = 1 / best.n;
    const ties = sorted.slice(1).filter((r) => (best[metric] as number) - (r[metric] as number) <= unit + 1e-9);
    out.push(`Under "${STAGE_LABEL[stage] ?? stage}" (${metricMeta?.avg ? metricMeta.avg + " " : ""}${metricMeta?.label}), the highest value in this experiment is ${pct(best[metric] as number)} for ${label(best)} (n=${best.n}).`);
    if (ties.length) out.push(`${ties.length} other configuration${ties.length > 1 ? "s are" : " is"} within one sample (${pct(unit)}) of it, so the ordering at the top is not statistically meaningful.`);
    const worst = sorted[sorted.length - 1];
    out.push(`The lowest is ${pct(worst[metric] as number)} for ${label(worst)}.`);
    // largest generalisation gap
    const cvStage = rows.some((r) => r.stage === "cv10_train") ? "cv10_train" : rows.some((r) => r.stage === "gridsearch_cv") ? "gridsearch_cv" : null;
    if (cvStage && stage === "test") {
      const gaps = rows.filter((r) => r.stage === "test" && r.accuracy != null).map((r) => {
        const c = rows.find((x) => x.stage === cvStage && x.model === r.model && x.feature_set === r.feature_set);
        return c?.accuracy != null ? { r, gap: (c.accuracy - (r.accuracy as number)) } : null; }).filter(Boolean) as { r: MetricRow; gap: number }[];
      if (gaps.length) { const g = [...gaps].sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))[0];
        out.push(`The largest gap between cross-validation and test accuracy is ${(g.gap * 100).toFixed(1)} points (${label(g.r)}; positive = CV higher than test).`); }
    }
    // most confused class pair across all models in this experiment
    const cms: Confusion[] = [];
    if (full.data?.confusion) Object.values<any>(full.data.confusion).forEach((c) => cms.push(c));
    else if (full.data?.final?.confusion) cms.push(full.data.final.confusion);
    const pairs: Record<string, number> = {};
    cms.forEach((c) => c.matrix.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v) { const k = `${c.labels[i]} → ${c.labels[j]}`; pairs[k] = (pairs[k] ?? 0) + v; } })));
    const top = Object.entries(pairs).sort((a, b) => b[1] - a[1])[0];
    if (top) out.push(`Across ${cms.length} confusion matrix${cms.length > 1 ? "es" : ""} in this experiment, the most frequent error is true ${top[0].replace(" → ", " predicted as ")} (${top[1]} test-set errors in total).`);
    return out;
  }, [rows, stage, metric, full.data]); // eslint-disable-line

  if (metrics.error) return <ErrorBox message={metrics.error} onRetry={metrics.reload} />;
  const exInfo = reg.data?.find((e) => e.id === exp);

  return (
    <>
      <PageHeader title="Research model comparison" description="Results are always shown within one experiment and one evaluation protocol. The three notebooks use different data cleaning, features, splits and metric averaging, so their numbers are not merged into a single ranking."
        actions={<a className="btn-secondary" href={api.url(`/experiments/${exp}/export`)} download><Download className="h-4 w-4" />Export metrics CSV</a>} />
      <Notice tone="warn"><b>Why results may not be comparable:</b> notebook A removes conflicting-label rows (442 records) while B and C do not (452); A reports macro averages and tunes with SMOTE + grid search, B reports weighted averages with fixed parameters, C mostly reports accuracy; "train" is cross-validated in B but resubstitution in C. Test sets hold ~90 records, with 1–2 Arcing samples.</Notice>

      <div className="mt-5 grid gap-3 md:grid-cols-4">
        <label className="text-xs font-medium text-slate-600 md:col-span-2">Experiment
          <select className="input mt-1" value={exp} onChange={(e) => setExp(e.target.value)}>{EXPS.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}</select></label>
        <label className="text-xs font-medium text-slate-600">Evaluation protocol
          <select className="input mt-1" value={stage} onChange={(e) => setStage(e.target.value)}>{stages.map((s) => <option key={s} value={s}>{STAGE_LABEL[s] ?? s}</option>)}</select></label>
        <label className="text-xs font-medium text-slate-600">Metric
          <select className="input mt-1" value={String(metric)} onChange={(e) => setMetric(e.target.value as keyof MetricRow)}>{available.map((m) => <option key={String(m.key)} value={String(m.key)}>{m.avg ? `${m.label} (${m.avg})` : m.label}</option>)}</select></label>
        {fsets.length > 1 && <label className="text-xs font-medium text-slate-600 md:col-span-2">Feature method
          <select className="input mt-1" value={fsel} onChange={(e) => setFsel(e.target.value)}><option value="all">All feature sets</option>{fsets.map((f) => <option key={f} value={f}>{FS_LABEL[f] ?? f}</option>)}</select></label>}
      </div>

      {metrics.loading ? <Loading /> : (
        <div className="mt-5 grid gap-5 xl:grid-cols-3">
          <Card className="xl:col-span-2" title={`${metricMeta?.label ?? "Metric"}${metricMeta?.avg ? ` (${metricMeta.avg} average)` : ""} · ${STAGE_LABEL[stage] ?? stage}`} subtitle={exInfo ? `${exInfo.n_rows} cleaned records · seed ${exInfo.seed} · computed ${exInfo.created_utc?.slice(0, 10)}` : undefined}>
            <div className="h-80"><ResponsiveContainer><BarChart data={chart} margin={{ bottom: 15 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" fontSize={11} interval={0} angle={models.length > 6 ? -20 : 0} textAnchor={models.length > 6 ? "end" : "middle"} height={models.length > 6 ? 60 : 30} />
              <YAxis unit="%" domain={[0, 100]} fontSize={11} /><Tooltip formatter={(v: number) => `${v}%`} />{series.length > 1 && <Legend formatter={(v) => FS_LABEL[v] ?? v} />}
              {series.map((f, i) => <Bar key={f} dataKey={f} name={FS_LABEL[f] ?? f} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[3, 3, 0, 0]} />)}</BarChart></ResponsiveContainer></div>
            {stage === "test" && <p className="mt-2 text-[11px] text-slate-500">Single held-out split – small differences are within sampling noise.</p>}
          </Card>
          <Card title="Research summary" subtitle="Generated from the selected results">
            {summary.length ? <ul className="space-y-2 text-sm text-slate-700">{summary.map((s, i) => <li key={i} className="leading-snug">• {s}</li>)}</ul> : <p className="text-sm text-slate-500">No data for this selection.</p>}
          </Card>

          <Card className="xl:col-span-3" title="All metrics for this protocol">
            <div className="overflow-x-auto rounded-md border border-slate-200"><table className="num w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs text-slate-600"><tr><th className="px-3 py-2">Feature set</th><th className="px-3 py-2">Model</th><th className="px-3 py-2 text-right">n</th><th className="px-3 py-2 text-right">Accuracy</th><th className="px-3 py-2 text-right">Macro F1</th><th className="px-3 py-2 text-right">Weighted F1</th><th className="px-3 py-2 text-right">Specificity (w)</th><th className="px-3 py-2 text-right">MCC</th></tr></thead>
              <tbody className="divide-y divide-slate-100">{shown.map((r, i) => <tr key={i} className="hover:bg-slate-50"><td className="px-3 py-1.5">{FS_LABEL[r.feature_set] ?? r.feature_set}</td><td className="px-3 py-1.5 font-medium">{r.model}</td><td className="px-3 py-1.5 text-right">{r.n}</td>
                <td className="px-3 py-1.5 text-right">{pct(r.accuracy)}{r.accuracy_std != null && <span className="text-slate-400"> ±{(r.accuracy_std * 100).toFixed(1)}</span>}</td><td className="px-3 py-1.5 text-right">{pct(r.macro_f1)}</td><td className="px-3 py-1.5 text-right">{pct(r.weighted_f1)}</td><td className="px-3 py-1.5 text-right">{pct(r.specificity_weighted)}</td><td className="px-3 py-1.5 text-right">{num(r.mcc, 3)}</td></tr>)}</tbody></table></div>
            <p className="mt-2 text-[11px] text-slate-500">"–" means the original notebook protocol does not compute that metric. Macro and weighted averages are both derived from the same predictions in this app; the notebook's own headline average is {shown[0]?.headline_average}.</p>
          </Card>

          <Card className="xl:col-span-2" title="Confusion matrix (held-out test set)" action={<select className="input !w-auto !py-1 text-xs" value={cmKey} onChange={(e) => setCmKey(e.target.value)} aria-label="Select model">{cmOptions.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}</select>}>
            {cmErr ? <ErrorBox message={cmErr} /> : cm ? <ConfusionMatrix cm={cm.confusion} /> : <Loading />}
          </Card>
          <Card title="Class-wise performance">
            {cm?.per_class ? <table className="num w-full text-sm"><thead className="text-xs text-slate-500"><tr><th className="py-1 text-left">Class</th><th className="text-right">P</th><th className="text-right">R</th><th className="text-right">F1</th><th className="text-right">n</th></tr></thead>
              <tbody className="divide-y divide-slate-100">{Object.entries(cm.per_class).map(([k, v]: any) => <tr key={k}><td className="py-1.5 font-medium">{k}{v.support <= 2 && <Badge tone="amber">n={v.support}</Badge>}</td><td className="text-right">{pct(v.precision, 0)}</td><td className="text-right">{pct(v.recall, 0)}</td><td className="text-right">{pct(v["f1-score"], 0)}</td><td className="text-right">{v.support}</td></tr>)}</tbody></table> : <p className="text-sm text-slate-500">No class-wise data.</p>}
            <p className="mt-2 text-[11px] text-slate-500">Classes with 1–2 test samples are flagged: their scores are 0%, 50% or 100% by construction.</p>
          </Card>

          {full.data?.feature_importance && (
            <Card className="xl:col-span-3" title="Random Forest permutation importance (notebook A)" subtitle="Mean drop in test accuracy when a feature is shuffled, 20 repeats">
              <div className="h-72"><ResponsiveContainer><BarChart data={full.data.feature_importance.slice(0, 15).map((f: any) => ({ f: f.feature, v: +(f.importance * 100).toFixed(2) }))} layout="vertical" margin={{ left: 30 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" fontSize={11} unit=" pp" /><YAxis dataKey="f" type="category" width={110} fontSize={11} /><Tooltip /><Bar dataKey="v" fill="#0e8a8c" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div>
              <p className="mt-2 text-[11px] text-slate-500">Importance values near zero (and negative ones) mean the test set cannot distinguish the feature's contribution; the test set has ~89 records.</p>
            </Card>)}

          <Card className="xl:col-span-3" title="Model configurations and evaluation protocols" subtitle="Documented from the source notebooks">
            <div className="overflow-x-auto rounded-md border border-slate-200"><table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-600"><tr>{["Notebook", "Cleaning", "Split", "Validation / tuning", "Features", "Imbalance", "Reported averaging"].map((h) => <th key={h} className="px-3 py-2">{h}</th>)}</tr></thead>
              <tbody className="divide-y divide-slate-100">{PROTOCOLS.map((p) => <tr key={p.nb}><td className="px-3 py-2 font-semibold">{p.nb}</td><td className="px-3 py-2">{p.data}</td><td className="px-3 py-2">{p.split}</td><td className="px-3 py-2">{p.val}</td><td className="px-3 py-2">{p.feats}</td><td className="px-3 py-2">{p.imb}</td><td className="px-3 py-2">{p.avg}</td></tr>)}</tbody></table></div>
            {full.data?.warnings?.length > 0 && <div className="mt-3"><Notice tone="warn">{full.data.warnings.length} run warning(s) were captured for this experiment (e.g. a class with fewer members than the number of CV folds, solver convergence). See History & method for details.</Notice></div>}
          </Card>
        </div>
      )}
    </>
  );
}
