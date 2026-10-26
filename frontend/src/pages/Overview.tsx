import { Link } from "react-router-dom";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { GitCompare, Zap } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { ExperimentRow, FS_LABEL, MetricRow } from "../types";
import { Badge, Card, CHART_COLORS, Empty, ErrorBox, Loading, PageHeader, Stat } from "../components/ui";

interface Overview { n_experiments: number; n_models: number; n_implementations: number; latest_experiment: ExperimentRow | null; n_predictions_logged: number;
  recent_predictions: { id: number; ts: string; mode: string; model_ids: string[]; n_records: number; n_valid: number; summary: any }[] }
interface DS { raw_rows: number; protocols: Record<string, { rows: number; class_counts: Record<string, number> }>; class_counts_raw: Record<string, number> }

const EXPS = [
  { id: "dga_main", label: "Notebook B · DGA cases" },
  { id: "rfrep_main", label: "Notebook A · RF replication" },
  { id: "pcaann_main", label: "Notebook C · PCA + ANN" },
];

export default function OverviewPage() {
  const ov = useApi<Overview>("/overview");
  const ds = useApi<DS>("/datasets/summary");
  const [exp, setExp] = useState("dga_main");
  const metrics = useApi<MetricRow[]>(`/experiments/${exp}/metrics`);

  if (ov.error) return <ErrorBox message={ov.error} onRetry={ov.reload} />;
  if (ov.loading || !ov.data) return <Loading />;
  const o = ov.data;
  const dist = ds.data ? (() => {
    const keys = Object.keys(ds.data.class_counts_raw);
    const protos = Object.entries(ds.data.protocols);
    return keys.map((k) => ({ cls: k, Raw: ds.data!.class_counts_raw[k], ...Object.fromEntries(protos.map(([p, v]) => [p.split(" (")[0], v.class_counts[k] ?? 0])) }));
  })() : [];
  const protoKeys = ds.data ? Object.keys(ds.data.protocols).map((p) => p.split(" (")[0]) : [];

  const test = (metrics.data ?? []).filter((m) => m.stage === "test" || m.stage === "mean_10_random_splits");
  const byModel = Array.from(new Set(test.map((m) => m.model)));
  const fsets = Array.from(new Set(test.map((m) => m.feature_set)));
  const chart = byModel.map((m) => ({ model: m, ...Object.fromEntries(fsets.map((f) => [f, (test.find((t) => t.model === m && t.feature_set === f)?.accuracy ?? null) !== null ? +((test.find((t) => t.model === m && t.feature_set === f)!.accuracy! * 100).toFixed(1)) : null])) }));
  const latest = o.latest_experiment;

  return (
    <>
      <PageHeader title="Power Transformer Fault Diagnosis Using Dissolved Gas Analysis and Machine Learning"
        description="A research platform unifying three published-paper implementations – Random Forest comparative ML, DGA feature cases with six classifiers, and PCA + ANN – behind one inference API and dashboard. All figures below are computed from the trained experiments."
        actions={<><Link to="/diagnosis" className="btn-primary"><Zap className="h-4 w-4" />Run diagnosis</Link><Link to="/compare" className="btn-secondary"><GitCompare className="h-4 w-4" />Compare research models</Link></>} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <Stat label="Raw records" value={ds.data?.raw_rows ?? "–"} hint="DuVal.csv" />
        <Stat label="Fault classes" value={ds.data ? Object.keys(ds.data.class_counts_raw).length : "–"} hint="Normal, PD, OH, Arcing" />
        <Stat label="Research approaches" value={o.n_implementations} accent />
        <Stat label="Trained models" value={o.n_models} accent />
        <Stat label="Experiments" value={o.n_experiments} />
        <Stat label="Predictions logged" value={o.n_predictions_logged} hint="this installation" />
      </div>

      {o.n_experiments === 0 && <div className="mt-5"><Empty title="No experiments yet">Run <code>python scripts/train_all.py</code> in the backend to train the models and populate this dashboard.</Empty></div>}

      <div className="mt-5 grid gap-5 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Test accuracy by model" subtitle="Held-out results of one experiment at a time – protocols differ between notebooks, so bars from different tabs are not a single ranking."
          action={<div className="flex gap-1">{EXPS.map((e) => <button key={e.id} onClick={() => setExp(e.id)} className={`rounded px-2.5 py-1 text-xs ${exp === e.id ? "bg-ink-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`}>{e.label}</button>)}</div>}>
          {metrics.loading ? <Loading /> : metrics.error ? <ErrorBox message={metrics.error} /> : (
            <div className="h-80"><ResponsiveContainer><BarChart data={chart} margin={{ bottom: 20 }}><CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="model" fontSize={11} interval={0} angle={-15} textAnchor="end" height={55} /><YAxis unit="%" domain={[0, 100]} fontSize={11} /><Tooltip formatter={(v: number) => `${v}%`} />
              {fsets.length > 1 && <Legend formatter={(v) => FS_LABEL[v] ?? v} />}
              {fsets.map((f, i) => <Bar key={f} dataKey={f} name={FS_LABEL[f] ?? f} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[3, 3, 0, 0]} />)}</BarChart></ResponsiveContainer></div>
          )}
          {exp === "pcaann_main" && <p className="mt-2 text-[11px] text-slate-500">Notebook C bars are means over 10 random 80/20 splits (accuracy only), plus the final PCA+ANN on the seed-42 split.</p>}
        </Card>
        <Card title="Latest experiment">
          {latest ? (
            <dl className="space-y-2 text-sm">
              <div><dt className="label">Title</dt><dd className="font-medium">{latest.title}</dd></div>
              <div className="flex gap-6"><div><dt className="label">Status</dt><dd><Badge tone="green">{latest.status}</Badge></dd></div><div><dt className="label">Type</dt><dd>{latest.kind}</dd></div></div>
              <div><dt className="label">Run (UTC)</dt><dd className="num">{latest.created_utc?.replace("T", " ")}</dd></div>
              <div><dt className="label">Rows used · seed</dt><dd className="num">{latest.n_rows} · {latest.seed}</dd></div>
              <Link to="/history" className="text-xs text-brand-600 hover:underline">All experiments →</Link>
            </dl>
          ) : <Empty title="None yet" />}
        </Card>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-3">
        <Card className="xl:col-span-2" title="Fault-class distribution" subtitle="Records per class under each cleaning protocol used by the notebooks">
          {ds.loading ? <Loading /> : ds.error ? <ErrorBox message={ds.error} /> : (
            <div className="h-64"><ResponsiveContainer><BarChart data={dist}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="cls" fontSize={11} /><YAxis fontSize={11} /><Tooltip /><Legend />
              <Bar dataKey="Raw" fill="#9fb3c8" />{protoKeys.map((p, i) => <Bar key={p} dataKey={p} fill={CHART_COLORS[i]} />)}</BarChart></ResponsiveContainer></div>
          )}
          <p className="mt-2 text-[11px] text-slate-500">The Arcing class is very small after cleaning; per-class metrics for it are statistically unreliable.</p>
        </Card>
        <Card title="Recent prediction activity">
          {o.recent_predictions.length === 0 ? <Empty title="No predictions yet">Run a diagnosis or upload a CSV – activity is logged locally.</Empty> : (
            <ul className="divide-y divide-slate-100 text-sm">
              {o.recent_predictions.map((p) => (
                <li key={p.id} className="flex items-center justify-between py-2"><div><Badge tone={p.mode === "batch" ? "teal" : "slate"}>{p.mode}</Badge> <span className="text-xs text-slate-600">{p.n_valid}/{p.n_records} records · {p.model_ids.length} model{p.model_ids.length > 1 ? "s" : ""}</span></div><span className="num text-[11px] text-slate-400">{p.ts.slice(5, 16).replace("T", " ")}</span></li>))}
            </ul>
          )}
        </Card>
      </div>
      <p className="mt-4 text-center text-[11px] text-slate-500">Test sets contain roughly 90 records, so one prediction changes accuracy by about one percentage point; small differences are within noise.</p>
    </>
  );
}
