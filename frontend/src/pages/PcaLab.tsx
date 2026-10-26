import { useMemo, useState } from "react";
import { Bar, BarChart, Cell, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../hooks/useApi";
import { Confusion } from "../types";
import { Badge, Card, CHART_COLORS, ErrorBox, Loading, Notice, NotImplemented, PageHeader, Stat } from "../components/ui";
import { ConfusionMatrix } from "../components/ConfusionMatrix";

interface Pca {
  pca: { features: string[]; explained_variance_pct: number[]; cumulative_variance_pct: number[]; k_for_95: number; k_for_99: number;
    loadings: Record<string, Record<string, number>>; correlation: { features: string[]; matrix: number[][]; note: string } };
  sweep: { k: number; cum_var_pct: number; train_acc_pct: number; test_acc_pct: number; cv5_acc_pct: number }[];
  k_best_test_paper_rule: number; k_best_cv: number;
  final: { k: number; metrics: any; confusion: Confusion; n_train: number; n_test: number };
  ann_config: Record<string, unknown>; split: Record<string, unknown>;
  comparison_10splits: { model: string; seed: number; train_acc_pct: number; test_acc_pct: number }[];
  comparison_corrected: { seed: number; k_selected: number; train_acc_pct: number; test_acc_pct: number }[] | null;
  provenance: { created_utc: string; n_rows: number; cleaning_protocol: string }; note: string;
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const sd = (v: number[]) => { const m = mean(v); return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)); };

export default function PcaLab() {
  const { data: d, error, loading, reload } = useApi<Pca>("/pca/summary");
  const [k, setK] = useState<number | null>(null);
  const [pc, setPc] = useState(1);

  const comp = useMemo(() => {
    if (!d) return [];
    const by: Record<string, number[]> = {};
    d.comparison_10splits.forEach((r) => (by[r.model] ||= []).push(r.test_acc_pct));
    const rows = Object.entries(by).map(([model, v]) => ({ model, mean: +mean(v).toFixed(1), sd: +sd(v).toFixed(1) }));
    if (d.comparison_corrected) { const v = d.comparison_corrected.map((r) => r.test_acc_pct); rows.push({ model: "PCA + ANN (k re-selected per split)", mean: +mean(v).toFixed(1), sd: +sd(v).toFixed(1) }); }
    return rows.sort((a, b) => b.mean - a.mean);
  }, [d]);

  if (error) return <ErrorBox message={error} onRetry={reload} />;
  if (loading || !d) return <Loading />;
  const nFeat = d.pca.features.length;
  const kSel = k ?? d.k_best_cv;
  const row = d.sweep.find((r) => r.k === kSel)!;
  const scree = d.pca.explained_variance_pct.map((v, i) => ({ pc: `PC${i + 1}`, individual: +v.toFixed(2), cumulative: Math.min(100, +d.pca.cumulative_variance_pct[i].toFixed(2)) }));
  const load = d.pca.features.map((f) => ({ f, v: +d.pca.loadings[`PC${pc}`][f].toFixed(3) }));
  const sweep = d.sweep.map((r) => ({ k: r.k, Train: +r.train_acc_pct.toFixed(1), "5-fold CV (train)": +r.cv5_acc_pct.toFixed(1), Test: +r.test_acc_pct.toFixed(1) }));

  return (
    <>
      <PageHeader title="PCA + ANN laboratory" description="Dimensionality reduction on 15 engineered gas features followed by a small neural network. Every chart is read from the saved training run; nothing here is simulated." />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Original features" value={nFeat} /><Stat label="Retained components" value={d.final.k} accent hint="chosen by 5-fold CV on training data" />
        <Stat label="k for 95% / 99% variance" value={`${d.pca.k_for_95} / ${d.pca.k_for_99}`} /><Stat label="Final test accuracy" value={`${(d.final.metrics.accuracy * 100).toFixed(1)}%`} hint={`n=${d.final.n_test} test records`} />
        <Stat label="k by best test accuracy" value={d.k_best_test_paper_rule} hint="paper's rule – optimistic" />
      </div>
      <div className="mt-4"><Notice tone="warn">Choosing k by <b>test</b> accuracy (the paper's rule, k={d.k_best_test_paper_rule}) uses the test set for model selection and is optimistic. The final model uses k={d.k_best_cv}, chosen by 5-fold cross-validation on the training data only.</Notice></div>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Card title="Explained variance" subtitle="Fitted on standardised training data only"><div className="h-72"><ResponsiveContainer><ComposedChart data={scree}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="pc" fontSize={11} /><YAxis unit="%" fontSize={11} domain={[0, 100]} /><Tooltip formatter={(v: number) => `${v}%`} /><Legend />
          <ReferenceLine y={95} stroke="#8a96a3" strokeDasharray="4 4" label={{ value: "95%", fontSize: 10, position: "right" }} /><ReferenceLine y={99} stroke="#c4860b" strokeDasharray="2 4" label={{ value: "99%", fontSize: 10, position: "right" }} />
          <Bar dataKey="individual" name="Individual" fill="#7aa7c7" /><Line dataKey="cumulative" name="Cumulative" stroke="#0e8a8c" strokeWidth={2} dot={{ r: 3 }} /></ComposedChart></ResponsiveContainer></div></Card>

        <Card title="Accuracy vs. number of components" subtitle="Training, cross-validation and test curves are different quantities – do not read them as one metric">
          <div className="h-72"><ResponsiveContainer><ComposedChart data={sweep}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="k" fontSize={11} label={{ value: "components k", position: "insideBottom", offset: -2, fontSize: 11 }} /><YAxis unit="%" domain={[70, 100]} ticks={[70, 80, 90, 100]} fontSize={11} /><Tooltip /><Legend verticalAlign="top" />
            <ReferenceLine x={kSel} stroke="#e8a317" strokeWidth={2} /><Line dataKey="Train" stroke="#2a4d6e" strokeDasharray="5 3" dot={false} /><Line dataKey="5-fold CV (train)" stroke="#0e8a8c" strokeWidth={2} dot={{ r: 2 }} /><Line dataKey="Test" stroke="#c4860b" dot={{ r: 2 }} /></ComposedChart></ResponsiveContainer></div>
        </Card>

        <Card title="Component-count selector" subtitle="Reads the precomputed k = 1…15 sweep">
          <div className="flex items-center gap-4"><input type="range" min={1} max={nFeat} value={kSel} onChange={(e) => setK(+e.target.value)} className="flex-1 accent-brand-600" aria-label="Number of components" /><span className="num w-16 text-right text-lg font-semibold">k = {kSel}</span></div>
          <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4"><Stat label="Cum. variance" value={`${row.cum_var_pct.toFixed(1)}%`} /><Stat label="Train acc." value={`${row.train_acc_pct.toFixed(1)}%`} hint="resubstitution" /><Stat label="5-fold CV acc." value={`${row.cv5_acc_pct.toFixed(1)}%`} hint="training data" /><Stat label="Test acc." value={`${row.test_acc_pct.toFixed(1)}%`} hint="held-out" /></div>
          <div className="mt-3"><NotImplemented>Interactive re-training for an arbitrary k is not available; the values above come from the saved run. {d.note}</NotImplemented></div>
        </Card>

        <Card title="PCA loadings" subtitle="Weights of the 15 standardised features in each component" action={<select className="input !w-auto !py-1 text-xs" value={pc} onChange={(e) => setPc(+e.target.value)} aria-label="Principal component">{Array.from({ length: nFeat }, (_, i) => <option key={i} value={i + 1}>PC{i + 1} ({d.pca.explained_variance_pct[i].toFixed(1)}%)</option>)}</select>}>
          <div className="h-72"><ResponsiveContainer><BarChart data={load} layout="vertical" margin={{ left: 30 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" domain={[-1, 1]} fontSize={11} /><YAxis dataKey="f" type="category" width={80} fontSize={10} interval={0} /><Tooltip /><ReferenceLine x={0} stroke="#94a3b8" /><Bar dataKey="v" fill="#0e8a8c" radius={2} /></BarChart></ResponsiveContainer></div>
        </Card>

        <Card className="xl:col-span-2" title="Feature correlation (Pearson)" subtitle={d.pca.correlation.note}><Heatmap f={d.pca.correlation.features} m={d.pca.correlation.matrix} /></Card>

        <Card title={`Confusion matrix · PCA + ANN (k=${d.final.k})`} subtitle="Held-out test set"><ConfusionMatrix cm={d.final.confusion} /></Card>
        <Card title="ANN configuration">
          <table className="w-full text-sm"><tbody className="divide-y divide-slate-100">{Object.entries(d.ann_config).map(([k, v]) => <tr key={k}><td className="py-1.5 text-slate-500">{k}</td><td className="py-1.5 text-right font-mono text-xs">{JSON.stringify(v)}</td></tr>)}
            <tr><td className="py-1.5 text-slate-500">preprocessing</td><td className="py-1.5 text-right font-mono text-xs">StandardScaler → PCA(k={d.final.k}) → MLP</td></tr>
            <tr><td className="py-1.5 text-slate-500">dataset</td><td className="py-1.5 text-right text-xs">{d.provenance.n_rows} records · {d.provenance.cleaning_protocol}</td></tr></tbody></table>
        </Card>

        <Card className="xl:col-span-2" title="PCA + ANN vs. baselines" subtitle="Test accuracy, mean ± std over 10 random stratified 80/20 splits (the notebook's protocol)">
          <div className="h-72"><ResponsiveContainer><BarChart data={comp} layout="vertical" margin={{ left: 120 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" domain={[50, 100]} unit="%" fontSize={11} /><YAxis dataKey="model" type="category" width={230} fontSize={11} /><Tooltip formatter={(v: number, _n, p: any) => [`${v}% ± ${p.payload.sd}`, "mean"]} />
            <Bar dataKey="mean" radius={[0, 3, 3, 0]}>{comp.map((r, i) => <Cell key={i} fill={r.model.startsWith("PCA") ? "#e8a317" : "#0e8a8c"} />)}</Bar></BarChart></ResponsiveContainer></div>
          <div className="mt-3 overflow-x-auto"><table className="num w-full text-sm"><thead className="text-xs text-slate-500"><tr><th className="text-left">Model</th><th className="text-right">Mean test acc.</th><th className="text-right">Std</th></tr></thead><tbody className="divide-y divide-slate-100">{comp.map((r) => <tr key={r.model}><td className="py-1.5">{r.model}</td><td className="text-right">{r.mean}%</td><td className="text-right">{r.sd}</td></tr>)}</tbody></table></div>
          <div className="mt-3 space-y-2"><Notice>In the notebook's 10-split comparison, k={d.final.k} was selected on the seed-42 training split but then evaluated on the other splits whose test records overlap that split's training data – a mild leakage that favours PCA + ANN. The "k re-selected per split" row repeats PCA + ANN with k chosen by 5-fold CV inside every split's own training data.</Notice>
            <Notice tone="warn">Differences of a few points are comparable to the standard deviations above; the ranking should be read as indicative, not definitive.</Notice></div>
        </Card>
      </div>
      <div className="mt-3 flex gap-2 text-xs text-slate-500"><Badge tone="teal">newly run</Badge>Computed {d.provenance.created_utc?.slice(0, 10)} by scripts/train_all.py.</div>
    </>
  );
}

function Heatmap({ f, m }: { f: string[]; m: number[][] }) {
  return (
    <div className="overflow-x-auto"><table className="num mx-auto border-separate border-spacing-px text-[10px]" aria-label="Correlation heatmap">
      <thead><tr><th />{f.map((c) => <th key={c} className="h-16 w-9 align-bottom font-normal text-slate-600"><div className="origin-bottom-left translate-x-3 whitespace-nowrap" style={{ transform: "translateX(12px) rotate(-60deg)" }}>{c}</div></th>)}</tr></thead>
      <tbody>{m.map((row, i) => <tr key={i}><th className="pr-1 text-right font-normal text-slate-600">{f[i]}</th>{row.map((v, j) => {
        const a = Math.min(1, Math.abs(v)); const bg = v >= 0 ? `rgba(14,138,140,${a})` : `rgba(232,163,23,${a})`;
        return <td key={j} title={`${f[i]} vs ${f[j]}: ${v.toFixed(2)}`} className="h-8 w-9 text-center" style={{ background: bg, color: a > 0.55 ? "#fff" : "#0f172a" }}>{v.toFixed(1)}</td>; })}</tr>)}</tbody></table>
      <p className="mt-2 text-center text-[11px] text-slate-500">Teal = positive, amber = negative correlation.</p></div>
  );
}
