import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Download } from "lucide-react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Badge, Card, CHART_COLORS, ErrorBox, Loading, Notice, PageHeader, Stat } from "../components/ui";

interface DS {
  raw_rows: number; columns: string[]; sha256: string; missing_values: Record<string, number>; duplicate_rows: number;
  conflicting_label_rows_6gas_key: number; conflicting_label_rows_5gas_key: number; zero_values: Record<string, number>; min_values: Record<string, number>;
  class_counts_raw: Record<string, number>; protocols: Record<string, { rows: number; class_counts: Record<string, number> }>;
  gas_histograms_log10: Record<string, { bin: string; count: number }[]>; gas_stats: Record<string, Record<string, number>>;
  provenance: { file: string; source_url: string; note: string; cleaning_actions: string[] };
}

export default function DataQuality() {
  const { data: d, error, loading, reload } = useApi<DS>("/datasets/summary");
  const [gas, setGas] = useState("H2");
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  if (loading || !d) return <Loading />;
  const classes = Object.keys(d.class_counts_raw);
  const dist = classes.map((c) => ({ cls: c, Raw: d.class_counts_raw[c], ...Object.fromEntries(Object.entries(d.protocols).map(([p, v]) => [p, v.class_counts[c] ?? 0])) }));
  const protoNames = Object.keys(d.protocols);
  const totalMissing = Object.values(d.missing_values).reduce((a, b) => a + b, 0);
  return (
    <>
      <PageHeader title="Dataset and data quality" description="Quality checks on the raw DuVal.csv used by all three notebooks. The raw file is never modified – each notebook's cleaning is applied in memory."
        actions={<a className="btn-secondary" href={api.url("/datasets/quality-report")} download="data_quality_report.json"><Download className="h-4 w-4" />Quality report (JSON)</a>} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Records" value={d.raw_rows} hint={`${d.columns.length} columns`} /><Stat label="Missing values" value={totalMissing} /><Stat label="Duplicate rows" value={d.duplicate_rows} accent />
        <Stat label="Conflicting-label rows" value={d.conflicting_label_rows_6gas_key} hint={`${d.conflicting_label_rows_5gas_key} if CO is ignored`} accent /><Stat label="Zero gas values" value={Object.values(d.zero_values).reduce((a, b) => a + b, 0)} />
      </div>
      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Card title="Class distribution" subtitle="Records per class: raw vs. each notebook's cleaning protocol">
          <div className="h-72"><ResponsiveContainer><BarChart data={dist}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="cls" fontSize={11} /><YAxis fontSize={11} /><Tooltip /><Legend />
            <Bar dataKey="Raw" fill="#9fb3c8" />{protoNames.map((p, i) => <Bar key={p} dataKey={p} fill={CHART_COLORS[i]} />)}</BarChart></ResponsiveContainer></div>
          <p className="mt-2 text-xs text-slate-600">{protoNames.map((p) => `${p}: ${d.protocols[p].rows} rows`).join(" · ")}</p>
          <div className="mt-2"><Notice tone="warn">Arcing has only {d.protocols[protoNames[0]].class_counts["Arcing"]} records after de-duplication. A stratified 80/20 split leaves about one in the test set, so Arcing metrics are not statistically reliable.</Notice></div>
        </Card>
        <Card title="Gas concentration distribution" subtitle="Histogram of log₁₀(concentration); bin labels show the original range" action={<select className="input !w-auto !py-1 text-xs" value={gas} onChange={(e) => setGas(e.target.value)} aria-label="Gas">{Object.keys(d.gas_histograms_log10).map((g) => <option key={g}>{g}</option>)}</select>}>
          <div className="h-72"><ResponsiveContainer><BarChart data={d.gas_histograms_log10[gas]}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="bin" fontSize={9} interval={0} angle={-30} textAnchor="end" height={60} /><YAxis fontSize={11} /><Tooltip /><Bar dataKey="count" fill="#0e8a8c" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div>
          <p className="mt-1 text-[11px] text-slate-500">min {d.gas_stats[gas].min} · median {d.gas_stats[gas]["50%"]} · max {d.gas_stats[gas].max}</p>
        </Card>
        <Card title="Cleaning actions">
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-700">{d.provenance.cleaning_actions.map((a, i) => <li key={i}>{a}</li>)}</ol>
          <div className="mt-3 overflow-x-auto rounded-md border border-slate-200"><table className="w-full text-sm"><thead className="bg-slate-50 text-xs text-slate-600"><tr><th className="px-3 py-2 text-left">Protocol</th><th className="px-3 py-2 text-left">Used by</th><th className="px-3 py-2 text-right">Rows</th></tr></thead><tbody className="divide-y divide-slate-100">
            <tr><td className="px-3 py-1.5">Raw file</td><td className="px-3 py-1.5">–</td><td className="num px-3 py-1.5 text-right">{d.raw_rows}</td></tr>
            {protoNames.map((p) => <tr key={p}><td className="px-3 py-1.5">{p}</td><td className="px-3 py-1.5">{p.includes("conflict") ? "Notebook A" : "Notebooks B, C"}</td><td className="num px-3 py-1.5 text-right">{d.protocols[p].rows}</td></tr>)}</tbody></table></div>
          <p className="mt-2 text-xs text-slate-500">Notebook A's conflict check includes CO in the key (10 rows). Checking the five model gases only would flag {d.conflicting_label_rows_5gas_key} rows; identical gas inputs with different labels can appear in both train and test sets in notebooks B and C.</p>
        </Card>
        <Card title="Provenance">
          <dl className="space-y-2 text-sm"><div><dt className="label">File</dt><dd>{d.provenance.file}</dd></div><div><dt className="label">Source</dt><dd className="break-all text-xs">{d.provenance.source_url}</dd></div>
            <div><dt className="label">SHA-256</dt><dd className="break-all font-mono text-[11px]">{d.sha256}</dd></div><div><dt className="label">Storage</dt><dd className="text-xs text-slate-600">{d.provenance.note}</dd></div>
            <div><dt className="label">Not documented in the source file</dt><dd className="text-xs text-slate-600">Gas units, sampling dates, transformer identifiers and how labels were assigned. <Badge tone="amber">unknown</Badge></dd></div></dl>
        </Card>
        <Card className="xl:col-span-2" title="Per-column checks">
          <div className="overflow-x-auto rounded-md border border-slate-200"><table className="num w-full text-sm"><thead className="bg-slate-50 text-xs text-slate-600"><tr><th className="px-3 py-2 text-left">Gas</th><th className="px-3 py-2 text-right">Missing</th><th className="px-3 py-2 text-right">Zeros</th><th className="px-3 py-2 text-right">Min</th><th className="px-3 py-2 text-right">Median</th><th className="px-3 py-2 text-right">Mean</th><th className="px-3 py-2 text-right">Max</th></tr></thead>
            <tbody className="divide-y divide-slate-100">{Object.keys(d.gas_stats).map((g) => <tr key={g}><td className="px-3 py-1.5 font-medium">{g}</td><td className="px-3 py-1.5 text-right">{d.missing_values[g]}</td><td className="px-3 py-1.5 text-right">{d.zero_values[g]}</td><td className="px-3 py-1.5 text-right">{d.gas_stats[g].min}</td><td className="px-3 py-1.5 text-right">{d.gas_stats[g]["50%"]}</td><td className="px-3 py-1.5 text-right">{d.gas_stats[g].mean.toFixed(1)}</td><td className="px-3 py-1.5 text-right">{d.gas_stats[g].max}</td></tr>)}</tbody></table></div>
        </Card>
      </div>
    </>
  );
}
