import { useEffect, useMemo, useRef, useState } from "react";
import { Download, FileUp, Loader2 } from "lucide-react";
import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { BatchResponse, FS_LABEL, ModelInfo } from "../types";
import { Badge, Card, CHART_COLORS, Disclaimer, ErrorBox, Loading, Notice, PageHeader, Stat } from "../components/ui";
import { ModelPicker } from "../components/ModelPicker";
import { Col, DataTable } from "../components/DataTable";
import { downloadText, toCsv } from "../utils/format";

interface Validation { n_rows: number; n_valid_rows: number; n_invalid_rows: number; issues: { row: number; column: string; issue: string }[]; issues_truncated: boolean; columns: string[]; preview: Record<string, any>[] }

export default function Batch() {
  const models = useApi<ModelInfo[]>("/models");
  const [file, setFile] = useState<File | null>(null);
  const [val, setVal] = useState<Validation | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [stage, setStage] = useState<"idle" | "validating" | "predicting">("idle");
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<BatchResponse | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (models.data && selected.length === 0) {
      const m = models.data.find((x) => x.implementation === "C") ?? models.data[0];
      if (m) setSelected([m.id]);
    }
  }, [models.data]); // eslint-disable-line

  async function pick(f: File | undefined) {
    if (!f) return;
    setFile(f); setVal(null); setRes(null); setErr(null);
    if (!/\.csv$/i.test(f.name)) { setErr("Please upload a .csv file."); return; }
    setStage("validating");
    try { setVal(await api.upload<Validation>("/batch/validate", f)); } catch (e: any) { setErr(e.message); }
    finally { setStage("idle"); }
  }
  async function run() {
    if (!file || selected.length === 0) return;
    setStage("predicting"); setErr(null);
    try { setRes(await api.upload<BatchResponse>("/batch/predict", file, { model_ids: selected.join(",") })); }
    catch (e: any) { setErr(e.message); } finally { setStage("idle"); }
  }

  const gasCols = ["H2", "CH4", "C2H6", "C2H4", "C2H2", "CO"];
  const tableCols: Col<any>[] = useMemo(() => !res ? [] : [
    { key: "row", header: "Row", sortValue: (r) => r.row },
    ...gasCols.filter((g) => res.rows.some((r) => g in r)).map((g) => ({ key: g, header: g, align: "right" as const, render: (r: any) => (r[g] == null ? <span className="text-red-600">missing</span> : String(r[g])) })),
    ...Object.entries(res.models).map(([id, m]) => ({ key: id, header: `${m.model_name} · ${FS_LABEL[m.feature_set] ?? m.feature_set}`,
      sortValue: (r: any) => r.predictions[id] ?? "", render: (r: any) => (r.predictions[id] ? <Badge tone="teal">{r.predictions[id]}</Badge> : <span className="text-slate-400">–</span>) })),
    { key: "status", header: "Status", render: (r: any) => <Badge tone={r.status === "ok" ? "green" : r.status === "invalid_input" ? "red" : "amber"}>{r.status.replace("_", " ")}</Badge> },
  ], [res]); // eslint-disable-line

  const chartData = useMemo(() => {
    if (!res) return { rows: [] as any[], classes: [] as string[] };
    const classes = Array.from(new Set(Object.values(res.models).flatMap((m) => Object.keys(m.class_counts)))).sort();
    return { classes, rows: Object.values(res.models).map((m) => ({ model: `${m.model_name} (${FS_LABEL[m.feature_set]})`, ...m.class_counts })) };
  }, [res]);

  function downloadResults() {
    if (!res) return;
    const rows = res.rows.map((r) => ({ row: r.row, ...Object.fromEntries(gasCols.filter((g) => g in r).map((g) => [g, r[g] ?? ""])),
      ...Object.fromEntries(Object.entries(res.models).map(([id, m]) => [`${m.model_name} [${m.feature_set}]`, r.predictions[id] ?? ""])),
      status: r.status, notes: r.notes.join(" | ") }));
    downloadText("dga_batch_results.csv", toCsv(rows));
  }

  if (models.error) return <ErrorBox message={models.error} onRetry={models.reload} />;
  if (models.loading || !models.data) return <Loading />;

  return (
    <>
      <PageHeader title="CSV batch analysis" description="Upload many readings at once. Models are loaded once and never retrained. Rows with missing, non-numeric or negative values are reported and skipped – never filled in."
        actions={<a className="btn-secondary" href={api.url("/batch/template")} download><Download className="h-4 w-4" />CSV template</a>} />
      <div className="grid gap-5 xl:grid-cols-3">
        <div className="space-y-5 xl:col-span-2">
          <Card title="1 · Upload">
            <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
              onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
              className={`flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border-2 border-dashed px-6 py-9 text-center transition-colors ${drag ? "border-brand-500 bg-brand-50" : "border-slate-300 hover:bg-slate-50"}`}>
              <FileUp className="h-6 w-6 text-brand-600" />
              <div className="text-sm font-medium text-slate-800">{file ? file.name : "Drag a CSV here, or click to browse"}</div>
              <div className="text-xs text-slate-500">Required columns: H2, CH4, C2H6, C2H4, C2H2 · optional: CO · max 5 MB</div>
              <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => pick(e.target.files?.[0])} />
            </div>
            {stage === "validating" && <div className="mt-3"><Loading label="Validating file…" /></div>}
            {err && <div className="mt-3"><ErrorBox message={err} /></div>}
            {val && (
              <div className="mt-4 space-y-3">
                <div className="grid grid-cols-3 gap-3"><Stat label="Rows" value={val.n_rows} /><Stat label="Valid" value={val.n_valid_rows} accent /><Stat label="Invalid" value={val.n_invalid_rows} /></div>
                {val.issues.length > 0 && (
                  <div><div className="label mb-1">Invalid-row report{val.issues_truncated ? " (first 500)" : ""}</div>
                    <div className="max-h-40 overflow-auto rounded-md border border-red-200 bg-red-50/50"><table className="w-full text-xs"><tbody className="divide-y divide-red-100">
                      {val.issues.map((i, k) => <tr key={k}><td className="px-3 py-1">Row {i.row}</td><td className="px-3 py-1">{i.column}</td><td className="px-3 py-1 text-red-700">{i.issue}</td></tr>)}</tbody></table></div></div>
                )}
                <div><div className="label mb-1">Preview (first 10 rows)</div>
                  <div className="overflow-x-auto rounded-md border border-slate-200"><table className="num w-full text-xs"><thead className="bg-slate-50"><tr>{val.columns.map((c) => <th key={c} className="px-3 py-1.5 text-left font-medium">{c}</th>)}</tr></thead>
                    <tbody className="divide-y divide-slate-100">{val.preview.map((r, i) => <tr key={i}>{val.columns.map((c) => <td key={c} className="px-3 py-1.5">{r[c] ?? <span className="text-red-600">missing</span>}</td>)}</tr>)}</tbody></table></div></div>
              </div>
            )}
          </Card>
        </div>
        <Card title="2 · Models" subtitle="Choose one or several (max 10)">
          <ModelPicker models={models.data} selected={selected} onChange={setSelected} />
          <button className="btn-primary mt-4 w-full" disabled={!val || val.n_valid_rows === 0 || selected.length === 0 || stage !== "idle"} onClick={run}>
            {stage === "predicting" ? <><Loader2 className="h-4 w-4 animate-spin" />Predicting {val?.n_valid_rows} rows…</> : "Run batch prediction"}
          </button>
          {!val && <p className="mt-2 text-xs text-slate-500">Upload a valid CSV to enable prediction.</p>}
        </Card>
      </div>

      {res && (
        <div className="mt-5 space-y-5">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Records" value={res.n_rows} /><Stat label="Predicted" value={res.n_valid_rows} accent /><Stat label="Skipped (invalid)" value={res.n_invalid_rows} />
            <Stat label="Models run" value={Object.keys(res.models).length} />
          </div>
          <Card title="Predicted class counts" subtitle="Counts of model outputs per model. Different models can produce different counts for the same file.">
            <div className="h-72"><ResponsiveContainer><BarChart data={chartData.rows}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" fontSize={11} interval={0} /><YAxis fontSize={11} allowDecimals={false} /><Tooltip /><Legend />
              {chartData.classes.map((c, i) => <Bar key={c} dataKey={c} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}</BarChart></ResponsiveContainer></div>
            {Object.values(res.models).some((m) => m.n_not_computable > 0) && <div className="mt-3"><Notice tone="warn">Some rows could not be scored by feature sets without epsilon handling (zero denominator). They are listed with a note and are not imputed.</Notice></div>}
          </Card>
          <Card title="Results" action={<button className="btn-secondary !py-1 text-xs" onClick={downloadResults}><Download className="h-3.5 w-3.5" />Download CSV</button>}>
            <DataTable rows={res.rows} columns={tableCols} pageSize={10} searchText={(r) => `${r.row} ${Object.values(r.predictions).join(" ")} ${r.status}`}
              filters={[{ label: "Status", options: [{ value: "ok", label: "ok" }, { value: "invalid_input", label: "invalid input" }, { value: "not_computable", label: "not computable" }], test: (r, v) => r.status === v },
                { label: "Class", options: chartData.classes.map((c) => ({ value: c, label: c })), test: (r, v) => Object.values(r.predictions).includes(v) }]} />
          </Card>
          <Disclaimer />
        </div>
      )}
    </>
  );
}
