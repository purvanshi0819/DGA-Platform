import { useEffect, useMemo, useState } from "react";
import { Download, FlaskConical, RotateCcw, Zap } from "lucide-react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { CO_FIELD, DiagnosisResponse, FAULT_NAMES, FeatureSetInfo, FS_LABEL, GASES, ModelInfo } from "../types";
import { Badge, Card, Disclaimer, ErrorBox, Loading, Notice, PageHeader } from "../components/ui";
import { ModelPicker } from "../components/ModelPicker";
import { checkGas, downloadText, num, pct, toCsv } from "../utils/format";

// Demonstration values only – NOT a real measurement or a dataset record.
const SAMPLE: Record<string, string> = { H2: "127", CH4: "24", C2H6: "5", C2H4: "32", C2H2: "81", CO: "60" };
const EMPTY: Record<string, string> = { H2: "", CH4: "", C2H6: "", C2H4: "", C2H2: "", CO: "" };

export default function Diagnosis() {
  const models = useApi<ModelInfo[]>("/models");
  const fsets = useApi<FeatureSetInfo[]>("/feature-sets");
  const [vals, setVals] = useState(EMPTY);
  const [touched, setTouched] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<DiagnosisResponse | null>(null);
  const [isSample, setIsSample] = useState(false);

  useEffect(() => {
    if (models.data && selected.length === 0) {
      const pick = (impl: string, fs?: string) => models.data!.find((m) => m.implementation === impl && (!fs || m.feature_set === fs) && /Random Forest|RF|PCA/.test(m.model_name));
      setSelected([pick("B", "duval3"), pick("A"), pick("C")].filter(Boolean).map((m) => m!.id));
    }
  }, [models.data]); // eslint-disable-line

  const checks = useMemo(() => Object.fromEntries([...GASES.map((g) => g.key), "CO"].map((k) => [k, checkGas(vals[k], k !== "CO")])), [vals]);
  const valid = Object.values(checks).every((c) => !c.error);
  const chosen = (models.data ?? []).filter((m) => selected.includes(m.id));
  const needsPositive = useMemo(() => {
    const ids = new Set(chosen.map((m) => m.feature_set));
    return (fsets.data ?? []).filter((f) => ids.has(f.id) && f.positive_required.length).map((f) => `${FS_LABEL[f.id]}: ${f.positive_required.join(", ")} must be > 0`);
  }, [chosen, fsets.data]);

  async function run() {
    setTouched(true);
    if (!valid || selected.length === 0) return;
    setBusy(true); setErr(null);
    const reading: Record<string, number> = {};
    Object.entries(checks).forEach(([k, c]) => { if (c.value !== undefined) reading[k] = c.value; });
    try { setRes(await api.post<DiagnosisResponse>("/diagnosis", { reading, model_ids: selected })); }
    catch (e: any) { setErr(e.message); setRes(null); }
    finally { setBusy(false); }
  }
  const reset = () => { setVals(EMPTY); setTouched(false); setRes(null); setErr(null); setIsSample(false); };

  function exportResult() {
    if (!res) return;
    const rows = res.results.map((r) => ({ model: r.model_name, notebook: r.implementation, feature_set: r.feature_set, status: r.status,
      predicted_class: r.predicted_class ?? "", ...Object.fromEntries(Object.entries(res.input).map(([k, v]) => [k, v ?? ""])) }));
    downloadText("diagnosis_result.csv", toCsv(rows));
  }

  if (models.error) return <ErrorBox message={models.error} onRetry={models.reload} />;
  if (models.loading || !models.data) return <Loading />;
  if (models.data.length === 0) return <Notice tone="warn">No trained models were found. Run <code>python scripts/train_all.py</code> in the backend first.</Notice>;

  return (
    <>
      <PageHeader title="Transformer diagnosis" description="Enter dissolved-gas concentrations, choose one or more trained models and compare their predictions. Every value is validated; missing readings are never replaced with zero." />
      <div className="grid gap-5 xl:grid-cols-5">
        <div className="space-y-5 xl:col-span-2">
          <Card title="Gas concentrations" subtitle="Units follow the dataset (the source file does not document units; commonly ppm)."
            action={<button className="btn-secondary !py-1 text-xs" onClick={() => { setVals(SAMPLE); setIsSample(true); setTouched(false); }}><FlaskConical className="h-3.5 w-3.5" />Load demo values</button>}>
            {isSample && <div className="mb-3"><Badge tone="amber">Demonstration values – not a real measurement</Badge></div>}
            <div className="grid gap-3 sm:grid-cols-2">
              {[...GASES, CO_FIELD].map((g) => {
                const c = checks[g.key]; const req = g.key !== "CO"; const show = (touched || vals[g.key] !== "") && c.error;
                return (
                  <div key={g.key}>
                    <label htmlFor={`g-${g.key}`} className="flex items-baseline justify-between text-sm font-medium text-slate-800">
                      <span>{g.sub} <span className="font-normal text-slate-500">{g.name}</span></span>
                      <span className="text-[10px] font-normal uppercase text-slate-400">{req ? "required" : "optional"}</span>
                    </label>
                    <input id={`g-${g.key}`} inputMode="decimal" className={`input mt-1 ${show ? "input-error" : ""}`} value={vals[g.key]} aria-invalid={!!show}
                      onChange={(e) => { setVals({ ...vals, [g.key]: e.target.value }); setIsSample(false); }} placeholder="e.g. 120" />
                    <p className={`mt-1 text-[11px] ${show ? "text-red-600" : "text-slate-500"}`}>{show ? c.error : g.desc}</p>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 flex gap-2">
              <button className="btn-primary flex-1" onClick={run} disabled={busy || selected.length === 0}><Zap className="h-4 w-4" />{busy ? "Running…" : "Run diagnosis"}</button>
              <button className="btn-secondary" onClick={reset}><RotateCcw className="h-4 w-4" />Reset</button>
            </div>
            {touched && selected.length === 0 && <p className="mt-2 text-xs text-red-600">Select at least one model.</p>}
          </Card>
          <Card title="Models" subtitle={`${selected.length} selected · inference uses each model's own saved feature pipeline`}>
            <ModelPicker models={models.data} selected={selected} onChange={setSelected} />
            <div className="mt-3 space-y-2">
              <Notice>All models need H₂, CH₄, C₂H₆, C₂H₄ and C₂H₂. CO is accepted but no model uses it.</Notice>
              {needsPositive.length > 0 && <Notice tone="warn">These feature sets use no epsilon in the source notebook, so a zero value is reported as "not computable" rather than altered: {needsPositive.join(" · ")}.</Notice>}
            </div>
          </Card>
        </div>

        <div className="space-y-5 xl:col-span-3">
          {err && <ErrorBox message={err} />}
          {!res && !err && (
            <Card><div className="py-10 text-center text-sm text-slate-500">Enter readings and press <b>Run diagnosis</b>. The prediction panel appears here.</div></Card>
          )}
          {res && <Result res={res} onExport={exportResult} />}
          <Disclaimer />
        </div>
      </div>
    </>
  );
}

function Result({ res, onExport }: { res: DiagnosisResponse; onExport: () => void }) {
  const ok = res.results.filter((r) => r.status === "ok");
  const top = Object.entries(res.agreement.votes).sort((a, b) => b[1] - a[1]);
  const [open, setOpen] = useState<string | null>(null);
  const sel = res.results.find((r) => r.model_id === open && r.status === "ok") ?? null;
  return (
    <>
      <Card title="Prediction summary" action={<button className="btn-secondary !py-1 text-xs" onClick={onExport}><Download className="h-3.5 w-3.5" />Export CSV</button>}>
        {ok.length === 0 ? <ErrorBox message="None of the selected models could compute features for this input. See details below." /> : (
          <div className="flex flex-wrap items-center gap-6">
            <div>
              <div className="label">{res.agreement.unanimous ? "All models agree" : ok.length > 1 ? "Models disagree – most common prediction" : "Predicted class"}</div>
              <div className="mt-1 text-3xl font-semibold text-slate-900">{FAULT_NAMES[top[0][0]] ?? top[0][0]}</div>
              <div className="text-xs text-slate-500">{top[0][1]} of {ok.length} model{ok.length > 1 ? "s" : ""} · model output, not a confirmed diagnosis</div>
            </div>
            {top.length > 1 && <div className="text-xs text-slate-600">{top.map(([k, v]) => <div key={k}><b>{FAULT_NAMES[k] ?? k}</b>: {v}</div>)}</div>}
          </div>
        )}
      </Card>
      <Card title="Per-model results" subtitle="Click a row for features and class scores">
        <div className="overflow-x-auto rounded-md border border-slate-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs text-slate-600"><tr><th className="px-3 py-2">Model</th><th className="px-3 py-2">Features</th><th className="px-3 py-2">Prediction</th><th className="px-3 py-2">Top score</th></tr></thead>
            <tbody className="divide-y divide-slate-100">
              {res.results.map((r) => {
                const top = r.probabilities ? Math.max(...Object.values(r.probabilities)) : null;
                return (
                  <tr key={r.model_id} className={`cursor-pointer hover:bg-slate-50 ${open === r.model_id ? "bg-brand-50" : ""}`} onClick={() => setOpen(open === r.model_id ? null : r.model_id)}>
                    <td className="px-3 py-2"><span className="font-medium">{r.model_name}</span> <span className="text-xs text-slate-500">· notebook {r.implementation}</span></td>
                    <td className="px-3 py-2 text-slate-600">{FS_LABEL[r.feature_set] ?? r.feature_set}</td>
                    <td className="px-3 py-2">{r.status === "ok" ? <Badge tone="teal">{FAULT_NAMES[r.predicted_class!] ?? r.predicted_class}</Badge> : <Badge tone="amber">not computable</Badge>}</td>
                    <td className="num px-3 py-2 text-slate-600">{top != null ? pct(top, 0) : r.status === "ok" ? "n/a" : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {res.results.filter((r) => r.status !== "ok").map((r) => <p key={r.model_id} className="mt-2 text-xs text-amber-800"><b>{r.model_name} ({FS_LABEL[r.feature_set]}):</b> {r.message}</p>)}
        <p className="mt-3 text-[11px] text-slate-500">Scores are classifier outputs, shown only for models that provide them. They are not calibrated probabilities and are not a real-world risk estimate. Models without a score show "n/a".</p>
      </Card>
      {sel && (
        <Card title={`${sel.model_name} · ${FS_LABEL[sel.feature_set]}`} subtitle="Computed features for this input and class scores">
          <div className="grid gap-5 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Feature values</div>
              <div className="max-h-64 overflow-auto rounded-md border border-slate-200">
                <table className="num w-full text-sm"><tbody className="divide-y divide-slate-100">
                  {Object.entries(sel.features ?? {}).map(([k, v]) => <tr key={k}><td className="px-3 py-1.5 text-slate-600">{k}</td><td className="px-3 py-1.5 text-right">{num(v, 4)}</td></tr>)}
                </tbody></table>
              </div>
            </div>
            <div>
              <div className="label mb-1.5">Class scores</div>
              {sel.probabilities ? (
                <div className="h-56"><ResponsiveContainer><BarChart data={Object.entries(sel.probabilities).map(([k, v]) => ({ k, v: +(v * 100).toFixed(1) }))} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" domain={[0, 100]} unit="%" fontSize={11} /><YAxis dataKey="k" type="category" fontSize={11} width={60} />
                  <Tooltip formatter={(v: number) => `${v}%`} /><Bar dataKey="v" fill="#0e8a8c" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div>
              ) : <p className="text-xs text-slate-500">This classifier does not provide class scores.</p>}
              {sel.probability_note && <p className="mt-1 text-[11px] text-slate-500">{sel.probability_note}</p>}
            </div>
          </div>
        </Card>
      )}
      <Card title="Limitations of this result">
        <ul className="list-disc space-y-1 pl-5 text-xs text-slate-600">
          <li>Models were trained on a small public dataset (hundreds of records; Arcing has only a handful) and have no validation on independent transformer fleets.</li>
          <li>Different models use different feature sets and cleaning protocols, so disagreement between them is expected and informative.</li>
          <li>Per-prediction explanations are not provided; global feature importance exists only for the notebook A Random Forest (see Model comparison).</li>
        </ul>
      </Card>
    </>
  );
}
