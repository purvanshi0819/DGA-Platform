import { useMemo, useState } from "react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { checkGas, num } from "../utils/format";
import { FeatureSetInfo, FS_LABEL, GASES, MetricRow } from "../types";
import { Badge, Card, ErrorBox, Loading, Notice, PageHeader } from "../components/ui";

interface Computed { id: string; status: "ok" | "not_computable"; values?: Record<string, number>; message?: string }
const SOURCE: Record<string, { exp: string; stage: string }> = {
  duval3: { exp: "dga_main", stage: "test" }, doern4: { exp: "dga_main", stage: "test" }, rogers3: { exp: "dga_main", stage: "test" },
  rf15: { exp: "rfrep_main", stage: "test" }, pca15: { exp: "pcaann_main", stage: "mean_10_random_splits" },
  raw5: { exp: "pcaann_main", stage: "mean_10_random_splits" }, iec3: { exp: "pcaann_main", stage: "mean_10_random_splits" },
};

export default function GasFeatures() {
  const fsets = useApi<FeatureSetInfo[]>("/feature-sets");
  const b = useApi<MetricRow[]>("/experiments/dga_main/metrics");
  const a = useApi<MetricRow[]>("/experiments/rfrep_main/metrics");
  const c = useApi<MetricRow[]>("/experiments/pcaann_main/metrics");
  const [vals, setVals] = useState<Record<string, string>>({ H2: "", CH4: "", C2H6: "", C2H4: "", C2H2: "" });
  const [calc, setCalc] = useState<Computed[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState("duval3");
  const checks = useMemo(() => Object.fromEntries(GASES.map((g) => [g.key, checkGas(vals[g.key], true)])), [vals]);
  const ok = Object.values(checks).every((x) => !x.error);

  async function compute() {
    setErr(null);
    const reading = Object.fromEntries(Object.entries(checks).map(([k, v]) => [k, v.value]));
    try { setCalc(await api.post<Computed[]>("/features", reading)); } catch (e: any) { setErr(e.message); }
  }
  if (fsets.error) return <ErrorBox message={fsets.error} onRetry={fsets.reload} />;
  if (fsets.loading || !fsets.data) return <Loading />;
  const fs = fsets.data.find((f) => f.id === active)!;
  const cur = calc?.find((x) => x.id === active);
  const metrics: MetricRow[] = [...(a.data ?? []), ...(b.data ?? []), ...(c.data ?? [])];
  const src = SOURCE[active];
  const related = metrics.filter((m) => m.experiment === src.exp && m.stage === src.stage && (m.feature_set === active || (src.exp === "pcaann_main" && active === "pcaann")));

  // notebook C compares feature sets only through model names, so match by name
  const cRows = metrics.filter((m) => m.experiment === "pcaann_main" && m.stage === "mean_10_random_splits");
  const cMatch = active === "raw5" ? cRows.filter((m) => /raw 5/.test(m.model)) : active === "iec3" ? cRows.filter((m) => /IEC 3/.test(m.model)) : active === "pca15" ? cRows.filter((m) => /15|PCA/.test(m.model)) : [];
  const evalRows = src.exp === "pcaann_main" ? cMatch : related;

  return (
    <>
      <PageHeader title="Gas feature methods" description="How each notebook turns dissolved-gas readings into classifier inputs. These are machine-learning input representations; they are not the full rule-based Duval, Doernenburg or Rogers diagnostic procedures." />
      <Notice><b>Distinction:</b> the "Duval", "Doernenburg" and "Rogers" cases in notebook B use only the gas percentages/ratios as features for trained classifiers. The notebooks do not implement the triangle zones or the ratio-range decision tables of those standards.</Notice>

      <div className="mt-5 grid gap-5 xl:grid-cols-3">
        <div className="space-y-5">
          <Card title="Try it on a reading" subtitle="Computes every feature set with the backend's exact code">
            <div className="grid grid-cols-2 gap-3">{GASES.map((g) => { const ck = checks[g.key]; const show = vals[g.key] !== "" && ck.error;
              return <div key={g.key}><label htmlFor={`f-${g.key}`} className="text-sm font-medium">{g.sub}</label><input id={`f-${g.key}`} className={`input mt-1 ${show ? "input-error" : ""}`} inputMode="decimal" value={vals[g.key]} onChange={(e) => { setVals({ ...vals, [g.key]: e.target.value }); setCalc(null); }} />{show && <p className="mt-1 text-[11px] text-red-600">{ck.error}</p>}</div>; })}</div>
            <button className="btn-primary mt-4 w-full" disabled={!ok} onClick={compute}>Calculate features</button>
            {err && <div className="mt-3"><ErrorBox message={err} /></div>}
          </Card>
          <Card title="Feature sets">
            <div className="space-y-1">{fsets.data.map((f) => (
              <button key={f.id} onClick={() => setActive(f.id)} className={`flex w-full items-center justify-between rounded-md border px-3 py-2 text-left text-sm ${active === f.id ? "border-brand-500 bg-brand-50" : "border-slate-200 hover:bg-slate-50"}`}>
                <span><span className="font-medium">{FS_LABEL[f.id] ?? f.label}</span><span className="block text-[11px] text-slate-500">Notebook {f.notebook} · {f.feature_names.length} features</span></span>
                {calc && <Badge tone={calc.find((x) => x.id === f.id)?.status === "ok" ? "green" : "amber"}>{calc.find((x) => x.id === f.id)?.status === "ok" ? "ok" : "n/a"}</Badge>}</button>))}</div>
          </Card>
        </div>

        <div className="space-y-5 xl:col-span-2">
          <Card title={`${FS_LABEL[fs.id] ?? fs.label} · notebook ${fs.notebook}`} subtitle={fs.uses_epsilon ? "Ratios use ε = 10⁻⁶ in the denominator (as in the notebook)" : "No epsilon in the source notebook – zero denominators are invalid, not replaced"}>
            {fs.note && <p className="mb-3 text-xs text-slate-600">{fs.note}</p>}
            <div className="overflow-x-auto rounded-md border border-slate-200"><table className="num w-full text-sm"><thead className="bg-slate-50 text-xs text-slate-600"><tr><th className="px-3 py-2 text-left">Feature / group</th><th className="px-3 py-2 text-left">Definition</th><th className="px-3 py-2 text-right">Current input</th></tr></thead>
              <tbody className="divide-y divide-slate-100">{Object.entries(fs.formulas).map(([k, v]) => {
                const direct = cur?.values?.[k];
                return <tr key={k}><td className="px-3 py-1.5 font-medium">{k}</td><td className="px-3 py-1.5 font-mono text-xs text-slate-600">{v}</td><td className="px-3 py-1.5 text-right">{direct != null ? num(direct, 4) : "–"}</td></tr>; })}</tbody></table></div>
            {cur?.status === "not_computable" && <div className="mt-3"><Notice tone="warn">Not computable for this input: {cur.message}</Notice></div>}
            {cur?.status === "ok" && <div className="mt-4"><div className="label mb-1">All {fs.feature_names.length} computed values</div><div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm md:grid-cols-3">{Object.entries(cur.values!).map(([k, v]) => <div key={k} className="flex justify-between border-b border-slate-100 py-1"><span className="text-slate-600">{k}</span><span className="num">{num(v, 4)}</span></div>)}</div></div>}
            {!calc && <p className="mt-3 text-xs text-slate-500">Enter a reading and press <b>Calculate features</b> to see values here.</p>}
          </Card>
          <Card title="Classifiers evaluated with this feature set" subtitle={src.exp === "dga_main" ? "Notebook B · held-out 20% test, weighted averages" : src.exp === "rfrep_main" ? "Notebook A · held-out 20% test, macro averages" : "Notebook C · mean over 10 random 80/20 splits (accuracy only)"}>
            {evalRows.length === 0 ? <p className="text-sm text-slate-500">No matching experiment results (the notebook did not evaluate this set on its own, or experiments have not been run).</p> : (
              <div className="overflow-x-auto rounded-md border border-slate-200"><table className="num w-full text-sm"><thead className="bg-slate-50 text-xs text-slate-600"><tr><th className="px-3 py-2 text-left">Classifier</th><th className="px-3 py-2 text-right">Accuracy</th><th className="px-3 py-2 text-right">{src.exp === "rfrep_main" ? "Macro F1" : "Weighted F1"}</th><th className="px-3 py-2 text-right">n</th></tr></thead>
                <tbody className="divide-y divide-slate-100">{evalRows.map((r, i) => <tr key={i}><td className="px-3 py-1.5 font-medium">{r.model}</td><td className="px-3 py-1.5 text-right">{r.accuracy != null ? `${(r.accuracy * 100).toFixed(1)}%` : "–"}{r.accuracy_std != null ? <span className="text-slate-400"> ±{(r.accuracy_std * 100).toFixed(1)}</span> : null}</td><td className="px-3 py-1.5 text-right">{(src.exp === "rfrep_main" ? r.macro_f1 : r.weighted_f1) != null ? `${(((src.exp === "rfrep_main" ? r.macro_f1 : r.weighted_f1) as number) * 100).toFixed(1)}%` : "–"}</td><td className="px-3 py-1.5 text-right">{r.n}</td></tr>)}</tbody></table></div>)}
            <p className="mt-2 text-[11px] text-slate-500">Metrics are from different protocols per notebook (see Model comparison); compare feature sets only within the same notebook.</p>
          </Card>
        </div>
      </div>
    </>
  );
}
