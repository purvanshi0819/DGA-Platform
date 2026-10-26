import { useState } from "react";
import { ChevronDown, ChevronRight, Download } from "lucide-react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { ExperimentRow } from "../types";
import { Badge, Card, ErrorBox, Loading, Notice, NotImplemented, PageHeader } from "../components/ui";

export default function History() {
  const { data, error, loading, reload } = useApi<ExperimentRow[]>("/experiments");
  const [open, setOpen] = useState<string | null>(null);
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  return (
    <>
      <PageHeader title="Experiment history and methodology" description="Every number in this application comes from a recorded run. The source notebooks contained no executed outputs, so all results are newly computed by the training pipeline." />
      <Card title="Experiment registry" subtitle="Seed, data hash and library versions are stored for reproducibility">
        {loading || !data ? <Loading /> : data.length === 0 ? <p className="text-sm text-slate-500">No experiments recorded.</p> : (
          <div className="divide-y divide-slate-100">
            {data.map((e) => (
              <div key={e.id}>
                <button className="flex w-full items-center gap-3 py-3 text-left text-sm" onClick={() => setOpen(open === e.id ? null : e.id)}>
                  {open === e.id ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  <span className="flex-1"><span className="font-medium">{e.title}</span><span className="block text-xs text-slate-500">Notebook {e.implementation} · {e.n_rows} records · seed {e.seed}</span></span>
                  <Badge tone="teal">{e.kind}</Badge><Badge tone="green">{e.status}</Badge><span className="num hidden text-xs text-slate-500 md:inline">{e.created_utc?.replace("T", " ")}</span>
                </button>
                {open === e.id && (
                  <div className="mb-3 grid gap-4 rounded-md bg-slate-50 p-4 text-xs md:grid-cols-2">
                    <dl className="space-y-1.5">
                      <div><dt className="label">Origin</dt><dd>{e.provenance.origin} – {e.provenance.note}</dd></div>
                      <div><dt className="label">Cleaning</dt><dd>{e.provenance.cleaning_protocol}</dd></div>
                      <div><dt className="label">Dataset SHA-256</dt><dd className="break-all font-mono">{e.dataset_sha256}</dd></div>
                      <div><dt className="label">Versions</dt><dd>Python {e.provenance.versions?.python} · scikit-learn {e.provenance.versions?.scikit_learn}</dd></div>
                      <div><dt className="label">Runtime</dt><dd>{e.seconds}s · {e.n_records} metric records</dd></div>
                    </dl>
                    <div>
                      <div className="label">Split / validation</div>
                      <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-white p-2 font-mono text-[11px]">{JSON.stringify(e.split, null, 1)}</pre>
                      <a className="btn-secondary mt-2 !py-1 text-xs" href={api.url(`/experiments/${e.id}/export`)} download><Download className="h-3.5 w-3.5" />Metrics CSV</a>
                    </div>
                    {e.warnings.length > 0 && <div className="md:col-span-2"><div className="label">Captured run warnings ({e.warnings.length})</div>
                      <ul className="mt-1 max-h-32 list-disc overflow-auto pl-5 text-slate-600">{Array.from(new Set(e.warnings.map((w) => w.warning))).map((w) => <li key={w}>{w}</li>)}</ul></div>}
                  </div>)}
              </div>))}
          </div>)}
        <div className="mt-4"><NotImplemented>Starting a new training run from the browser. Re-run <code>python scripts/train_all.py</code> in the backend; new results appear here after the API restarts.</NotImplemented></div>
      </Card>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Card title="Architecture"><Arch /></Card>
        <Card title="Methodology">
          <div className="space-y-3 text-sm text-slate-700">
            <Sec t="Data">DuVal.csv, 627 records of H₂, CH₄, C₂H₆, C₂H₄, C₂H₂, CO with a fault label (Normal, PD, OH, Arcing). Exact duplicates are removed by all notebooks; notebook A also drops rows whose gas vector appears with different labels.</Sec>
            <Sec t="Feature engineering">Notebook A: log₁(1+gas), six ratios, total gas and Duval percentages (15). Notebook B: three small input sets (Duval %, Doernenburg ratios, Rogers ratios) with no epsilon. Notebook C: 15 features (raw gases, total hydrocarbons, nine ratios). The same functions are used for training and inference.</Sec>
            <Sec t="Training and validation">All notebooks use a stratified 80/20 split (seed 42). A tunes four models with a grid search and SMOTE inside the pipeline; B evaluates six fixed-parameter classifiers with 10-fold CV; C selects the PCA component count with 5-fold CV and compares baselines over 10 random splits.</Sec>
            <Sec t="Inference">Each deployed model is the notebook's fitted estimator saved with a metadata file (feature set, class order, dataset hash). Predictions are made only on rows whose features can be computed; nothing is imputed.</Sec>
          </div>
        </Card>
        <Card title="Methodological notes">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
            <li>Notebook C's paper rule picks k by test accuracy, which is optimistic; the CV-based choice is used for the final model and reported separately.</li>
            <li>Notebook C's 10-split comparison fixes k from one split; a corrected run (k re-selected per split) is stored as a separate result.</li>
            <li>Notebooks B and C keep conflicting-label rows, which can leak identical inputs between train and test.</li>
            <li>The Arcing class has very few records, making Arcing-specific metrics unreliable.</li>
          </ul>
        </Card>
        <Card title="Limitations and future work">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
            <li>Single public dataset with undocumented units and provenance; no external validation across transformer fleets.</li>
            <li>Classifier scores are uncalibrated; calibration has not been tested.</li>
            <li>No per-prediction explanations; global importance exists only for notebook A's Random Forest.</li>
            <li>The IEC 60599, Duval triangle and Rogers rule-based methods are not implemented as rules.</li>
            <li>Future work: cross-dataset testing, probability calibration, rule-based baselines, in-app retraining.</li>
          </ul>
          <div className="mt-3"><Notice tone="warn">Academic research prototype. Outputs are decision support and require review by qualified transformer engineers.</Notice></div>
        </Card>
      </div>
    </>
  );
}
const Sec = ({ t, children }: { t: string; children: React.ReactNode }) => <div><div className="label">{t}</div><p className="mt-0.5">{children}</p></div>;

function Arch() {
  const box = (x: number, y: number, w: number, h: number, t: string, s: string, fill: string) => (
    <g><rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke="#1b3a56" strokeWidth={1} /><text x={x + w / 2} y={y + 20} textAnchor="middle" fontSize={11} fontWeight={600} fill="#0b1b2b">{t}</text><text x={x + w / 2} y={y + 36} textAnchor="middle" fontSize={9} fill="#475569">{s}</text></g>);
  const arrow = (x1: number, y1: number, x2: number, y2: number) => <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#2a4d6e" strokeWidth={1.3} markerEnd="url(#ah)" />;
  return (
    <svg viewBox="0 0 520 300" className="w-full" role="img" aria-label="Architecture diagram">
      <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#2a4d6e" /></marker></defs>
      {box(10, 10, 150, 50, "React + Vite UI", "8 pages · Recharts", "#e8fafa")}
      {box(190, 10, 150, 50, "FastAPI", "REST · Pydantic validation", "#e8fafa")}
      {box(370, 10, 140, 50, "SQLite", "experiment index · activity", "#fff8e6")}
      {box(10, 110, 150, 50, "CSV upload", "validated, size-limited", "#f3f6f9")}
      {box(190, 110, 150, 50, "Inference service", "features → saved pipeline", "#e8fafa")}
      {box(370, 110, 140, 50, "Artifacts", "joblib models + JSON metadata", "#fff8e6")}
      {box(10, 210, 150, 50, "DuVal.csv (raw)", "immutable · SHA-256", "#f3f6f9")}
      {box(190, 210, 150, 50, "ML modules", "A: RF · B: DGA · C: PCA+ANN", "#e8fafa")}
      {box(370, 210, 140, 50, "train_all.py", "experiments + model export", "#fff8e6")}
      {arrow(160, 35, 190, 35)}{arrow(340, 35, 370, 35)}{arrow(85, 110, 230, 60)}{arrow(265, 60, 265, 110)}{arrow(340, 135, 370, 135)}
      {arrow(440, 210, 440, 160)}{arrow(370, 235, 340, 235)}{arrow(190, 235, 160, 235)}{arrow(265, 210, 265, 160)}
      <text x="260" y="290" textAnchor="middle" fontSize="9" fill="#64748b">Training and inference are separate; the API never retrains.</text>
    </svg>
  );
}
