import { ReactNode } from "react";
import { AlertTriangle, Info, Loader2, Inbox } from "lucide-react";

export function Card({ title, subtitle, action, children, className = "" }: { title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, accent = false }: { label: string; value: ReactNode; hint?: ReactNode; accent?: boolean }) {
  return (
    <div className="card px-4 py-3.5">
      <div className="label">{label}</div>
      <div className={`num mt-1 text-2xl font-semibold ${accent ? "text-brand-600" : "text-slate-900"}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div className="max-w-3xl">
        <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
        {description && <p className="mt-1 text-sm text-slate-600">{description}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return <div className="flex items-center gap-2 py-10 text-sm text-slate-500" role="status"><Loader2 className="h-4 w-4 animate-spin" />{label}</div>;
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1">{message}</div>
      {onRetry && <button className="btn-secondary !py-1" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-1 rounded-md border border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center">
      <Inbox className="h-5 w-5 text-slate-400" /><div className="text-sm font-medium text-slate-700">{title}</div>
      {children && <div className="max-w-md text-xs text-slate-500">{children}</div>}
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warn"; children: ReactNode }) {
  const c = tone === "warn" ? "border-amber-200 bg-amber-50 text-amber-900" : "border-brand-100 bg-brand-50 text-ink-800";
  return <div className={`flex gap-2.5 rounded-md border p-3 text-xs leading-relaxed ${c}`}><Info className="mt-0.5 h-3.5 w-3.5 shrink-0" /><div>{children}</div></div>;
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "teal" | "amber" | "red" | "green" }) {
  const t = { slate: "bg-slate-100 text-slate-700", teal: "bg-brand-50 text-brand-700", amber: "bg-amber-50 text-amber-700", red: "bg-red-50 text-red-700", green: "bg-emerald-50 text-emerald-700" }[tone];
  return <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-medium ${t}`}>{children}</span>;
}

export const DISCLAIMER = "Research decision support only. Model predictions are not confirmed engineering diagnoses and must be validated by qualified transformer engineers.";
export function Disclaimer() { return <Notice tone="warn">{DISCLAIMER}</Notice>; }

export function NotImplemented({ children }: { children: ReactNode }) {
  return <div className="flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"><Badge>Not implemented</Badge>{children}</div>;
}

export const CHART_COLORS = ["#0e8a8c", "#e8a317", "#2a4d6e", "#7aa7c7", "#c4860b", "#14a8a8", "#8a96a3"];
