import { NavLink, Outlet } from "react-router-dom";
import { Activity, BarChart3, Database, FlaskConical, GitCompare, History, LayoutDashboard, Layers, Upload, Zap } from "lucide-react";
import { useApi } from "../hooks/useApi";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/diagnosis", label: "Diagnosis", icon: Zap },
  { to: "/batch", label: "Batch analysis", icon: Upload },
  { to: "/compare", label: "Model comparison", icon: GitCompare },
  { to: "/pca", label: "PCA + ANN lab", icon: FlaskConical },
  { to: "/features", label: "Gas feature methods", icon: Layers },
  { to: "/data", label: "Data quality", icon: Database },
  { to: "/history", label: "History & method", icon: History },
];

export default function AppLayout() {
  const { data, error } = useApi<{ models_available: number; experiments_available: number }>("/health");
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-ink-900 text-slate-300 md:flex">
        <div className="flex items-center gap-2.5 border-b border-ink-700 px-5 py-4">
          <div className="grid h-8 w-8 place-items-center rounded bg-brand-600"><Activity className="h-4 w-4 text-white" /></div>
          <div><div className="text-sm font-semibold leading-tight text-white">DGA Diagnosis</div><div className="text-[11px] text-slate-400">Transformer fault analytics</div></div>
        </div>
        <nav className="flex-1 space-y-0.5 p-3" aria-label="Primary">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => `flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${isActive ? "bg-ink-700 text-white" : "hover:bg-ink-800 hover:text-white"}`}>
              <n.icon className="h-4 w-4" />{n.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-ink-700 p-4 text-[11px] text-slate-400">
          <div className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${error ? "bg-red-400" : data ? "bg-emerald-400" : "bg-amber-500"}`} />
            {error ? "Backend unreachable" : data ? `API online · ${data.models_available} models` : "Connecting…"}
          </div>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3 md:hidden">
          <span className="text-sm font-semibold">DGA Diagnosis</span>
          <nav className="flex gap-3 overflow-x-auto text-xs">{NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? "font-semibold text-brand-600" : "text-slate-600")}>{n.label}</NavLink>)}</nav>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-6 lg:px-8"><Outlet /></main>
        <footer className="border-t border-slate-200 bg-white px-8 py-3 text-center text-[11px] text-slate-500">
          <BarChart3 className="mr-1 inline h-3 w-3" />Academic research prototype · BTech final-year project · Outputs are research decision support, not confirmed diagnoses.
        </footer>
      </div>
    </div>
  );
}
