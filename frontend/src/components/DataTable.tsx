import { ReactNode, useMemo, useState } from "react";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";

export interface Col<T> { key: string; header: string; render?: (r: T) => ReactNode; sortValue?: (r: T) => string | number; align?: "right" }

export function DataTable<T extends Record<string, any>>({ rows, columns, pageSize = 10, searchText, filters }: {
  rows: T[]; columns: Col<T>[]; pageSize?: number; searchText?: (r: T) => string;
  filters?: { label: string; options: { value: string; label: string }[]; test: (r: T, v: string) => boolean }[];
}) {
  const [q, setQ] = useState(""); const [page, setPage] = useState(0);
  const [fv, setFv] = useState<Record<string, string>>({});
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const filtered = useMemo(() => {
    let r = rows;
    if (q && searchText) r = r.filter((x) => searchText(x).toLowerCase().includes(q.toLowerCase()));
    (filters ?? []).forEach((f) => { const v = fv[f.label]; if (v) r = r.filter((x) => f.test(x, v)); });
    if (sort) {
      const c = columns.find((c) => c.key === sort.key);
      const val = (x: T) => (c?.sortValue ? c.sortValue(x) : x[sort.key]);
      r = [...r].sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort.dir);
    }
    return r;
  }, [rows, q, fv, sort, columns, searchText, filters]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const cur = Math.min(page, pages - 1);
  const slice = filtered.slice(cur * pageSize, cur * pageSize + pageSize);
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {searchText && (
          <div className="relative"><Search className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-400" />
            <input className="input !w-56 !pl-8" placeholder="Search…" value={q} aria-label="Search rows" onChange={(e) => { setQ(e.target.value); setPage(0); }} /></div>
        )}
        {(filters ?? []).map((f) => (
          <select key={f.label} className="input !w-auto" aria-label={f.label} value={fv[f.label] ?? ""} onChange={(e) => { setFv({ ...fv, [f.label]: e.target.value }); setPage(0); }}>
            <option value="">{f.label}: all</option>{f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        ))}
        <span className="ml-auto text-xs text-slate-500">{filtered.length} row{filtered.length === 1 ? "" : "s"}</span>
      </div>
      <div className="overflow-x-auto rounded-md border border-slate-200">
        <table className="num w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs text-slate-600">
            <tr>{columns.map((c) => (
              <th key={c.key} className={`px-3 py-2 font-medium ${c.align === "right" ? "text-right" : ""}`}>
                <button className="hover:text-slate-900" onClick={() => setSort(sort?.key === c.key ? { key: c.key, dir: (sort.dir * -1) as 1 | -1 } : { key: c.key, dir: 1 })}>
                  {c.header}{sort?.key === c.key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}</button></th>))}</tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {slice.length === 0 && <tr><td colSpan={columns.length} className="px-3 py-8 text-center text-slate-500">No matching rows</td></tr>}
            {slice.map((r, i) => <tr key={i} className="hover:bg-slate-50">{columns.map((c) => <td key={c.key} className={`px-3 py-2 ${c.align === "right" ? "text-right" : ""}`}>{c.render ? c.render(r) : String(r[c.key] ?? "–")}</td>)}</tr>)}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center justify-end gap-2 text-xs text-slate-600">
        <button className="btn-secondary !p-1.5" disabled={cur === 0} onClick={() => setPage(cur - 1)} aria-label="Previous page"><ChevronLeft className="h-4 w-4" /></button>
        <span>Page {cur + 1} of {pages}</span>
        <button className="btn-secondary !p-1.5" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)} aria-label="Next page"><ChevronRight className="h-4 w-4" /></button>
      </div>
    </div>
  );
}
