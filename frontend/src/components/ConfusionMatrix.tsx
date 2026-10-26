import type { Confusion } from "../types";

/** Row-normalised heat map with raw counts printed in every cell. Rows = true class, columns = predicted. */
export function ConfusionMatrix({ cm }: { cm: Confusion }) {
  const { labels, matrix } = cm;
  return (
    <div className="overflow-x-auto">
      <table className="num mx-auto border-separate border-spacing-1 text-sm" aria-label="Confusion matrix">
        <thead>
          <tr><th /><th colSpan={labels.length} className="pb-1 text-center text-[11px] font-medium uppercase tracking-wide text-slate-500">Predicted</th></tr>
          <tr><th />{labels.map((l) => <th key={l} className="px-3 py-1 text-xs font-medium text-slate-600">{l}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={labels[i]}>
                <th className="pr-2 text-right text-xs font-medium text-slate-600">{labels[i]}<div className="text-[10px] font-normal text-slate-400">n={total}</div></th>
                {row.map((v, j) => {
                  const f = total ? v / total : 0;
                  const diag = i === j;
                  const bg = diag ? `rgba(14,138,140,${0.12 + 0.75 * f})` : v ? `rgba(232,163,23,${0.15 + 0.6 * f})` : "#f8fafc";
                  return <td key={j} title={`True ${labels[i]} → predicted ${labels[j]}: ${v} (${(f * 100).toFixed(0)}% of row)`}
                    className="h-12 w-16 rounded text-center font-medium" style={{ background: bg, color: f > 0.55 && diag ? "#fff" : "#0f172a" }}>{v}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-center text-[11px] text-slate-500">Teal = correct (diagonal), amber = misclassified. Shade shows share of the true-class row.</p>
    </div>
  );
}
