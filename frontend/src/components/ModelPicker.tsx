import { ModelInfo, IMPL_NAMES, FS_LABEL } from "../types";

export function ModelPicker({ models, selected, onChange }: { models: ModelInfo[]; selected: string[]; onChange: (ids: string[]) => void }) {
  const groups = ["A", "B", "C"].map((k) => ({ k, items: models.filter((m) => m.implementation === k) })).filter((g) => g.items.length);
  const toggle = (id: string) => onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  return (
    <div className="space-y-4">
      {groups.map((g) => (
        <fieldset key={g.k}>
          <legend className="mb-1.5 flex w-full items-center justify-between text-xs font-semibold text-slate-700">
            <span>Notebook {g.k} · {IMPL_NAMES[g.k]}</span>
            <span className="space-x-2 font-normal">
              <button type="button" className="text-brand-600 hover:underline" onClick={() => onChange(Array.from(new Set([...selected, ...g.items.map((m) => m.id)])))}>all</button>
              <button type="button" className="text-slate-500 hover:underline" onClick={() => onChange(selected.filter((s) => !g.items.some((m) => m.id === s)))}>none</button>
            </span>
          </legend>
          <div className="grid grid-cols-1 gap-1 sm:grid-cols-2">
            {g.items.map((m) => (
              <label key={m.id} className={`flex cursor-pointer items-center gap-2 rounded border px-2.5 py-1.5 text-xs ${selected.includes(m.id) ? "border-brand-500 bg-brand-50" : "border-slate-200 hover:bg-slate-50"}`}>
                <input type="checkbox" className="accent-brand-600" checked={selected.includes(m.id)} onChange={() => toggle(m.id)} />
                <span className="font-medium text-slate-800">{m.model_name}</span>
                <span className="ml-auto text-slate-500">{FS_LABEL[m.feature_set] ?? m.feature_set}</span>
              </label>
            ))}
          </div>
        </fieldset>
      ))}
    </div>
  );
}
