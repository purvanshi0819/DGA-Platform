export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000/api";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

function detailToText(d: unknown): string {
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((x: any) => `${(x.loc ?? []).filter((p: string) => p !== "body").join(".")}: ${x.msg}`).join("; ");
  }
  return "Request failed";
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = detailToText((await r.json()).detail); } catch { /* body was not JSON */ }
    throw new ApiError(msg, r.status);
  }
  return r.json() as Promise<T>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    return await handle<T>(await fetch(`${API_BASE}${path}`, init));
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(`Cannot reach the backend at ${API_BASE}. Is the FastAPI server running?`, 0);
  }
}

export const api = {
  get: <T,>(p: string) => request<T>(p),
  post: <T,>(p: string, body: unknown) =>
    request<T>(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  upload: <T,>(p: string, file: File, fields: Record<string, string> = {}) => {
    const fd = new FormData(); fd.append("file", file);
    Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
    return request<T>(p, { method: "POST", body: fd });
  },
  url: (p: string) => `${API_BASE}${p}`,
};
