import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../services/api";

export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const load = useCallback(() => {
    if (!path) return;
    setLoading(true); setError(null);
    api.get<T>(path).then(setData).catch((e: ApiError) => setError(e.message)).finally(() => setLoading(false));
  }, [path]);
  useEffect(() => { load(); }, [load]);
  return { data, error, loading, reload: load };
}
