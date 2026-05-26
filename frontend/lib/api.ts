import type { SolveResponse, GenerateResponse, ModelStatus, ModelName } from "./types";

const BASE = "/api/backend";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Erreur API");
  }
  return res.json();
}

export async function solveGrid(puzzle: number[]): Promise<SolveResponse> {
  const res = await fetch(`${BASE}/solve/grid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ puzzle }),
  });
  return handleResponse(res);
}

export async function solveImage(file: File): Promise<SolveResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/solve/image`, { method: "POST", body: form });
  return handleResponse(res);
}

export async function generatePuzzle(clues = 30): Promise<GenerateResponse> {
  const res = await fetch(`${BASE}/generate?clues=${clues}`);
  return handleResponse(res);
}

export async function getModelsStatus(): Promise<Record<ModelName, ModelStatus>> {
  const res = await fetch(`${BASE}/models/status`);
  return handleResponse(res);
}

export async function startTraining(quick = false, model?: string): Promise<{ message: string }> {
  const res = await fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quick, model }),
  });
  return handleResponse(res);
}
