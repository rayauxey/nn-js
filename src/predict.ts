// POST /api/predict handler, shared by the local server and the Vercel function.

import { MLP, type MLPJSON } from "./mlp.ts";

export const MODEL_PATH = "model.json";

/** Strongest connections returned per layer transition, for the network view. */
const TRACE_EDGES = [160, 120, 60];

/** A connection's contribution to a forward pass: [from unit, to unit, weight * activation]. */
export type TraceEdge = [number, number, number];

export interface Trace {
  /** Activations of each hidden layer. */
  hidden: number[][];
  /** Per layer transition, the connections that contributed most (by |weight * activation|). */
  edges: TraceEdge[][];
}

/** Loads the trained network from model.json in the working directory, or null if it's missing. */
export async function loadModel(path = MODEL_PATH): Promise<MLP | null> {
  const file = Bun.file(path);
  if (!(await file.exists())) return null;
  return MLP.fromJSON((await file.json()) as MLPJSON);
}

// Body: { "pixels": number[784] } in MNIST format (white ink on black, 0..1).
export function predictHandler(net: MLP) {
  const inputSize = net.sizes[0]!;
  return async (req: Request): Promise<Response> => {
    const body = (await req.json().catch(() => null)) as { pixels?: unknown } | null;
    const pixels = body?.pixels;
    if (!Array.isArray(pixels) || pixels.length !== inputSize || !pixels.every((p) => typeof p === "number")) {
      return Response.json({ error: `Expected { pixels: number[${inputSize}] }` }, { status: 400 });
    }
    const input = Float32Array.from(pixels as number[], (p) => Math.min(1, Math.max(0, p)));
    const { acts } = net.forward(input, 1);
    const probabilities = Array.from(acts[acts.length - 1]!);
    const prediction = probabilities.indexOf(Math.max(...probabilities));
    return Response.json({ prediction, probabilities, trace: trace(net, acts) });
  };
}

function trace(net: MLP, acts: Float32Array[]): Trace {
  const round = (v: number) => Number(v.toPrecision(4));
  return {
    hidden: acts.slice(1, -1).map((a) => Array.from(a, round)),
    edges: net.weights.map((W, l) => {
      const inSize = net.sizes[l]!;
      const outSize = net.sizes[l + 1]!;
      const prev = acts[l]!;
      const edges: TraceEdge[] = [];
      for (let i = 0; i < inSize; i++) {
        const a = prev[i]!;
        if (a === 0) continue; // silent units contribute nothing
        for (let j = 0; j < outSize; j++) edges.push([i, j, W[j * inSize + i]! * a]);
      }
      edges.sort((x, y) => Math.abs(y[2]) - Math.abs(x[2]));
      return edges.slice(0, TRACE_EDGES[l]).map(([i, j, c]) => [i, j, round(c)] as TraceEdge);
    }),
  };
}
