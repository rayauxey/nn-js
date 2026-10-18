// POST /api/predict handler, shared by the local server and the Vercel function.

import { MLP, type MLPJSON } from "./mlp.ts";

export const MODEL_PATH = "model.json";

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
    return Response.json({ prediction, probabilities });
  };
}
