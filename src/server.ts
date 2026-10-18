// Serves the digit-drawing web app and a prediction API backed by model.json.

import index from "../web/index.html";
import { MLP, type MLPJSON } from "./mlp.ts";

const MODEL_PATH = "model.json";
const PORT = Number(process.env.PORT ?? 3000);

const modelFile = Bun.file(MODEL_PATH);
if (!(await modelFile.exists())) {
  console.error(`No ${MODEL_PATH} found. Train a model first with: bun run train`);
  process.exit(1);
}
const net = MLP.fromJSON((await modelFile.json()) as MLPJSON);
const inputSize = net.sizes[0]!;

const server = Bun.serve({
  port: PORT,
  routes: {
    "/": index,
    "/api/predict": {
      // Body: { "pixels": number[784] } in MNIST format (white ink on black, 0..1).
      POST: async (req) => {
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
      },
    },
  },
});

console.log(`Digit classifier running at ${server.url}`);
