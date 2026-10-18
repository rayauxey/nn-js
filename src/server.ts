// Serves the digit-drawing web app and a prediction API backed by model.json.

import index from "../web/index.html";
import { loadModel, MODEL_PATH, predictHandler } from "./predict.ts";

const PORT = Number(process.env.PORT ?? 3000);
const VISUALIZATION_PATH = "nn-visualization.html";

const net = await loadModel();
if (!net) {
  console.error(`No ${MODEL_PATH} found. Train a model first with: bun run train`);
  process.exit(1);
}

const server = Bun.serve({
  port: PORT,
  routes: {
    "/": index,
    "/visualization.html": async () => {
      const page = Bun.file(VISUALIZATION_PATH);
      return (await page.exists())
        ? new Response(page)
        : new Response("Not generated yet. Run: bun run visualize", { status: 404 });
    },
    "/api/predict": { POST: predictHandler(net) },
  },
});

console.log(`Digit classifier running at ${server.url}`);
