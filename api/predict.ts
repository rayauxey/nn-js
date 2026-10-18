// Vercel Function (Bun runtime) for POST /api/predict. The page itself is a
// static build of web/ (bun run build); model.json ships via includeFiles.

import { loadModel, MODEL_PATH, predictHandler } from "../src/predict.ts";

const net = await loadModel();
if (!net) throw new Error(`No ${MODEL_PATH} found in the deployment`);

Bun.serve({
  routes: {
    "/api/predict": { POST: predictHandler(net) },
  },
  fetch: () => Response.json({ error: "Use POST /api/predict" }, { status: 405 }),
});
