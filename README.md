# nn-js

A fully connected neural network in TypeScript (no ML libraries) that classifies MNIST digits.

**Network:** 784 → 256 (ReLU) → 128 (ReLU) → 10 (softmax), He init, dropout 0.2, Adam (lr 0.001), batch size 64, cross-entropy loss.

```bash
bun install
bun run train      # downloads MNIST into data/ on first run, saves model.json
bun run evaluate   # test accuracy + a few ASCII-rendered predictions
bun run serve      # web app at http://localhost:3000: draw a digit or drop in a photo of one
bun test           # checks preprocessing on simulated photos of test digits
bun run build      # static site in dist/ (used by the Vercel deployment)
bun run visualize  # rebuilds nn-visualization.html from model.json + history.json (--pdf also prints a PDF via Chrome)
```

Set `EPOCHS=3 bun run train` for a quicker run. Expected test accuracy is ~98%.

## Layout

- `src/mlp.ts` – network: forward pass, backprop, Adam, save/load
- `src/mnist.ts` – downloads and parses the IDX files
- `src/train.ts` – training loop with validation split and best-checkpoint saving
- `src/evaluate.ts` – accuracy helper and evaluation CLI
- `src/visualize.ts` – illustrated walkthrough page, served at `/visualization.html`
- `src/server.ts` – serves `web/` plus `POST /api/predict` ({ pixels: number[784] })
- `src/preprocess.ts` – converts any digit image to MNIST format (crop, 20×20, center of mass)
- `web/` – drawing pad, image upload, and probability display
- `api/predict.ts`, `vercel.json` – Vercel deployment (Bun runtime); pushes to `main` deploy automatically
- `src/random.ts` – seeded RNG (uniform, gaussian, shuffle)
