// Simulates "photos" of real MNIST test digits (dark ink on noisy light paper,
// enlarged and off-centre) and checks the preprocessed result still classifies well.

import { expect, test } from "bun:test";
import { loadMnist } from "./mnist.ts";
import { MLP, type MLPJSON } from "./mlp.ts";
import { resizeBox, toMnist } from "./preprocess.ts";
import { createRng } from "./random.ts";

const modelFile = Bun.file("model.json");
const SAMPLES = 1000;

test.skipIf(!(await modelFile.exists()))("simulated paper photos of test digits classify correctly", async () => {
  const net = MLP.fromJSON((await modelFile.json()) as MLPJSON);
  const { test: set } = await loadMnist();
  const rng = createRng(7);

  let correct = 0;
  for (let i = 0; i < SAMPLES; i++) {
    const digit = set.images.subarray(i * 784, (i + 1) * 784);

    // Enlarge 10x and drop onto a 400x330 sheet at a random offset.
    const big = resizeBox(digit, 28, 28, 280, 280);
    const W = 400;
    const H = 330;
    const ox = Math.floor(rng.next() * (W - 280));
    const oy = Math.floor(rng.next() * (H - 280));
    const paper = new Float32Array(W * H);
    for (let k = 0; k < paper.length; k++) paper[k] = 0.9 + (rng.next() - 0.5) * 0.08;
    for (let y = 0; y < 280; y++) {
      for (let x = 0; x < 280; x++) paper[(oy + y) * W + ox + x]! -= 0.75 * big[y * 280 + x]!;
    }

    const pixels = toMnist(paper, W, H);
    expect(pixels).not.toBeNull();
    if (net.predict(pixels!, 1)[0] === set.labels[i]) correct++;
  }

  const accuracy = correct / SAMPLES;
  console.log(`accuracy on simulated photos: ${(accuracy * 100).toFixed(1)}%`);
  expect(accuracy).toBeGreaterThan(0.95);
});

test("blank image returns null", () => {
  const blank = new Float32Array(100 * 100).fill(0.95);
  expect(toMnist(blank, 100, 100)).toBeNull();
});
