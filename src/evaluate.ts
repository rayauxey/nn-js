// Accuracy helper, and a CLI that scores a saved model on the MNIST test set.

import { loadMnist, type Dataset } from "./mnist.ts";
import { MLP, type MLPJSON } from "./mlp.ts";

export function accuracy(net: MLP, set: Dataset, batchSize = 1000): number {
  let correct = 0;
  for (let start = 0; start < set.count; start += batchSize) {
    const n = Math.min(batchSize, set.count - start);
    const input = set.images.subarray(start * set.inputSize, (start + n) * set.inputSize);
    const predictions = net.predict(input, n);
    for (let s = 0; s < n; s++) {
      if (predictions[s] === set.labels[start + s]) correct++;
    }
  }
  return correct / set.count;
}

// Renders a 28x28 digit as ASCII art.
function drawDigit(pixels: Float32Array): string {
  const shades = " .:-=+*#%@";
  let out = "";
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      out += shades[Math.min(shades.length - 1, Math.floor(pixels[r * 28 + c]! * shades.length))];
    }
    out += "\n";
  }
  return out;
}

if (import.meta.main) {
  const modelPath = process.argv[2] ?? "model.json";
  const net = MLP.fromJSON((await Bun.file(modelPath).json()) as MLPJSON);
  const { test } = await loadMnist();

  console.log(`Test accuracy: ${(accuracy(net, test) * 100).toFixed(2)}%\n`);

  // Show a few random test samples with their predictions.
  for (let k = 0; k < 3; k++) {
    const idx = Math.floor(Math.random() * test.count);
    const pixels = test.images.subarray(idx * test.inputSize, (idx + 1) * test.inputSize);
    const [prediction] = net.predict(pixels, 1);
    console.log(drawDigit(pixels));
    console.log(`Label: ${test.labels[idx]}  Prediction: ${prediction}\n`);
  }
}
