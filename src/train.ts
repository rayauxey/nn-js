// Trains a 784-256-128-10 MLP on MNIST and saves it to model.json.

import { loadMnist, splitSet } from "./mnist.ts";
import { MLP, type MLPJSON } from "./mlp.ts";
import { createRng } from "./random.ts";
import { accuracy } from "./evaluate.ts";

const CONFIG = {
  layers: [784, 256, 128, 10],
  dropout: 0.2,
  learningRate: 0.001,
  batchSize: 64,
  epochs: Number(process.env.EPOCHS ?? 15),
  validationSize: 5000,
  seed: 42,
  modelPath: "model.json",
};

const { train: fullTrain, test } = await loadMnist();
const [train, validation] = splitSet(fullTrain, CONFIG.validationSize);
console.log(`Train: ${train.count}  Validation: ${validation.count}  Test: ${test.count}`);
console.log(`Network: ${CONFIG.layers.join(" → ")}, dropout ${CONFIG.dropout}\n`);

const net = new MLP(CONFIG.layers, { dropout: CONFIG.dropout, seed: CONFIG.seed });
const rng = createRng(CONFIG.seed + 1);
const order = Array.from({ length: train.count }, (_, i) => i);
const inputSize = train.inputSize;
const batchInput = new Float32Array(CONFIG.batchSize * inputSize);
const batchLabels = new Uint8Array(CONFIG.batchSize);
let bestValAcc = 0;

for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
  rng.shuffle(order);
  const started = performance.now();
  let lossSum = 0;
  let batches = 0;

  for (let start = 0; start < train.count; start += CONFIG.batchSize) {
    const n = Math.min(CONFIG.batchSize, train.count - start);
    for (let s = 0; s < n; s++) {
      const idx = order[start + s]!;
      batchInput.set(train.images.subarray(idx * inputSize, (idx + 1) * inputSize), s * inputSize);
      batchLabels[s] = train.labels[idx]!;
    }
    const input = batchInput.subarray(0, n * inputSize);

    const pass = net.forward(input, n, true);
    lossSum += net.loss(pass.acts[pass.acts.length - 1]!, batchLabels, n);
    net.backward(pass, batchLabels, n);
    net.step(CONFIG.learningRate);
    batches++;

    if (batches % 50 === 0) {
      const pct = (((start + n) / train.count) * 100).toFixed(0);
      process.stdout.write(`\rEpoch ${epoch}/${CONFIG.epochs}  ${pct}%  loss ${(lossSum / batches).toFixed(4)}`);
    }
  }

  const valAcc = accuracy(net, validation);
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  console.log(
    `\rEpoch ${epoch}/${CONFIG.epochs}  loss ${(lossSum / batches).toFixed(4)}  ` +
      `val acc ${(valAcc * 100).toFixed(2)}%  (${seconds}s)`,
  );

  // Keep the checkpoint that does best on the validation set.
  if (valAcc > bestValAcc) {
    bestValAcc = valAcc;
    await Bun.write(CONFIG.modelPath, JSON.stringify(net));
  }
}

const best = MLP.fromJSON((await Bun.file(CONFIG.modelPath).json()) as MLPJSON);
console.log(`\nBest validation accuracy: ${(bestValAcc * 100).toFixed(2)}%`);
console.log(`Test accuracy: ${(accuracy(best, test) * 100).toFixed(2)}%`);
console.log(`Model saved to ${CONFIG.modelPath}`);
