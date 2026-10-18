// Fully connected network: ReLU hidden layers, softmax output,
// cross-entropy loss, inverted dropout and the Adam optimizer.
//
// Weights for layer l are a flat row-major Float32Array of shape [out, in].
// Activations are batched: a Float32Array of shape [batch, size].

import { createRng, type Rng } from "./random.ts";

export interface MLPOptions {
  dropout?: number;
  seed?: number;
}

export interface ForwardPass {
  /** Activations per layer; acts[0] is the input, the last is the softmax output. */
  acts: Float32Array[];
  /** Dropout masks, indexed like `acts` (only set for hidden layers while training). */
  masks: (Float32Array | undefined)[];
}

export interface MLPJSON {
  sizes: number[];
  dropout: number;
  weights: number[][];
  biases: number[][];
}

interface AdamState {
  t: number;
  mW: Float32Array[];
  vW: Float32Array[];
  mB: Float32Array[];
  vB: Float32Array[];
}

export class MLP {
  readonly sizes: number[];
  readonly dropout: number;
  weights: Float32Array[] = [];
  biases: Float32Array[] = [];
  private readonly gradW: Float32Array[];
  private readonly gradB: Float32Array[];
  private readonly rng: Rng;
  private adam: AdamState | null = null;

  constructor(sizes: number[], { dropout = 0, seed = 42 }: MLPOptions = {}) {
    this.sizes = sizes;
    this.dropout = dropout;
    this.rng = createRng(seed);

    for (let l = 1; l < sizes.length; l++) {
      const fanIn = sizes[l - 1]!;
      const fanOut = sizes[l]!;
      // He initialization, suited to ReLU
      const std = Math.sqrt(2 / fanIn);
      const W = new Float32Array(fanOut * fanIn);
      for (let k = 0; k < W.length; k++) W[k] = this.rng.gaussian() * std;
      this.weights.push(W);
      this.biases.push(new Float32Array(fanOut));
    }

    this.gradW = this.weights.map((W) => new Float32Array(W.length));
    this.gradB = this.biases.map((b) => new Float32Array(b.length));
  }

  get layerCount(): number {
    return this.weights.length;
  }

  get outputSize(): number {
    return this.sizes[this.sizes.length - 1]!;
  }

  // Returns every layer's activations (acts[0] is the input) and the dropout masks.
  forward(input: Float32Array, batchSize: number, training = false): ForwardPass {
    const acts: Float32Array[] = [input];
    const masks: (Float32Array | undefined)[] = [];
    const useDropout = training && this.dropout > 0;

    for (let l = 0; l < this.layerCount; l++) {
      const inSize = this.sizes[l]!;
      const outSize = this.sizes[l + 1]!;
      const W = this.weights[l]!;
      const b = this.biases[l]!;
      const prev = acts[l]!;
      const out = new Float32Array(batchSize * outSize);
      const isOutput = l === this.layerCount - 1;

      for (let s = 0; s < batchSize; s++) {
        const xo = s * inSize;
        const oo = s * outSize;
        for (let j = 0; j < outSize; j++) {
          const wo = j * inSize;
          let sum = b[j]!;
          for (let i = 0; i < inSize; i++) sum += W[wo + i]! * prev[xo + i]!;
          out[oo + j] = isOutput || sum > 0 ? sum : 0;
        }
        if (isOutput) softmax(out, oo, outSize);
      }

      if (!isOutput && useDropout) {
        const keep = 1 - this.dropout;
        const mask = new Float32Array(out.length);
        for (let k = 0; k < out.length; k++) {
          mask[k] = this.rng.next() < keep ? 1 / keep : 0;
          out[k]! *= mask[k]!;
        }
        masks[l + 1] = mask;
      }

      acts.push(out);
    }

    return { acts, masks };
  }

  // Mean cross-entropy loss of the output probabilities against the labels.
  loss(probs: Float32Array, labels: Uint8Array, batchSize: number): number {
    const classes = this.outputSize;
    let total = 0;
    for (let s = 0; s < batchSize; s++) {
      total -= Math.log(Math.max(probs[s * classes + labels[s]!]!, 1e-12));
    }
    return total / batchSize;
  }

  // Fills gradW / gradB with the batch-averaged gradients.
  backward({ acts, masks }: ForwardPass, labels: Uint8Array, batchSize: number): void {
    const L = this.layerCount;
    const classes = this.outputSize;

    // softmax + cross-entropy gradient: (p - y) / batch
    let delta = new Float32Array(acts[L]!);
    for (let s = 0; s < batchSize; s++) delta[s * classes + labels[s]!]! -= 1;
    for (let k = 0; k < delta.length; k++) delta[k]! /= batchSize;

    for (let l = L - 1; l >= 0; l--) {
      const inSize = this.sizes[l]!;
      const outSize = this.sizes[l + 1]!;
      const W = this.weights[l]!;
      const gW = this.gradW[l]!.fill(0);
      const gB = this.gradB[l]!.fill(0);
      const prev = acts[l]!;
      const prevDelta = l > 0 ? new Float32Array(batchSize * inSize) : null;

      for (let s = 0; s < batchSize; s++) {
        const xo = s * inSize;
        const oo = s * outSize;
        for (let j = 0; j < outSize; j++) {
          const d = delta[oo + j]!;
          if (d === 0) continue;
          gB[j]! += d;
          const wo = j * inSize;
          for (let i = 0; i < inSize; i++) gW[wo + i]! += d * prev[xo + i]!;
          if (prevDelta) {
            for (let i = 0; i < inSize; i++) prevDelta[xo + i]! += W[wo + i]! * d;
          }
        }
      }

      if (!prevDelta) break;

      // ReLU derivative, plus the dropout scaling applied in the forward pass.
      // A dropped unit has prev === 0, so it is zeroed by the ReLU check.
      const mask = masks[l];
      for (let k = 0; k < prevDelta.length; k++) {
        prevDelta[k] = prev[k]! > 0 ? prevDelta[k]! * (mask ? mask[k]! : 1) : 0;
      }
      delta = prevDelta;
    }
  }

  // Applies one Adam step using the gradients from the last backward pass.
  step(lr: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8): void {
    if (!this.adam) {
      const zeros = (arrays: Float32Array[]) => arrays.map((a) => new Float32Array(a.length));
      this.adam = {
        t: 0,
        mW: zeros(this.weights),
        vW: zeros(this.weights),
        mB: zeros(this.biases),
        vB: zeros(this.biases),
      };
    }
    const a = this.adam;
    a.t++;
    const c1 = 1 - beta1 ** a.t;
    const c2 = 1 - beta2 ** a.t;

    const update = (params: Float32Array, grads: Float32Array, m: Float32Array, v: Float32Array) => {
      for (let k = 0; k < params.length; k++) {
        const g = grads[k]!;
        m[k] = beta1 * m[k]! + (1 - beta1) * g;
        v[k] = beta2 * v[k]! + (1 - beta2) * g * g;
        params[k]! -= (lr * (m[k]! / c1)) / (Math.sqrt(v[k]! / c2) + eps);
      }
    };

    for (let l = 0; l < this.layerCount; l++) {
      update(this.weights[l]!, this.gradW[l]!, a.mW[l]!, a.vW[l]!);
      update(this.biases[l]!, this.gradB[l]!, a.mB[l]!, a.vB[l]!);
    }
  }

  // Returns the predicted class for each sample in the batch.
  predict(input: Float32Array, batchSize: number): Uint8Array {
    const { acts } = this.forward(input, batchSize, false);
    const probs = acts[acts.length - 1]!;
    const classes = this.outputSize;
    const predictions = new Uint8Array(batchSize);
    for (let s = 0; s < batchSize; s++) {
      predictions[s] = argmax(probs, s * classes, classes);
    }
    return predictions;
  }

  toJSON(): MLPJSON {
    return {
      sizes: this.sizes,
      dropout: this.dropout,
      weights: this.weights.map((W) => Array.from(W)),
      biases: this.biases.map((b) => Array.from(b)),
    };
  }

  static fromJSON(json: MLPJSON): MLP {
    const net = new MLP(json.sizes, { dropout: json.dropout });
    net.weights = json.weights.map((W) => Float32Array.from(W));
    net.biases = json.biases.map((b) => Float32Array.from(b));
    return net;
  }
}

function softmax(values: Float32Array, offset: number, length: number): void {
  let max = -Infinity;
  for (let i = 0; i < length; i++) max = Math.max(max, values[offset + i]!);
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const e = Math.exp(values[offset + i]! - max);
    values[offset + i] = e;
    sum += e;
  }
  for (let i = 0; i < length; i++) values[offset + i]! /= sum;
}

function argmax(values: Float32Array, offset: number, length: number): number {
  let best = 0;
  for (let i = 1; i < length; i++) {
    if (values[offset + i]! > values[offset + best]!) best = i;
  }
  return best;
}
