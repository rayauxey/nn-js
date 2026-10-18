// Builds a multi-page visual explanation of the trained network from
// model.json + history.json, then prints it to nn-visualization.pdf
// with headless Chrome.

import { deflateSync } from "node:zlib";
import { loadMnist, type Dataset } from "./mnist.ts";
import { MLP, type MLPJSON } from "./mlp.ts";

const MODEL_PATH = "model.json";
const HISTORY_PATH = "history.json";
const HTML_PATH = "nn-visualization.html";
const PDF_PATH = "nn-visualization.pdf";
const SAMPLE_INDEX = 0; // test image traced through the network

// ---------- palette (light, print) ----------

const C = {
  surface: "#fcfcfb",
  page: "#ffffff",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  series: "#2a78d6",
  critical: "#d03b3b",
  good: "#006300",
  neutral: "#f0efec",
};

// Sequential blue ramp, 100 -> 700.
const BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
// Diverging arms from the neutral midpoint out to each pole.
const DIV_POS = ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"];
const DIV_NEG = ["#f0efec", "#f2b2a8", "#e06b5f", "#a8302e"];

type RGB = [number, number, number];

function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]: RGB): string {
  return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
}

function ramp(stops: string[], t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const a = hexToRgb(stops[i]!);
  const b = hexToRgb(stops[i + 1]!);
  const f = x - i;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const sequential = (t: number) => ramp([C.surface, ...BLUE], t);
const diverging = (t: number) => (t >= 0 ? ramp(DIV_POS, t) : ramp(DIV_NEG, -t));
const ink = (t: number) => ramp([C.surface, C.ink], t);

// ---------- tiny PNG encoder (for pixel images) ----------

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]!) & 255]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  out.set(new TextEncoder().encode(type), 4);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

// Encodes a w*h grid of colors as a PNG data URL, upscaled by `scale` (nearest neighbour).
function pngDataUrl(w: number, h: number, colorAt: (x: number, y: number) => RGB, scale = 8): string {
  const W = w * scale;
  const H = h * scale;
  const raw = new Uint8Array(H * (1 + W * 3));
  for (let y = 0; y < H; y++) {
    const row = y * (1 + W * 3);
    for (let x = 0; x < W; x++) {
      const [r, g, b] = colorAt(Math.floor(x / scale), Math.floor(y / scale));
      raw[row + 1 + x * 3] = r;
      raw[row + 2 + x * 3] = g;
      raw[row + 3 + x * 3] = b;
    }
  }
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, W);
  v.setUint32(4, H);
  ihdr.set([8, 2, 0, 0, 0], 8); // 8-bit RGB
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", new Uint8Array(deflateSync(raw))),
    chunk("IEND", new Uint8Array()),
  ];
  return "data:image/png;base64," + Buffer.concat(parts).toString("base64");
}

const digitPng = (pixels: Float32Array, scale = 8) =>
  pngDataUrl(28, 28, (x, y) => ink(pixels[y * 28 + x]!), scale);

// ---------- data ----------

const net = MLP.fromJSON((await Bun.file(MODEL_PATH).json()) as MLPJSON);
const history = (await Bun.file(HISTORY_PATH).json()) as { epoch: number; loss: number; valAcc: number }[];
const { test } = await loadMnist();

const image = (set: Dataset, i: number) => set.images.subarray(i * set.inputSize, (i + 1) * set.inputSize);

function probabilities(i: number): Float32Array {
  const { acts } = net.forward(image(test, i), 1);
  return acts[acts.length - 1]!;
}

const argmax = (a: ArrayLike<number>) => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i]! > a[best]!) best = i;
  return best;
};

// Full test-set predictions + confusion matrix.
const predictions = new Uint8Array(test.count);
for (let start = 0; start < test.count; start += 1000) {
  const n = Math.min(1000, test.count - start);
  predictions.set(net.predict(test.images.subarray(start * 784, (start + n) * 784), n), start);
}
const confusion = Array.from({ length: 10 }, () => new Array<number>(10).fill(0));
let correct = 0;
for (let i = 0; i < test.count; i++) {
  confusion[test.labels[i]!]![predictions[i]!]!++;
  if (predictions[i] === test.labels[i]) correct++;
}
const testAcc = correct / test.count;

const params = net.weights.reduce((s, W) => s + W.length, 0) + net.biases.reduce((s, b) => s + b.length, 0);
const pct = (x: number, d = 2) => `${(x * 100).toFixed(d)}%`;
const fmt = (n: number) => n.toLocaleString("en-US");

// Sample forward pass.
const sample = image(test, SAMPLE_INDEX);
const sampleActs = net.forward(sample, 1).acts;
const sampleLabel = test.labels[SAMPLE_INDEX]!;
const sampleProbs = sampleActs[3]!;
const samplePred = argmax(sampleProbs);

// ---------- page 1: architecture ----------

function architectureSvg(): string {
  const W = 1040;
  const H = 520;
  const top = 40;
  const bottom = 420;
  const shown = 12;
  const r = 9;
  const cols = [330, 525, 710, 865];
  const sizes = net.sizes;

  // Pick which units to draw: brightest pixels for the input, most active units for hidden layers.
  const pick = (values: Float32Array, k: number) =>
    Array.from(values, (v, i) => [v, i] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, k)
      .map(([, i]) => i)
      .sort((a, b) => a - b);
  const inputIdx = (() => {
    const bright = Array.from(sample, (v, i) => [v, i] as const).filter(([v]) => v > 0.6).map(([, i]) => i);
    return Array.from({ length: shown }, (_, k) => bright[Math.floor((k * (bright.length - 1)) / (shown - 1))]!);
  })();
  const layers: number[][] = [inputIdx, pick(sampleActs[1]!, shown), pick(sampleActs[2]!, shown), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
  const ys = (n: number) => Array.from({ length: n }, (_, k) => top + 10 + (k * (bottom - top - 20)) / (n - 1));

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;

  // Digit image with the sampled pixels marked.
  const imgX = 20;
  const imgY = 110;
  const imgS = 196;
  const cell = imgS / 28;
  svg += `<image href="${digitPng(sample)}" x="${imgX}" y="${imgY}" width="${imgS}" height="${imgS}"/>`;
  svg += `<rect x="${imgX}" y="${imgY}" width="${imgS}" height="${imgS}" fill="none" stroke="${C.axis}"/>`;
  svg += `<text x="${imgX + imgS / 2}" y="${imgY - 14}" text-anchor="middle" font-size="13" fill="${C.ink2}">Test image #${SAMPLE_INDEX} (28 × 28)</text>`;
  svg += `<text x="${imgX + imgS / 2}" y="${imgY + imgS + 24}" text-anchor="middle" font-size="12" fill="${C.muted}">flattened row by row into</text>`;
  svg += `<text x="${imgX + imgS / 2}" y="${imgY + imgS + 40}" text-anchor="middle" font-size="12" fill="${C.muted}">784 numbers between 0 and 1</text>`;

  const inYs = ys(shown);
  inputIdx.forEach((p, k) => {
    const px = imgX + (p % 28) * cell + cell / 2;
    const py = imgY + Math.floor(p / 28) * cell + cell / 2;
    svg += `<line x1="${px}" y1="${py}" x2="${cols[0]! - r}" y2="${inYs[k]}" stroke="${C.axis}" stroke-width="0.8"/>`;
    svg += `<rect x="${px - cell / 2}" y="${py - cell / 2}" width="${cell}" height="${cell}" fill="none" stroke="${C.series}" stroke-width="1.5"/>`;
  });

  // Connections, coloured by weight sign, opacity by magnitude.
  for (let l = 0; l < 3; l++) {
    const W_ = net.weights[l]!;
    const inSize = sizes[l]!;
    const from = layers[l]!;
    const to = layers[l + 1]!;
    const fy = ys(from.length);
    const ty = ys(to.length);
    let max = 0;
    for (const j of to) for (const i of from) max = Math.max(max, Math.abs(W_[j * inSize + i]!));
    to.forEach((j, b) =>
      from.forEach((i, a) => {
        const w = W_[j * inSize + i]!;
        const o = (Math.abs(w) / max) ** 1.2 * 0.85;
        if (o < 0.04) return;
        const color = w >= 0 ? DIV_POS[3] : DIV_NEG[3];
        svg += `<line x1="${cols[l]! + r}" y1="${fy[a]}" x2="${cols[l + 1]! - r}" y2="${ty[b]}" stroke="${color}" stroke-opacity="${o.toFixed(3)}" stroke-width="1"/>`;
      }),
    );
  }

  // Nodes, filled by activation (normalised per layer).
  layers.forEach((idx, l) => {
    const values = sampleActs[l]!;
    let max = 0;
    for (const i of idx) max = Math.max(max, values[i]!);
    const y = ys(idx.length);
    idx.forEach((i, k) => {
      const fill = rgbToHex(sequential(max > 0 ? values[i]! / max : 0));
      svg += `<circle cx="${cols[l]}" cy="${y[k]}" r="${r}" fill="${fill}" stroke="${C.ink2}" stroke-width="1"/>`;
      if (l === 3) svg += `<text x="${cols[l]! - 18}" y="${y[k]! + 4}" text-anchor="end" font-size="12" fill="${C.ink2}">${i}</text>`;
    });
  });

  // Output probability bars.
  const outYs = ys(10);
  const barX = cols[3]! + 20;
  const barW = 100;
  for (let d = 0; d < 10; d++) {
    const p = sampleProbs[d]!;
    const w = Math.max(p * barW, 1.5);
    const isPred = d === samplePred;
    svg += `<rect x="${barX}" y="${outYs[d]! - 5}" width="${barW}" height="10" rx="2" fill="${C.neutral}"/>`;
    svg += `<rect x="${barX}" y="${outYs[d]! - 5}" width="${w}" height="10" rx="2" fill="${isPred ? C.series : BLUE[1]}"/>`;
    if (isPred || p > 0.01) {
      svg += `<text x="${barX + barW + 6}" y="${outYs[d]! + 4}" font-size="11" fill="${C.ink}" font-weight="${isPred ? 600 : 400}">${pct(p, 1)}</text>`;
    }
  }

  // Column captions.
  const captions = [
    ["Input layer", "784 pixels", `showing ${shown} bright pixels`],
    ["Hidden layer 1", "256 neurons · ReLU", `showing ${shown} most active`],
    ["Hidden layer 2", "128 neurons · ReLU", `showing ${shown} most active`],
    ["Output layer", "10 neurons · softmax", "one per digit"],
  ];
  captions.forEach(([a, b, c], l) => {
    svg += `<text x="${cols[l]}" y="${bottom + 32}" text-anchor="middle" font-size="13" font-weight="600" fill="${C.ink}">${a}</text>`;
    svg += `<text x="${cols[l]}" y="${bottom + 50}" text-anchor="middle" font-size="12" fill="${C.ink2}">${b}</text>`;
    svg += `<text x="${cols[l]}" y="${bottom + 66}" text-anchor="middle" font-size="11" fill="${C.muted}">${c}</text>`;
  });

  // Weight matrix labels between columns.
  const mats = ["W1 · 256×784", "W2 · 128×256", "W3 · 10×128"];
  const counts = [784 * 256 + 256, 256 * 128 + 128, 128 * 10 + 10];
  mats.forEach((m, l) => {
    const x = (cols[l]! + cols[l + 1]!) / 2;
    svg += `<text x="${x}" y="${top - 18}" text-anchor="middle" font-size="12" fill="${C.ink2}">${m}</text>`;
    svg += `<text x="${x}" y="${top - 3}" text-anchor="middle" font-size="11" fill="${C.muted}">${fmt(counts[l]!)} params</text>`;
  });

  svg += `<text x="${barX + barW / 2}" y="${top - 10}" text-anchor="middle" font-size="12" fill="${C.ink2}">P(digit)</text>`;
  return svg + "</svg>";
}

// ---------- page 2: forward pass + training loop ----------

function activationGrid(values: Float32Array, cols: number, cellSize: number): string {
  const rows = Math.ceil(values.length / cols);
  let max = 0;
  for (const v of values) max = Math.max(max, v);
  const gap = 2;
  const w = cols * (cellSize + gap) - gap;
  const h = rows * (cellSize + gap) - gap;
  let svg = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">`;
  values.forEach((v, i) => {
    const x = (i % cols) * (cellSize + gap);
    const y = Math.floor(i / cols) * (cellSize + gap);
    const fill = v > 0 ? rgbToHex(sequential(0.15 + 0.85 * (v / max))) : C.neutral;
    svg += `<rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" rx="1.5" fill="${fill}"/>`;
  });
  return svg + "</svg>";
}

const countActive = (a: Float32Array) => a.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);

function trainingLoopSvg(): string {
  const boxes = [
    { x: 20, y: 20, title: "1 · Forward pass", text: ["64 images in,", "probabilities out"] },
    { x: 260, y: 20, title: "2 · Loss", text: ["cross-entropy:", "−log P(correct digit)"] },
    { x: 260, y: 150, title: "3 · Backpropagation", text: ["chain rule gives", "∂loss/∂w for every weight"] },
    { x: 20, y: 150, title: "4 · Adam update", text: ["nudge each weight", "against its gradient"] },
  ];
  const bw = 200;
  const bh = 90;
  let svg = `<svg viewBox="0 0 480 270" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;
  svg += `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${C.ink2}"/></marker></defs>`;
  for (const b of boxes) {
    svg += `<rect x="${b.x}" y="${b.y}" width="${bw}" height="${bh}" rx="8" fill="${C.surface}" stroke="${C.axis}"/>`;
    svg += `<text x="${b.x + 14}" y="${b.y + 28}" font-size="14" font-weight="600" fill="${C.ink}">${b.title}</text>`;
    b.text.forEach((t, i) => {
      svg += `<text x="${b.x + 14}" y="${b.y + 52 + i * 18}" font-size="12.5" fill="${C.ink2}">${t}</text>`;
    });
  }
  const arrow = (x1: number, y1: number, x2: number, y2: number) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${C.ink2}" stroke-width="1.5" marker-end="url(#arrow)"/>`;
  svg += arrow(222, 65, 256, 65);
  svg += arrow(360, 112, 360, 146);
  svg += arrow(258, 195, 224, 195);
  svg += arrow(120, 148, 120, 114);
  svg += `<text x="240" y="262" text-anchor="middle" font-size="12" fill="${C.muted}">860 batches per epoch × ${history.length} epochs ≈ ${fmt(860 * history.length)} updates</text>`;
  return svg + "</svg>";
}

// ---------- page 3: learned features ----------

function featureMaps(count: number): string {
  const W1 = net.weights[0]!;
  // Units whose weights vary the most tend to have the most visible structure.
  const scored = Array.from({ length: 256 }, (_, j) => {
    let s = 0;
    for (let i = 0; i < 784; i++) s += W1[j * 784 + i]! ** 2;
    return [s, j] as const;
  })
    .sort((a, b) => b[0] - a[0])
    .slice(0, count)
    .map(([, j]) => j)
    .sort((a, b) => a - b);

  return scored
    .map((j) => {
      let max = 0;
      for (let i = 0; i < 784; i++) max = Math.max(max, Math.abs(W1[j * 784 + i]!));
      const src = pngDataUrl(28, 28, (x, y) => diverging(W1[j * 784 + y * 28 + x]! / max), 4);
      return `<figure class="feat"><img src="${src}" alt=""/><figcaption>#${j}</figcaption></figure>`;
    })
    .join("");
}

function divergingLegend(): string {
  const stops = Array.from({ length: 21 }, (_, k) => {
    const t = -1 + k / 10;
    return `<stop offset="${k / 20}" stop-color="${rgbToHex(diverging(t))}"/>`;
  }).join("");
  return `<svg viewBox="0 0 320 44" width="320" height="44" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">
    <defs><linearGradient id="div">${stops}</linearGradient></defs>
    <rect x="0" y="4" width="320" height="12" rx="2" fill="url(#div)"/>
    <text x="0" y="34" font-size="11" fill="${C.ink2}">negative weight (inhibits)</text>
    <text x="160" y="34" font-size="11" fill="${C.muted}" text-anchor="middle">0</text>
    <text x="320" y="34" font-size="11" fill="${C.ink2}" text-anchor="end">positive (excites)</text>
  </svg>`;
}

// ---------- page 4: predictions + confusion ----------

function predictionCards(): string {
  const wrong: number[] = [];
  const right: number[] = [];
  for (let i = 0; i < test.count && (wrong.length < 4 || right.length < 12); i++) {
    if (predictions[i] === test.labels[i]) {
      if (right.length < 12 && i % 37 === 0) right.push(i);
    } else if (wrong.length < 4) wrong.push(i);
  }
  return [...right, ...wrong]
    .map((i) => {
      const probs = probabilities(i);
      const pred = predictions[i]!;
      const label = test.labels[i]!;
      const ok = pred === label;
      // Mini vertical bar chart: one bar per digit, height = probability.
      const bw = 6;
      const gap = 2;
      const plotH = 40;
      let bars = `<svg class="bars" viewBox="0 0 ${10 * (bw + gap) - gap} ${plotH + 12}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;
      bars += `<line x1="0" x2="${10 * (bw + gap) - gap}" y1="${plotH + 0.5}" y2="${plotH + 0.5}" stroke="${C.axis}" stroke-width="1"/>`;
      probs.forEach((p, d) => {
        const h = Math.max(p * plotH, 1);
        const x = d * (bw + gap);
        const fill = d === pred ? (ok ? C.series : C.critical) : BLUE[2];
        bars += `<rect x="${x}" y="${plotH - h}" width="${bw}" height="${h}" rx="1.5" fill="${fill}"/>`;
        bars += `<text x="${x + bw / 2}" y="${plotH + 10}" text-anchor="middle" font-size="7.5" fill="${d === pred ? C.ink : C.muted}">${d}</text>`;
      });
      bars += "</svg>";
      const verdict = ok
        ? `<b>${pred}</b> · ${pct(probs[pred]!, 1)}`
        : `<b class="bad">✕ ${pred}</b> · true ${label}`;
      return `<div class="card${ok ? "" : " wrong"}"><img src="${digitPng(image(test, i), 3)}" alt=""/>${bars}<div class="verdict">${verdict}</div></div>`;
    })
    .join("");
}

function confusionSvg(): string {
  const cell = 34;
  const left = 46;
  const topPad = 40;
  let maxOff = 0;
  for (let a = 0; a < 10; a++) for (let b = 0; b < 10; b++) if (a !== b) maxOff = Math.max(maxOff, confusion[a]![b]!);
  const size = left + cell * 10 + 10;
  let svg = `<svg viewBox="0 0 ${size} ${topPad + cell * 10 + 10}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;
  svg += `<text x="${left + cell * 5}" y="14" text-anchor="middle" font-size="12" fill="${C.ink2}">Predicted digit</text>`;
  svg += `<text transform="translate(12 ${topPad + cell * 5}) rotate(-90)" text-anchor="middle" font-size="12" fill="${C.ink2}">True digit</text>`;
  for (let d = 0; d < 10; d++) {
    svg += `<text x="${left + d * cell + cell / 2}" y="${topPad - 8}" text-anchor="middle" font-size="11" fill="${C.muted}">${d}</text>`;
    svg += `<text x="${left - 8}" y="${topPad + d * cell + cell / 2 + 4}" text-anchor="end" font-size="11" fill="${C.muted}">${d}</text>`;
  }
  for (let a = 0; a < 10; a++) {
    for (let b = 0; b < 10; b++) {
      const n = confusion[a]![b]!;
      const x = left + b * cell;
      const y = topPad + a * cell;
      const diag = a === b;
      const t = diag ? 0 : n / maxOff;
      const fill = diag ? C.neutral : n === 0 ? C.surface : rgbToHex(sequential(0.12 + 0.88 * t));
      svg += `<rect x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="2" fill="${fill}"/>`;
      if (n > 0) {
        const color = !diag && t > 0.5 ? "#ffffff" : diag ? C.muted : C.ink;
        svg += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle" font-size="${diag ? 9.5 : 11}" fill="${color}">${n}</text>`;
      }
    }
  }
  return svg + "</svg>";
}

// ---------- page 5: training curves ----------

function lineChart(
  values: number[],
  { yMin, yMax, ticks, format, highlight }: { yMin: number; yMax: number; ticks: number[]; format: (v: number) => string; highlight: number },
): string {
  const W = 440;
  const H = 300;
  const m = { l: 52, r: 16, t: 16, b: 36 };
  const x = (i: number) => m.l + (i / (values.length - 1)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;
  for (const t of ticks) {
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="${t === yMin ? C.axis : C.grid}" stroke-width="1"/>`;
    svg += `<text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end" font-size="11" fill="${C.muted}" style="font-variant-numeric:tabular-nums">${format(t)}</text>`;
  }
  values.forEach((_, i) => {
    if ((i + 1) % 2 === 1 || i === values.length - 1) {
      svg += `<text x="${x(i)}" y="${H - m.b + 18}" text-anchor="middle" font-size="11" fill="${C.muted}">${i + 1}</text>`;
    }
  });
  svg += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="${C.ink2}">Epoch</text>`;
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  svg += `<path d="${path}" fill="none" stroke="${C.series}" stroke-width="2" stroke-linejoin="round"/>`;
  values.forEach((v, i) => {
    const hi = i === highlight;
    svg += `<circle cx="${x(i)}" cy="${y(v)}" r="${hi ? 5 : 3}" fill="${hi ? C.series : C.surface}" stroke="${hi ? C.surface : C.series}" stroke-width="2"/>`;
  });
  const hv = values[highlight]!;
  const anchor = highlight > values.length / 2 ? "end" : "start";
  const dx = anchor === "end" ? -10 : 10;
  svg += `<text x="${x(highlight) + dx}" y="${y(hv) - 10}" text-anchor="${anchor}" font-size="12" font-weight="600" fill="${C.ink}">${format(hv)}</text>`;
  return svg + "</svg>";
}

const losses = history.map((h) => h.loss);
const accs = history.map((h) => h.valAcc);
const bestEpoch = argmax(accs);
const accFloor = Math.floor(Math.min(...accs) * 100) / 100;

// ---------- assemble ----------

const eq = (a: string, b: string, c: string) => `<tr><td class="m">${a}</td><td>${b}</td><td class="shape">${c}</td></tr>`;

const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>How the MNIST network works</title>
<style>
  .card .bars { width: 100%; height: auto; display: block; }
  @page { size: A4 landscape; margin: 0; }
  * { box-sizing: border-box; }
  body { margin: 0; background: ${C.page}; color: ${C.ink}; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .page { width: 297mm; height: 210mm; padding: 13mm 15mm 11mm; page-break-after: always; display: flex; flex-direction: column; overflow: hidden; }
  .page:last-child { page-break-after: auto; }
  .kicker { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: ${C.muted}; margin: 0 0 4px; }
  h1 { font-size: 26px; margin: 0 0 6px; font-weight: 650; }
  h2 { font-size: 21px; margin: 0 0 6px; font-weight: 650; }
  h3 { font-size: 14px; margin: 0 0 8px; font-weight: 600; }
  .lede { font-size: 13.5px; color: ${C.ink2}; margin: 0 0 14px; max-width: 230mm; line-height: 1.45; }
  .row { display: flex; gap: 12mm; flex: 1; min-height: 0; }
  .col { flex: 1; min-width: 0; }
  .note { font-size: 12px; color: ${C.ink2}; line-height: 1.5; }
  .muted { color: ${C.muted}; }
  .foot { font-size: 10.5px; color: ${C.muted}; margin-top: auto; padding-top: 6px; display: flex; justify-content: space-between; }
  .legend { display: flex; gap: 18px; font-size: 11.5px; color: ${C.ink2}; align-items: center; }
  .sw { display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
  .dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; vertical-align: middle; margin-right: 6px; border: 1px solid ${C.ink2}; }
  table.eq { border-collapse: collapse; font-size: 12.5px; width: 100%; }
  table.eq td { padding: 6px 8px; border-bottom: 1px solid ${C.grid}; vertical-align: top; }
  table.eq td.m { font-family: ui-monospace, "SF Mono", Menlo, monospace; white-space: nowrap; color: ${C.ink}; }
  table.eq td.shape { color: ${C.muted}; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .grids { display: flex; flex-direction: column; gap: 14px; }
  .feats { display: grid; grid-template-columns: repeat(12, 1fr); gap: 7px 8px; }
  .feat { margin: 0; text-align: center; }
  .feat img { width: 100%; image-rendering: pixelated; display: block; border-radius: 2px; }
  .feat figcaption { font-size: 9px; color: ${C.muted}; margin-top: 2px; }
  .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .card { border: 1px solid ${C.grid}; border-radius: 6px; padding: 8px; display: grid; grid-template-columns: 48px 1fr; align-items: end; grid-template-rows: auto auto; gap: 4px 8px; background: ${C.surface}; }
  .card.wrong { border-color: ${C.critical}; }
  .card img { width: 48px; height: 48px; image-rendering: pixelated; border-radius: 2px; }
  .verdict { grid-column: 1 / -1; font-size: 11px; color: ${C.ink2}; }
  .verdict b { color: ${C.ink}; }
  .verdict b.bad { color: ${C.critical}; }
  .tiles { display: flex; gap: 10px; margin-bottom: 14px; }
  .tile { flex: 1; border: 1px solid ${C.grid}; border-radius: 8px; padding: 10px 14px; background: ${C.surface}; }
  .tile .v { font-size: 24px; font-weight: 650; }
  .tile .l { font-size: 11.5px; color: ${C.ink2}; }
</style></head><body>

<section class="page">
  <p class="kicker">nn-js · MNIST digit classifier</p>
  <h1>How the network reads a handwritten digit</h1>
  <p class="lede">A fully connected network turns 784 pixel brightnesses into 10 probabilities. Each layer multiplies the previous layer by a weight matrix, adds a bias, and applies an activation. Below, real test image #${SAMPLE_INDEX} (a ${sampleLabel}) flows through the trained model. It predicts <b>${samplePred}</b> with ${pct(sampleProbs[samplePred]!, 2)} confidence.</p>
  <div style="flex:1;min-height:0">${architectureSvg()}</div>
  <div class="legend">
    <span><i class="sw" style="background:${DIV_POS[3]}"></i>positive weight</span>
    <span><i class="sw" style="background:${DIV_NEG[3]}"></i>negative weight</span>
    <span class="muted">line opacity = |weight|</span>
    <span><i class="dot" style="background:${BLUE[5]}"></i>strongly active</span>
    <span><i class="dot" style="background:${C.surface}"></i>quiet</span>
  </div>
  <div class="foot"><span>${fmt(params)} trainable parameters · ${pct(testAcc)} test accuracy</span><span>1 / 5</span></div>
</section>

<section class="page">
  <p class="kicker">One forward pass</p>
  <h2>Every layer is a matrix multiply plus a simple non-linearity</h2>
  <p class="lede">Hidden layers use ReLU (max(0, z)), so a neuron either stays silent or passes its signal on. For any given digit, only part of the network lights up. The output layer uses softmax to turn 10 scores into probabilities that sum to 1.</p>
  <div class="row">
    <div class="col">
      <h3>The math, with shapes</h3>
      <table class="eq">
        ${eq("x", "pixels / 255", "784")}
        ${eq("h₁ = ReLU(W₁x + b₁)", "first hidden layer", "256")}
        ${eq("h₂ = ReLU(W₂h₁ + b₂)", "second hidden layer", "128")}
        ${eq("p = softmax(W₃h₂ + b₃)", "pᵢ = eᶻⁱ / Σ eᶻʲ", "10")}
        ${eq("prediction = argmax(p)", "most likely digit", "1")}
      </table>
      <h3 style="margin-top:18px">How it learned the weights</h3>
      ${trainingLoopSvg()}
      <p class="note">During training, 20% of hidden neurons are randomly switched off on each step (dropout). This stops the network from relying on any single neuron.</p>
    </div>
    <div class="col">
      <h3>Hidden activity for image #${SAMPLE_INDEX}</h3>
      <div class="grids">
        <div>
          <p class="note" style="margin:0 0 6px">Hidden layer 1: <b>${countActive(sampleActs[1]!)}</b> of 256 neurons fire</p>
          ${activationGrid(sampleActs[1]!, 32, 14)}
        </div>
        <div>
          <p class="note" style="margin:0 0 6px">Hidden layer 2: <b>${countActive(sampleActs[2]!)}</b> of 128 neurons fire</p>
          ${activationGrid(sampleActs[2]!, 32, 14)}
        </div>
        <div class="legend">
          <span><i class="dot" style="background:${C.neutral};border-color:${C.axis};border-radius:2px"></i>silent (ReLU output 0)</span>
          <span><i class="dot" style="background:${BLUE[1]};border:none;border-radius:2px"></i>weak</span>
          <span><i class="dot" style="background:${BLUE[6]};border:none;border-radius:2px"></i>strong</span>
        </div>
        <p class="note">Each square is one neuron, in index order. A different digit lights up a different pattern. The output layer reads these patterns to decide which digit it sees.</p>
      </div>
    </div>
  </div>
  <div class="foot"><span>Shapes are per image. Training runs 64 images at a time.</span><span>2 / 5</span></div>
</section>

<section class="page">
  <p class="kicker">Inside hidden layer 1</p>
  <h2>What the first layer learned to look for</h2>
  <p class="lede">Each neuron in hidden layer 1 has 784 weights, one per pixel, so its weights can be drawn as a 28×28 image. Blue pixels push the neuron to fire when ink is there. Red pixels hold it back. Together they act as stroke and edge detectors. These are the 60 neurons with the largest weights.</p>
  <div class="feats">${featureMaps(60)}</div>
  <div style="margin-top:12px">${divergingLegend()}</div>
  <div class="foot"><span>Each map is scaled to its own largest |weight|.</span><span>3 / 5</span></div>
</section>

<section class="page">
  <p class="kicker">Results on 10,000 unseen test images</p>
  <h2>Predictions, and where it gets confused</h2>
  <p class="lede">The network never saw these images during training. It gets ${fmt(correct)} of ${fmt(test.count)} right (${pct(testAcc)}). Most mistakes are on digits that are genuinely ambiguous or badly written.</p>
  <div class="row">
    <div class="col" style="flex:1.7">
      <h3>Sample predictions: 12 correct, then the first 4 mistakes</h3>
      <div class="cards">${predictionCards()}</div>
    </div>
    <div class="col" style="flex:1">
      <h3>Confusion matrix: off-diagonal cells are errors</h3>
      ${confusionSvg()}
      <p class="note">Diagonal cells (gray) count correct answers. Darker blue means more errors. For example, the cell at row 4, column 9 shows that ${confusion[4]![9]} images of a 4 were read as a 9.</p>
    </div>
  </div>
  <div class="foot"><span>Bars show the softmax probability for each digit. Red marks a wrong prediction.</span><span>4 / 5</span></div>
</section>

<section class="page">
  <p class="kicker">Training run</p>
  <h2>How it improved, epoch by epoch</h2>
  <p class="lede">One epoch is one pass over the 55,000 training images. Loss keeps falling, but validation accuracy levels off after a few epochs. The model saved to disk is the one with the best validation accuracy (epoch ${bestEpoch + 1}).</p>
  <div class="tiles">
    <div class="tile"><div class="v">${pct(testAcc)}</div><div class="l">test accuracy</div></div>
    <div class="tile"><div class="v">${pct(accs[bestEpoch]!)}</div><div class="l">best validation accuracy (epoch ${bestEpoch + 1})</div></div>
    <div class="tile"><div class="v">${fmt(params)}</div><div class="l">trainable parameters</div></div>
    <div class="tile"><div class="v">${history.length}</div><div class="l">epochs · Adam, lr 0.001, batch 64</div></div>
  </div>
  <div class="row">
    <div class="col">
      <h3>Training loss (cross-entropy)</h3>
      ${lineChart(losses, { yMin: 0, yMax: 0.35, ticks: [0, 0.1, 0.2, 0.3], format: (v) => v.toFixed(2), highlight: losses.length - 1 })}
    </div>
    <div class="col">
      <h3>Validation accuracy <span class="muted" style="font-weight:400">(axis starts at ${pct(accFloor, 0)})</span></h3>
      ${lineChart(accs, {
        yMin: accFloor,
        yMax: 0.99,
        ticks: Array.from({ length: Math.round((0.99 - accFloor) * 200) + 1 }, (_, k) => accFloor + k * 0.005),
        format: (v) => pct(v, 1),
        highlight: bestEpoch,
      })}
    </div>
  </div>
  <div class="foot"><span>Validation set: 5,000 images held out from the training data.</span><span>5 / 5</span></div>
</section>

</body></html>`;

await Bun.write(HTML_PATH, html);

const chrome = ["google-chrome", "chromium", "chromium-browser"].find((c) => Bun.which(c));
if (!chrome) throw new Error(`No Chrome/Chromium found; open ${HTML_PATH} and print it to PDF manually.`);
const proc = Bun.spawnSync([
  chrome,
  "--headless",
  "--disable-gpu",
  "--no-pdf-header-footer",
  `--print-to-pdf=${PDF_PATH}`,
  `file://${process.cwd()}/${HTML_PATH}`,
]);
if (proc.exitCode !== 0) throw new Error(`Chrome failed: ${proc.stderr.toString()}`);
console.log(`Wrote ${PDF_PATH}`);
