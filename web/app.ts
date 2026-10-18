// Drawing pad + image upload -> MNIST-style 28x28 input -> /api/predict.

import { MNIST_SIZE, toMnist } from "../src/preprocess.ts";

const MAX_SOURCE = 400; // uploaded images are downscaled to this before processing
const STROKE_WIDTH = 22;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const pad = $<HTMLCanvasElement>("pad");
const preview = $<HTMLCanvasElement>("preview");
const drop = $<HTMLDivElement>("drop");
const digitEl = $<HTMLDivElement>("digit");
const confidenceEl = $<HTMLDivElement>("confidence");
const barsEl = $<HTMLOListElement>("bars");
const padCtx = pad.getContext("2d", { willReadFrequently: true })!;
const previewCtx = preview.getContext("2d")!;

// Image currently being classified: the pad itself, or an uploaded image.
let uploaded: HTMLCanvasElement | null = null;

// ---------- result bars ----------

const bars = Array.from({ length: 10 }, (_, d) => {
  const li = document.createElement("li");
  li.innerHTML = `<span class="label">${d}</span><span class="track"><span class="fill"></span></span><span class="value"></span>`;
  barsEl.append(li);
  return { li, fill: li.querySelector<HTMLElement>(".fill")!, value: li.querySelector<HTMLElement>(".value")! };
});

function showResult(result: { prediction: number; probabilities: number[] } | null, message?: string) {
  bars.forEach((b, d) => {
    const p = result?.probabilities[d] ?? 0;
    b.fill.style.width = `${p * 100}%`;
    b.value.textContent = result ? `${(p * 100).toFixed(1)}%` : "";
    b.li.classList.toggle("top", result?.prediction === d);
  });
  digitEl.textContent = result ? String(result.prediction) : "–";
  confidenceEl.textContent = result
    ? `${(result.probabilities[result.prediction]! * 100).toFixed(1)}% confident`
    : (message ?? "");
}

// ---------- classification ----------

function grayscale(canvas: HTMLCanvasElement): Float32Array {
  const { width, height } = canvas;
  const data = canvas.getContext("2d")!.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i++) {
    const a = data[i * 4 + 3]! / 255;
    const lum = (0.299 * data[i * 4]! + 0.587 * data[i * 4 + 1]! + 0.114 * data[i * 4 + 2]!) / 255;
    gray[i] = lum * a + (1 - a); // transparent pixels count as white paper
  }
  return gray;
}

function drawPreview(pixels: Float32Array | null) {
  const img = previewCtx.createImageData(MNIST_SIZE, MNIST_SIZE);
  for (let i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
    const v = Math.round((pixels?.[i] ?? 0) * 255);
    img.data.set([v, v, v, 255], i * 4);
  }
  previewCtx.putImageData(img, 0, 0);
}

let pending = false;
let queued = false;

async function classify() {
  if (pending) {
    queued = true;
    return;
  }
  pending = true;
  try {
    const source = uploaded ?? pad;
    const pixels = toMnist(grayscale(source), source.width, source.height);
    drawPreview(pixels);
    if (!pixels) {
      showResult(null, uploaded ? "Couldn't find a digit in that image" : "Draw something to start");
      return;
    }
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: Array.from(pixels) }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showResult(await res.json());
  } catch (err) {
    showResult(null, `Prediction failed: ${(err as Error).message}`);
  } finally {
    pending = false;
    if (queued) {
      queued = false;
      classify();
    }
  }
}

// ---------- drawing ----------

function clearPad() {
  uploaded = null;
  padCtx.fillStyle = "#000";
  padCtx.fillRect(0, 0, pad.width, pad.height);
}

function padPoint(e: PointerEvent): [number, number] {
  const rect = pad.getBoundingClientRect();
  return [((e.clientX - rect.left) / rect.width) * pad.width, ((e.clientY - rect.top) / rect.height) * pad.height];
}

let last: [number, number] | null = null;

pad.addEventListener("pointerdown", (e) => {
  if (uploaded) clearPad(); // start a fresh drawing over an uploaded image
  pad.setPointerCapture(e.pointerId);
  last = padPoint(e);
  stroke(last, last);
});

pad.addEventListener("pointermove", (e) => {
  if (!last) return;
  const p = padPoint(e);
  stroke(last, p);
  last = p;
  classify();
});

const endStroke = () => {
  if (!last) return;
  last = null;
  classify();
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

function stroke(from: [number, number], to: [number, number]) {
  padCtx.strokeStyle = "#fff";
  padCtx.lineWidth = STROKE_WIDTH;
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.moveTo(...from);
  padCtx.lineTo(...to);
  padCtx.stroke();
}

$<HTMLButtonElement>("clear").addEventListener("click", () => {
  clearPad();
  classify();
});

// ---------- uploads ----------

async function loadImage(file: File) {
  if (!file.type.startsWith("image/")) return;
  const bitmap = await createImageBitmap(file);

  // Keep a downscaled copy at the image's own aspect ratio for processing.
  const scale = Math.min(1, MAX_SOURCE / Math.max(bitmap.width, bitmap.height));
  const source = document.createElement("canvas");
  source.width = Math.max(1, Math.round(bitmap.width * scale));
  source.height = Math.max(1, Math.round(bitmap.height * scale));
  const sctx = source.getContext("2d", { willReadFrequently: true })!;
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(bitmap, 0, 0, source.width, source.height);
  bitmap.close();

  // Show it on the pad, letterboxed.
  clearPad();
  const fit = Math.min(pad.width / source.width, pad.height / source.height);
  const w = source.width * fit;
  const h = source.height * fit;
  padCtx.drawImage(source, (pad.width - w) / 2, (pad.height - h) / 2, w, h);

  uploaded = source;
  classify();
}

$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) loadImage(file);
  input.value = "";
});

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("dragging");
});
drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragging");
  const file = e.dataTransfer?.files[0];
  if (file) loadImage(file);
});

// Paste an image from the clipboard.
window.addEventListener("paste", (e) => {
  const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
  if (file) loadImage(file);
});

clearPad();
drawPreview(null);
showResult(null, "Draw something to start");
