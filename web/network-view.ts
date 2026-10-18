// Canvas view of the network during a prediction. Each layer is drawn as a
// grid of neurons lit by their activation. When a result arrives the signal
// travels left to right: each layer lights up in turn while pulses run along
// the connections that contributed most to the next layer.

import type { Trace, TraceEdge } from "../src/predict.ts";

const LAYERS = [
  { label: "Input", detail: "784 pixels", size: 784, cols: 28, rows: 28 },
  { label: "Hidden 1", detail: "256 · ReLU", size: 256, cols: 16, rows: 16 },
  { label: "Hidden 2", detail: "128 · ReLU", size: 128, cols: 8, rows: 16 },
  { label: "Output", detail: "10 · softmax", size: 10, cols: 1, rows: 10 },
];
const STAGE_DELAY = 160; // ms between consecutive layers lighting up
const PULSE_MS = 420; // how long a pulse takes to cross one gap
const PULSES_PER_GAP = 28; // pulses drawn on the strongest connections only
const EASE = 0.22; // per-frame approach of displayed activations to their targets
const LABEL_TOP = 30;
const LABEL_BOTTOM = 40;
const OUTPUT_WIDTH = 120; // node + digit label + probability bar
const OUTPUT_WIDTH_NARROW = 44; // node + digit label only
const NARROW = 560; // below this panel width, drop the bars and tighten gaps
const MIN_GAP = 36;
const MIN_GAP_NARROW = 14;
const MAX_GRID = 260;
const MIN_OUTPUT_STEP = 16; // keeps the 10 output labels legible

interface Point {
  x: number;
  y: number;
}

export interface NetworkResult {
  prediction: number;
  probabilities: number[];
  trace: Trace;
}

export class NetworkView {
  private readonly ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private positions: Point[][] = [];
  private columns: { x: number; w: number }[] = [];
  private nodeRadius: number[] = [];
  private cell = 0;
  private showBars = true;
  private gridTop = LABEL_TOP;
  private lastWidth = 0;
  private resizeFrame = 0;

  private current = LAYERS.map((l) => new Float32Array(l.size));
  private target = LAYERS.map((l) => new Float32Array(l.size));
  private layerMax = LAYERS.map(() => 1);
  private edges: TraceEdge[][] = [[], [], []];
  private edgeMax = [1, 1, 1];
  private prediction: number | null = null;
  private startedAt = -Infinity;
  private frame = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    // Defer to the next frame: resizing the canvas changes its container's height.
    new ResizeObserver(() => {
      if (!this.resizeFrame) this.resizeFrame = requestAnimationFrame(() => this.resize());
    }).observe(canvas.parentElement!);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.start());
    this.resize();
  }

  /** Shows a new forward pass: `pixels` is the 784-value network input. */
  update(pixels: Float32Array, result: NetworkResult): void {
    const values = [pixels, ...result.trace.hidden.map((h) => Float32Array.from(h)), Float32Array.from(result.probabilities)];
    values.forEach((v, l) => this.target[l]!.set(v));
    this.layerMax = this.target.map((t, l) => (l === 0 || l === 3 ? 1 : Math.max(1e-6, ...t)));
    this.edges = result.trace.edges;
    this.edgeMax = this.edges.map((e) => Math.max(1e-6, ...e.map(([, , c]) => Math.abs(c))));
    this.prediction = result.prediction;
    this.startedAt = performance.now();
    this.start();
  }

  clear(): void {
    this.target.forEach((t) => t.fill(0));
    this.edges = [[], [], []];
    this.prediction = null;
    this.start();
  }

  private start(): void {
    if (!this.frame) this.frame = requestAnimationFrame(() => this.tick());
  }

  private tick(): void {
    this.frame = 0;
    const settling = this.draw();
    const pulsing = performance.now() < this.startedAt + (LAYERS.length - 1) * STAGE_DELAY + PULSE_MS;
    if (settling || pulsing) this.start();
  }

  private resize(): void {
    this.resizeFrame = 0;
    const width = this.canvas.parentElement!.clientWidth;
    if (!width || width === this.lastWidth) return;
    this.lastWidth = this.width = width;

    // Grid height shared by the hidden layers and the input; shrink it when the
    // panel is narrow. The output column keeps enough height for its labels.
    const narrow = width < NARROW;
    this.showBars = !narrow;
    const outputW = narrow ? OUTPUT_WIDTH_NARROW : OUTPUT_WIDTH;
    const fixed = outputW + 3 * (narrow ? MIN_GAP_NARROW : MIN_GAP);
    const gridH = Math.max(40, Math.min(MAX_GRID, (width - fixed) / 2.5));
    const contentH = Math.max(gridH, MIN_OUTPUT_STEP * 10);
    const gridTop = LABEL_TOP + (contentH - gridH) / 2;
    this.height = Math.round(contentH + LABEL_TOP + LABEL_BOTTOM);

    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Column widths: input and hidden 1 are square, hidden 2 is half as wide.
    const widths = [gridH, gridH, gridH / 2, outputW];
    const gap = (width - widths.reduce((a, b) => a + b, 0)) / 3;
    let x = 0;
    this.columns = widths.map((w) => {
      const col = { x, w };
      x += w + gap;
      return col;
    });

    this.cell = gridH / 28;
    this.gridTop = gridTop;
    this.positions = LAYERS.map((layer, l) => {
      const col = this.columns[l]!;
      if (l === 3) {
        const step = contentH / layer.rows;
        return Array.from({ length: layer.size }, (_, d) => ({ x: col.x + 22, y: LABEL_TOP + step * (d + 0.5) }));
      }
      const step = gridH / layer.rows;
      return Array.from({ length: layer.size }, (_, i) => ({
        x: col.x + step * ((i % layer.cols) + 0.5),
        y: gridTop + step * (Math.floor(i / layer.cols) + 0.5),
      }));
    });
    this.nodeRadius = LAYERS.map((layer, l) => Math.min(9, ((l === 3 ? contentH : gridH) / layer.rows) * 0.36));
    this.start();
  }

  // Draws one frame; returns true while activations are still easing.
  private draw(): boolean {
    const { ctx } = this;
    const css = getComputedStyle(document.documentElement);
    const color = (name: string) => css.getPropertyValue(name).trim();
    const surface = color("--surface");
    const track = color("--track");
    const accent = color("--accent");
    const negative = color("--negative");
    const ink = color("--ink");
    const ink2 = color("--ink-2");
    const muted = color("--muted");
    const now = performance.now();

    // Ease each layer toward its target once the signal has reached it.
    let settling = false;
    this.current.forEach((cur, l) => {
      if (now < this.startedAt + l * STAGE_DELAY) {
        settling = true;
        return;
      }
      const tgt = this.target[l]!;
      for (let i = 0; i < cur.length; i++) {
        const d = tgt[i]! - cur[i]!;
        if (Math.abs(d) > 0.002) settling = true;
        cur[i]! += d * EASE;
      }
    });

    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, this.width, this.height);

    // Connections, with pulses travelling along the strongest ones.
    this.edges.forEach((edges, t) => {
      const from = this.positions[t]!;
      const to = this.positions[t + 1]!;
      const progress = (now - this.startedAt - t * STAGE_DELAY) / PULSE_MS;
      const reveal = Math.min(1, Math.max(0, progress));
      if (reveal === 0) return;
      ctx.lineWidth = 1;
      for (const [i, j, c] of edges) {
        const strength = (Math.abs(c) / this.edgeMax[t]!) ** 0.8;
        ctx.globalAlpha = strength * 0.55 * reveal;
        ctx.strokeStyle = c >= 0 ? accent : negative;
        ctx.beginPath();
        ctx.moveTo(from[i]!.x, from[i]!.y);
        ctx.lineTo(to[j]!.x, to[j]!.y);
        ctx.stroke();
      }
      if (progress > 0 && progress < 1) {
        const p = 1 - (1 - progress) ** 2; // ease out
        for (const [i, j, c] of edges.slice(0, PULSES_PER_GAP)) {
          ctx.globalAlpha = 1 - progress * 0.6;
          ctx.fillStyle = c >= 0 ? accent : negative;
          ctx.beginPath();
          ctx.arc(from[i]!.x + (to[j]!.x - from[i]!.x) * p, from[i]!.y + (to[j]!.y - from[i]!.y) * p, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });
    ctx.globalAlpha = 1;

    // Input: the 28x28 image the network sees.
    const inputCol = this.columns[0]!;
    ctx.fillStyle = "#000";
    ctx.fillRect(inputCol.x, this.gridTop, inputCol.w, inputCol.w);
    this.current[0]!.forEach((v, i) => {
      if (v < 0.02) return;
      const g = Math.round(v * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      const p = this.positions[0]![i]!;
      ctx.fillRect(p.x - this.cell / 2, p.y - this.cell / 2, this.cell + 0.5, this.cell + 0.5);
    });

    // Hidden and output neurons: a track-coloured dot overlaid with accent by activation.
    for (let l = 1; l < LAYERS.length; l++) {
      const r = this.nodeRadius[l]!;
      this.current[l]!.forEach((v, i) => {
        const p = this.positions[l]![i]!;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = track;
        ctx.fill();
        const a = Math.min(1, v / this.layerMax[l]!);
        if (a > 0.01) {
          ctx.globalAlpha = 0.15 + 0.85 * a;
          ctx.fillStyle = accent;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      });
    }

    // Output labels and probability bars.
    const outCol = this.columns[3]!;
    const barX = outCol.x + 36;
    const barW = outCol.w - 36;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    this.current[3]!.forEach((v, d) => {
      const p = this.positions[3]![d]!;
      const top = d === this.prediction;
      ctx.fillStyle = top ? ink : muted;
      ctx.font = `${top ? 600 : 400} 12px system-ui, sans-serif`;
      ctx.textAlign = "right";
      ctx.fillText(String(d), p.x - this.nodeRadius[3]! - 5, p.y);
      if (!this.showBars) return;
      ctx.fillStyle = track;
      ctx.fillRect(barX, p.y - 3, barW, 6);
      ctx.fillStyle = accent;
      ctx.fillRect(barX, p.y - 3, barW * Math.min(1, v), 6);
    });

    // Layer captions.
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    LAYERS.forEach((layer, l) => {
      const col = this.columns[l]!;
      const cx = col.x + col.w / 2;
      ctx.fillStyle = ink2;
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText(layer.label, cx, LABEL_TOP - 12);
      ctx.fillStyle = muted;
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(this.showBars ? layer.detail : String(layer.size), cx, this.height - LABEL_BOTTOM + 22);
    });

    return settling;
  }
}
