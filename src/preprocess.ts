// Turns an arbitrary grayscale image of a digit into MNIST format:
// white ink on black, cropped, scaled to fit a 20x20 box (aspect kept),
// then placed in a 28x28 frame with its centre of mass at the centre.
// Pure TypeScript so it runs in the browser and in tests alike.

export const MNIST_SIZE = 28;
const BOX = 20;

/**
 * @param gray luminance in [0, 1] per pixel, row-major, length w*h
 * @returns 784 values in [0, 1], or null if no digit-like ink is found
 */
export function toMnist(gray: Float32Array, w: number, h: number): Float32Array | null {
  const ink = extractInk(gray, w, h);
  if (!ink) return null;

  const box = boundingBox(ink, w, h, 0.05);
  if (!box) return null;
  const cropped = crop(ink, w, box);

  const scale = BOX / Math.max(box.w, box.h);
  const tw = Math.max(1, Math.round(box.w * scale));
  const th = Math.max(1, Math.round(box.h * scale));
  const small = resizeBox(cropped, box.w, box.h, tw, th);

  // Shift so the centre of mass lands on the frame centre, as MNIST does.
  const [cx, cy] = centerOfMass(small, tw, th);
  const ox = clamp(Math.round(MNIST_SIZE / 2 - cx), 0, MNIST_SIZE - tw);
  const oy = clamp(Math.round(MNIST_SIZE / 2 - cy), 0, MNIST_SIZE - th);

  const out = new Float32Array(MNIST_SIZE * MNIST_SIZE);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) out[(oy + y) * MNIST_SIZE + ox + x] = small[y * tw + x]!;
  }
  return out;
}

// Maps pixels to "ink amount" in [0, 1], whichever polarity the image has.
// The background level is taken from the border; faint paper texture is cut off
// and strokes are stretched to full intensity.
function extractInk(gray: Float32Array, w: number, h: number): Float32Array | null {
  const border: number[] = [];
  for (let x = 0; x < w; x++) border.push(gray[x]!, gray[(h - 1) * w + x]!);
  for (let y = 0; y < h; y++) border.push(gray[y * w]!, gray[y * w + w - 1]!);
  border.sort((a, b) => a - b);
  const bg = border[Math.floor(border.length / 2)]!;
  const darkBackground = bg < 0.5;

  const ink = new Float32Array(gray.length);
  let max = 0;
  for (let i = 0; i < gray.length; i++) {
    ink[i] = darkBackground ? gray[i]! : 1 - gray[i]!;
    max = Math.max(max, ink[i]!);
  }
  const bgInk = darkBackground ? bg : 1 - bg;
  const range = max - bgInk;
  if (range < 0.15) return null;

  const low = bgInk + 0.25 * range;
  const span = 0.6 * range;
  for (let i = 0; i < ink.length; i++) ink[i] = clamp((ink[i]! - low) / span, 0, 1);
  return ink;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

function boundingBox(img: Float32Array, w: number, h: number, threshold: number): Box | null {
  let x0 = w;
  let y0 = h;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (img[y * w + x]! > threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

function crop(img: Float32Array, w: number, box: Box): Float32Array {
  const out = new Float32Array(box.w * box.h);
  for (let y = 0; y < box.h; y++) {
    out.set(img.subarray((box.y + y) * w + box.x, (box.y + y) * w + box.x + box.w), y * box.w);
  }
  return out;
}

// Area-averaging resize (separable box filter); exact for downscaling.
export function resizeBox(src: Float32Array, w: number, h: number, tw: number, th: number): Float32Array {
  const rows = new Float32Array(tw * h);
  for (let y = 0; y < h; y++) resample1d(src, y * w, 1, w, rows, y * tw, 1, tw);
  const out = new Float32Array(tw * th);
  for (let x = 0; x < tw; x++) resample1d(rows, x, tw, h, out, x, tw, th);
  return out;
}

function resample1d(
  src: Float32Array, srcOffset: number, srcStride: number, n: number,
  dst: Float32Array, dstOffset: number, dstStride: number, m: number,
): void {
  const step = n / m;
  for (let o = 0; o < m; o++) {
    const start = o * step;
    const end = start + step;
    let sum = 0;
    for (let i = Math.floor(start); i < Math.min(n, Math.ceil(end)); i++) {
      const overlap = Math.min(end, i + 1) - Math.max(start, i);
      sum += src[srcOffset + i * srcStride]! * overlap;
    }
    dst[dstOffset + o * dstStride] = sum / step;
  }
}

function centerOfMass(img: Float32Array, w: number, h: number): [number, number] {
  let total = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = img[y * w + x]!;
      total += v;
      sx += v * (x + 0.5);
      sy += v * (y + 0.5);
    }
  }
  return total > 0 ? [sx / total, sy / total] : [w / 2, h / 2];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
