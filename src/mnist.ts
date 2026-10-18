// Downloads, caches and parses the MNIST dataset (IDX file format).

export interface Dataset {
  /** Flat pixel data, shape [count, inputSize], scaled to [0, 1]. */
  images: Float32Array;
  labels: Uint8Array;
  count: number;
  inputSize: number;
}

const BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const DATA_DIR = new URL("../data/", import.meta.url).pathname;

const FILES = {
  trainImages: "train-images-idx3-ubyte.gz",
  trainLabels: "train-labels-idx1-ubyte.gz",
  testImages: "t10k-images-idx3-ubyte.gz",
  testLabels: "t10k-labels-idx1-ubyte.gz",
} as const;

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;

async function loadFile(gzName: string): Promise<Uint8Array> {
  const path = DATA_DIR + gzName.replace(/\.gz$/, "");
  const cached = Bun.file(path);
  if (await cached.exists()) return new Uint8Array(await cached.arrayBuffer());

  console.log(`Downloading ${gzName}...`);
  const res = await fetch(BASE_URL + gzName);
  if (!res.ok) throw new Error(`Failed to download ${gzName}: HTTP ${res.status}`);
  const raw = Bun.gunzipSync(new Uint8Array(await res.arrayBuffer()));
  await Bun.write(path, raw);
  return raw;
}

function parseImages(bytes: Uint8Array): { pixels: Float32Array; count: number; size: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const magic = view.getUint32(0);
  if (magic !== IMAGE_MAGIC) throw new Error(`Bad image file magic: ${magic}`);
  const count = view.getUint32(4);
  const rows = view.getUint32(8);
  const cols = view.getUint32(12);
  const size = rows * cols;

  const pixels = new Float32Array(count * size);
  for (let i = 0; i < pixels.length; i++) pixels[i] = bytes[16 + i]! / 255;
  return { pixels, count, size };
}

function parseLabels(bytes: Uint8Array): Uint8Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const magic = view.getUint32(0);
  if (magic !== LABEL_MAGIC) throw new Error(`Bad label file magic: ${magic}`);
  const count = view.getUint32(4);
  return bytes.slice(8, 8 + count);
}

async function loadSet(imagesFile: string, labelsFile: string): Promise<Dataset> {
  const [imageBytes, labelBytes] = await Promise.all([loadFile(imagesFile), loadFile(labelsFile)]);
  const images = parseImages(imageBytes);
  const labels = parseLabels(labelBytes);
  if (labels.length !== images.count) throw new Error("Image/label count mismatch");
  return { images: images.pixels, labels, count: images.count, inputSize: images.size };
}

export async function loadMnist(): Promise<{ train: Dataset; test: Dataset }> {
  const [train, test] = await Promise.all([
    loadSet(FILES.trainImages, FILES.trainLabels),
    loadSet(FILES.testImages, FILES.testLabels),
  ]);
  return { train, test };
}

// Splits off the last `size` samples of a set (e.g. for validation).
export function splitSet(set: Dataset, size: number): [Dataset, Dataset] {
  const keep = set.count - size;
  const n = set.inputSize;
  return [
    { ...set, images: set.images.subarray(0, keep * n), labels: set.labels.subarray(0, keep), count: keep },
    { ...set, images: set.images.subarray(keep * n), labels: set.labels.subarray(keep), count: size },
  ];
}
