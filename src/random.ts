// Small seeded PRNG so training runs are reproducible.

export interface Rng {
  /** Uniform float in [0, 1). */
  next(): number;
  /** Standard normal sample. */
  gaussian(): number;
  /** In-place Fisher-Yates shuffle. */
  shuffle<T>(array: T[]): T[];
}

export function createRng(seed = 42): Rng {
  let state = seed >>> 0;

  // mulberry32
  function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Box-Muller
  function gaussian(): number {
    let u = 0;
    while (u === 0) u = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  }

  function shuffle<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [array[i], array[j]] = [array[j]!, array[i]!];
    }
    return array;
  }

  return { next, gaussian, shuffle };
}
