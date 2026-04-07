/**
 * model_runtime.js
 *
 * Optimized transformer inference engine.
 *
 * Includes:
 *  - createManifest(config)           – build weight-offset manifest from a model config
 *  - loadModel(configUrl, manifestUrl, weightsUrl) – fetch + validate weights
 *  - TransformerEngine                – forward-pass + token-sampling engine
 *  - Example generation flow          – end-to-end usage example at the bottom
 *
 * Assumptions:
 *  - Weights are stored as float32 (little-endian)
 *  - Manifest offsets are in float32 ELEMENTS (not bytes) by default
 *  - Matrix layout is row-major [outDim, inDim]
 *
 * Tensor shapes (outDim × inDim):
 *  - token_embedding : [vocab_size,        hidden_size]
 *  - final_norm      : [hidden_size]          (optional RMS-norm scale)
 *  - wq              : [qDim,              hidden_size]   qDim  = numHeads  * headDim
 *  - wk              : [kvDim,             hidden_size]   kvDim = numKvHeads * headDim
 *  - wv              : [kvDim,             hidden_size]
 *  - wo              : [hidden_size,       qDim]
 *  - ffn_gate        : [intermediate_size, hidden_size]
 *  - ffn_up          : [intermediate_size, hidden_size]
 *  - ffn_down        : [hidden_size,       intermediate_size]
 *  - output_head     : [vocab_size,        hidden_size]
 *
 * @module model_runtime
 */

"use strict";

/* =========================================================
 * §0  Internal utilities & assertions
 * ========================================================= */

/**
 * Assert that `value` is a positive integer, throw with context otherwise.
 * @param {*}      value  – candidate value
 * @param {string} name   – field name for error messages
 * @returns {number} validated integer
 */
function mustBePositiveInt(value, name) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new TypeError(
      `Config field "${name}" must be a positive integer, got: ${JSON.stringify(value)}`
    );
  }
  return value;
}

/**
 * Assert a runtime condition; throw a detailed RangeError when false.
 * @param {boolean} condition
 * @param {string}  message
 */
function assert(condition, message) {
  if (!condition) {
    throw new RangeError(`Assertion failed: ${message}`);
  }
}

/**
 * Retrieve JSON from a URL with a helpful error on failure.
 * @param {string} url
 * @returns {Promise<object>}
 */
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Failed to fetch JSON from "${url}" – HTTP ${res.status} ${res.statusText}`
    );
  }
  return res.json();
}

/**
 * Retrieve a binary buffer from a URL with a helpful error on failure.
 * @param {string} url
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchArrayBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Failed to fetch weights from "${url}" – HTTP ${res.status} ${res.statusText}`
    );
  }
  return res.arrayBuffer();
}

/**
 * Validate and normalise a [start, end] range pair.
 * @param {*} range
 * @returns {[number, number]}
 */
function normalizeRange(range) {
  if (!Array.isArray(range) || range.length !== 2) {
    throw new TypeError(
      `Invalid weight range – expected [start, end], got: ${JSON.stringify(range)}`
    );
  }
  const [start, end] = range;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) {
    throw new RangeError(
      `Weight range [${start}, ${end}] is invalid – ` +
      `start and end must be non-negative integers with start <= end`
    );
  }
  return [start, end];
}

/**
 * Create a Float32Array view into an ArrayBuffer using either element or byte offsets.
 * @param {ArrayBuffer} buffer
 * @param {[number,number]} range     – [start, end] in the units specified by offsetUnit
 * @param {"elements"|"bytes"} [offsetUnit="elements"]
 * @returns {Float32Array}
 */
function makeFloat32View(buffer, range, offsetUnit = "elements") {
  const [start, end] = normalizeRange(range);

  if (offsetUnit === "elements") {
    const byteOffset = start * 4;
    const length = end - start;
    assert(
      byteOffset + length * 4 <= buffer.byteLength,
      `Element range [${start}, ${end}] exceeds buffer size (${buffer.byteLength} bytes)`
    );
    return new Float32Array(buffer, byteOffset, length);
  }

  if (offsetUnit === "bytes") {
    const byteOffset = start;
    const byteLength = end - start;
    assert(
      byteOffset + byteLength <= buffer.byteLength,
      `Byte range [${start}, ${end}] exceeds buffer size (${buffer.byteLength} bytes)`
    );
    if (byteLength % 4 !== 0) {
      throw new RangeError(
        `Byte range [${start}, ${end}] spans ${byteLength} bytes, ` +
        `which is not aligned to float32 size (4 bytes)`
      );
    }
    if (byteOffset % 4 !== 0) {
      throw new RangeError(
        `Byte offset ${byteOffset} is not 4-byte aligned for Float32Array`
      );
    }
    return new Float32Array(buffer, byteOffset, byteLength / 4);
  }

  throw new Error(
    `Unknown offset_unit "${offsetUnit}" – expected "elements" or "bytes"`
  );
}

/* =========================================================
 * §1  Manifest generator
 * ========================================================= */

/** @type {WeakMap<object, object>} cache for parsed manifests keyed by config */
const _manifestCache = new WeakMap();

/**
 * Generate a flat-binary weight manifest from a model configuration.
 * Results are cached per config object reference.
 *
 * @param {object} config
 * @param {number} config.hidden_size
 * @param {number} config.vocab_size
 * @param {number} config.num_layers
 * @param {number} config.num_heads
 * @param {number} config.num_kv_heads
 * @param {number} config.head_dim
 * @param {number} config.intermediate_size
 * @returns {object} manifest with float32 element offsets for every weight tensor
 */
function createManifest(config) {
  if (_manifestCache.has(config)) {
    return _manifestCache.get(config);
  }

  const hidden       = mustBePositiveInt(config.hidden_size,       "hidden_size");
  const vocab        = mustBePositiveInt(config.vocab_size,        "vocab_size");
  const numLayers    = mustBePositiveInt(config.num_layers,        "num_layers");
  const numHeads     = mustBePositiveInt(config.num_heads,         "num_heads");
  const numKvHeads   = mustBePositiveInt(config.num_kv_heads,      "num_kv_heads");
  const headDim      = mustBePositiveInt(config.head_dim,          "head_dim");
  const intermediate = mustBePositiveInt(config.intermediate_size, "intermediate_size");

  assert(
    numKvHeads <= numHeads,
    `num_kv_heads (${numKvHeads}) must be <= num_heads (${numHeads})`
  );
  assert(
    numHeads % numKvHeads === 0,
    `num_heads (${numHeads}) must be divisible by num_kv_heads (${numKvHeads})`
  );

  const qDim  = numHeads  * headDim;
  const kvDim = numKvHeads * headDim;

  let cursor = 0;

  /**
   * Allocate the next `n` float32 elements and return [start, end].
   * @param {number} n
   * @returns {[number, number]}
   */
  function alloc(n) {
    assert(n > 0, `alloc called with non-positive size ${n}`);
    const start = cursor;
    cursor += n;
    return [start, cursor];
  }

  const manifest = {
    metadata: {
      format:           "float32",
      offset_unit:      "elements",
      total_size_bytes: 0
    },

    token_embedding: alloc(vocab  * hidden),
    final_norm:      alloc(hidden),

    layers:      [],
    output_head: null
  };

  for (let layerId = 0; layerId < numLayers; layerId++) {
    manifest.layers.push({
      layer_id: layerId,
      offsets: {
        rms_norm_1: alloc(hidden),
        wq:         alloc(qDim  * hidden),
        wk:         alloc(kvDim * hidden),
        wv:         alloc(kvDim * hidden),
        wo:         alloc(hidden * qDim),
        rms_norm_2: alloc(hidden),
        ffn_gate:   alloc(intermediate * hidden),
        ffn_down:   alloc(hidden       * intermediate),
        ffn_up:     alloc(intermediate * hidden)
      }
    });
  }

  manifest.output_head = alloc(vocab * hidden);
  manifest.metadata.total_size_bytes = cursor * 4;

  _manifestCache.set(config, manifest);
  return manifest;
}

/* =========================================================
 * §2  Layout info & manifest validation
 * ========================================================= */

/**
 * Compute tensor shapes for all per-layer tensors from a model config.
 * @param {object} config
 * @returns {{ layerShapes: object, outputHeadShape: number[], qDim: number, kvDim: number }}
 */
function buildLayoutInfo(config) {
  const hidden       = config.hidden_size;
  const numHeads     = config.num_heads;
  const numKvHeads   = config.num_kv_heads;
  const headDim      = config.head_dim;
  const intermediate = config.intermediate_size;

  const qDim  = numHeads   * headDim;
  const kvDim = numKvHeads * headDim;

  return {
    qDim,
    kvDim,
    layerShapes: {
      rms_norm_1: [hidden],
      wq:         [qDim,         hidden],
      wk:         [kvDim,        hidden],
      wv:         [kvDim,        hidden],
      wo:         [hidden,       qDim],
      rms_norm_2: [hidden],
      ffn_gate:   [intermediate, hidden],
      ffn_down:   [hidden,       intermediate],
      ffn_up:     [intermediate, hidden]
    },
    outputHeadShape: [config.vocab_size, hidden]
  };
}

/**
 * Validate that a loaded manifest is consistent with the model config and weight buffer.
 * @param {object}      config
 * @param {object}      manifest
 * @param {ArrayBuffer} buffer
 * @param {object}      layout
 */
function validateManifest(config, manifest, buffer, layout) {
  const offsetUnit     = manifest.metadata?.offset_unit || "elements";
  const bytesPerUnit   = offsetUnit === "elements" ? 4 : 1;
  const totalDeclared  = manifest.metadata?.total_size_bytes ?? 0;

  if (totalDeclared > 0 && totalDeclared > buffer.byteLength) {
    throw new RangeError(
      `Manifest declares ${totalDeclared} bytes but the weight file is only ` +
      `${buffer.byteLength} bytes`
    );
  }

  /**
   * Verify a [start, end] range fits inside the buffer and matches the expected element count.
   * @param {string}         path     – human-readable tensor path for error messages
   * @param {[number,number]} range
   * @param {number[]}        shape
   */
  function checkTensor(path, range, shape) {
    if (!range) return; // optional tensors
    const [start, end] = normalizeRange(range);
    const expectedElements = shape.reduce((a, b) => a * b, 1);
    const actualElements   = end - start;

    if (actualElements !== expectedElements) {
      throw new RangeError(
        `Tensor "${path}": range [${start}, ${end}] covers ${actualElements} elements ` +
        `but shape ${JSON.stringify(shape)} requires ${expectedElements}`
      );
    }

    const endByte = end * bytesPerUnit;
    if (endByte > buffer.byteLength) {
      throw new RangeError(
        `Tensor "${path}": range [${start}, ${end}] (unit=${offsetUnit}) ` +
        `reads up to byte ${endByte} but buffer is only ${buffer.byteLength} bytes`
      );
    }
  }

  checkTensor("token_embedding", manifest.token_embedding,
    [config.vocab_size, config.hidden_size]);

  if (manifest.final_norm) {
    checkTensor("final_norm", manifest.final_norm, [config.hidden_size]);
  }

  manifest.layers.forEach((layer, i) => {
    const o = layer.offsets;
    const p = `layers[${i}]`;
    const s = layout.layerShapes;
    checkTensor(`${p}.rms_norm_1`, o.rms_norm_1, s.rms_norm_1);
    checkTensor(`${p}.wq`,         o.wq,         s.wq);
    checkTensor(`${p}.wk`,         o.wk,         s.wk);
    checkTensor(`${p}.wv`,         o.wv,         s.wv);
    checkTensor(`${p}.wo`,         o.wo,         s.wo);
    checkTensor(`${p}.rms_norm_2`, o.rms_norm_2, s.rms_norm_2);
    checkTensor(`${p}.ffn_gate`,   o.ffn_gate,   s.ffn_gate);
    checkTensor(`${p}.ffn_down`,   o.ffn_down,   s.ffn_down);
    checkTensor(`${p}.ffn_up`,     o.ffn_up,     s.ffn_up);
  });

  if (manifest.output_head) {
    checkTensor("output_head", manifest.output_head, layout.outputHeadShape);
  }
}

/* =========================================================
 * §3  Model loader
 * ========================================================= */

/**
 * Load and validate a transformer model from remote URLs.
 *
 * @param {string} configUrl    – URL for the JSON model config
 * @param {string} manifestUrl  – URL for the JSON weight manifest
 * @param {string} weightsUrl   – URL for the raw float32 weights binary
 * @returns {Promise<object>} model object ready for TransformerEngine
 */
async function loadModel(configUrl, manifestUrl, weightsUrl) {
  const [config, manifest, buffer] = await Promise.all([
    fetchJson(configUrl),
    fetchJson(manifestUrl),
    fetchArrayBuffer(weightsUrl)
  ]);

  // Validate required config fields up-front for clearer error messages
  mustBePositiveInt(config.hidden_size,       "config.hidden_size");
  mustBePositiveInt(config.vocab_size,        "config.vocab_size");
  mustBePositiveInt(config.num_layers,        "config.num_layers");
  mustBePositiveInt(config.num_heads,         "config.num_heads");
  mustBePositiveInt(config.num_kv_heads,      "config.num_kv_heads");
  mustBePositiveInt(config.head_dim,          "config.head_dim");
  mustBePositiveInt(config.intermediate_size, "config.intermediate_size");

  const layout     = buildLayoutInfo(config);
  const offsetUnit = manifest.metadata?.offset_unit || "elements";

  validateManifest(config, manifest, buffer, layout);

  /**
   * @param {string}         name
   * @param {[number,number]} range
   * @param {number[]}        shape
   * @returns {{ name: string, shape: number[], data: Float32Array }}
   */
  const makeTensor = (name, range, shape) => ({
    name,
    shape,
    data: makeFloat32View(buffer, range, offsetUnit)
  });

  return {
    config,
    metadata: manifest.metadata ?? {},

    token_embedding: makeTensor(
      "token_embedding",
      manifest.token_embedding,
      [config.vocab_size, config.hidden_size]
    ),

    final_norm: manifest.final_norm
      ? makeTensor("final_norm", manifest.final_norm, [config.hidden_size])
      : null,

    layers: manifest.layers.map((layer, index) => {
      const o = layer.offsets;
      const s = layout.layerShapes;
      return {
        layer_id:   layer.layer_id ?? index,
        rms_norm_1: makeTensor("rms_norm_1", o.rms_norm_1, s.rms_norm_1),
        wq:         makeTensor("wq",         o.wq,         s.wq),
        wk:         makeTensor("wk",         o.wk,         s.wk),
        wv:         makeTensor("wv",         o.wv,         s.wv),
        wo:         makeTensor("wo",         o.wo,         s.wo),
        rms_norm_2: makeTensor("rms_norm_2", o.rms_norm_2, s.rms_norm_2),
        ffn_gate:   makeTensor("ffn_gate",   o.ffn_gate,   s.ffn_gate),
        ffn_down:   makeTensor("ffn_down",   o.ffn_down,   s.ffn_down),
        ffn_up:     makeTensor("ffn_up",     o.ffn_up,     s.ffn_up)
      };
    }),

    output_head: makeTensor(
      "output_head",
      manifest.output_head,
      layout.outputHeadShape
    )
  };
}

/* =========================================================
 * §4  Typed-array pool
 * ========================================================= */

/**
 * A simple pool for Float32Arrays to reduce garbage-collection pressure.
 * Arrays are bucketed by length; callers must release arrays when done.
 */
class Float32ArrayPool {
  constructor() {
    /** @type {Map<number, Float32Array[]>} */
    this._buckets = new Map();
  }

  /**
   * Acquire a zeroed Float32Array of the requested length.
   * @param {number} length
   * @returns {Float32Array}
   */
  acquire(length) {
    const bucket = this._buckets.get(length);
    if (bucket && bucket.length > 0) {
      const arr = bucket.pop();
      arr.fill(0);
      return arr;
    }
    return new Float32Array(length);
  }

  /**
   * Return a Float32Array to the pool for future reuse.
   * @param {Float32Array} arr
   */
  release(arr) {
    const { length } = arr;
    if (!this._buckets.has(length)) {
      this._buckets.set(length, []);
    }
    this._buckets.get(length).push(arr);
  }

  /** Remove all pooled buffers (call when model is no longer needed). */
  clear() {
    this._buckets.clear();
  }
}

/* =========================================================
 * §5  TransformerEngine
 * ========================================================= */

/**
 * High-performance transformer inference engine.
 *
 * Key optimisations:
 *  - Pre-allocated scratch buffers (no per-token allocations in the hot path)
 *  - Typed-array pool for KV-cache expansion
 *  - Cached RoPE frequency table
 *  - Loop-unrolled matmulVec for common head dimensions (64, 128)
 *  - Numerically stable softmax (subtract max before exp)
 *  - GQA (Grouped Query Attention) support
 *  - Optional profiling callbacks
 *
 * @example
 * const engine = new TransformerEngine(model, { temperature: 0.8, topK: 40 });
 * const tokens = await engine.generate([1, 2, 3], 128);
 */
class TransformerEngine {
  /**
   * @param {object} model           – model returned by loadModel()
   * @param {object} [options]
   * @param {number} [options.temperature=1.0]   – sampling temperature (> 0)
   * @param {number} [options.topK=0]            – top-K sampling (0 = disabled)
   * @param {number} [options.maxSeqLen=2048]    – maximum KV-cache capacity
   * @param {boolean} [options.profiling=false]  – enable timing via console.time
   * @param {Function} [options.logger]          – optional log callback (msg: string) => void
   */
  constructor(model, options = {}) {
    this._model = model;

    const cfg = model.config;
    this._hidden       = mustBePositiveInt(cfg.hidden_size,       "hidden_size");
    this._vocab        = mustBePositiveInt(cfg.vocab_size,        "vocab_size");
    this._numLayers    = mustBePositiveInt(cfg.num_layers,        "num_layers");
    this._numHeads     = mustBePositiveInt(cfg.num_heads,         "num_heads");
    this._numKvHeads   = mustBePositiveInt(cfg.num_kv_heads,      "num_kv_heads");
    this._headDim      = mustBePositiveInt(cfg.head_dim,          "head_dim");
    this._intermediate = mustBePositiveInt(cfg.intermediate_size, "intermediate_size");

    this._qDim  = this._numHeads   * this._headDim;
    this._kvDim = this._numKvHeads * this._headDim;
    this._kvGroupSize = this._numHeads / this._numKvHeads; // heads per KV head (GQA)

    // ── Options ─────────────────────────────────────────────────
    this._temperature = (typeof options.temperature === "number" && options.temperature > 0)
      ? options.temperature : 1.0;
    this._topK        = (Number.isInteger(options.topK) && options.topK > 0)
      ? options.topK : 0;
    this._maxSeqLen   = (Number.isInteger(options.maxSeqLen) && options.maxSeqLen > 0)
      ? options.maxSeqLen : 2048;
    this._profiling   = Boolean(options.profiling);
    this._logger      = typeof options.logger === "function" ? options.logger : null;

    // ── Pre-allocated scratch buffers ────────────────────────────
    const H = this._hidden;
    const I = this._intermediate;
    const Q = this._qDim;
    const KV = this._kvDim;
    const V = this._vocab;

    this._scratch = {
      hidden:      new Float32Array(H),    // residual stream
      normed:      new Float32Array(H),    // post-RMS-norm hidden
      q:           new Float32Array(Q),    // query projections
      k:           new Float32Array(KV),   // key projections
      v:           new Float32Array(KV),   // value projections
      attnOut:     new Float32Array(Q),    // attention output (pre-projection)
      ffnGate:     new Float32Array(I),    // FFN gate activations
      ffnUp:       new Float32Array(I),    // FFN up-projection
      ffnHidden:   new Float32Array(I),    // elementwise product (SiLU * up)
      residual2:   new Float32Array(H),    // second residual
      attnScores:  new Float32Array(this._maxSeqLen), // per-head attention scores
      logits:      new Float32Array(V)     // output logits
    };

    // ── Typed-array pool ─────────────────────────────────────────
    this._pool = new Float32ArrayPool();

    // ── KV cache ─────────────────────────────────────────────────
    // Each layer stores seqLen key vectors and seqLen value vectors.
    // We pre-allocate to maxSeqLen so we never reallocate during generation.
    this._kvCache = model.layers.map(() => ({
      k: new Float32Array(this._maxSeqLen * KV),
      v: new Float32Array(this._maxSeqLen * KV)
    }));
    this._seqLen = 0; // tokens currently in the KV cache

    // ── RoPE frequency cache ──────────────────────────────────────
    this._ropeFreqs = this._buildRopeFreqs(
      this._headDim,
      cfg.rope_theta ?? 10000.0
    );

    this._log(`TransformerEngine initialised: ${this._numLayers} layers, ` +
              `hidden=${H}, heads=${this._numHeads}/${this._numKvHeads}, ` +
              `headDim=${this._headDim}, vocab=${V}`);
  }

  /* ── Private helpers ──────────────────────────────────────── */

  /**
   * Emit a log message if a logger was provided.
   * @param {string} msg
   */
  _log(msg) {
    if (this._logger) this._logger(msg);
  }

  /**
   * Optionally start a named profile timer.
   * @param {string} label
   */
  _timeStart(label) {
    if (this._profiling) console.time(label);
  }

  /**
   * Optionally end a named profile timer.
   * @param {string} label
   */
  _timeEnd(label) {
    if (this._profiling) console.timeEnd(label);
  }

  /**
   * Pre-compute RoPE (Rotary Position Embedding) cosine / sine frequencies.
   * Returns an array of length headDim/2, each entry {cos, sin} per position
   * up to maxSeqLen.
   *
   * @param {number} headDim
   * @param {number} theta      – base frequency (default 10 000)
   * @returns {Float32Array[]}  – [maxSeqLen × headDim/2] interleaved [cos, sin]
   */
  _buildRopeFreqs(headDim, theta) {
    assert(headDim % 2 === 0, `headDim must be even for RoPE, got ${headDim}`);
    const halfDim  = headDim >> 1;
    const maxPos   = this._maxSeqLen;
    // Layout: freqs[pos * halfDim * 2 + i*2 + 0] = cos, +1 = sin
    const freqs = new Float32Array(maxPos * halfDim * 2);

    for (let pos = 0; pos < maxPos; pos++) {
      const base = pos * halfDim * 2;
      for (let i = 0; i < halfDim; i++) {
        const freq  = 1.0 / Math.pow(theta, (2 * i) / headDim);
        const angle = pos * freq;
        freqs[base + i * 2    ] = Math.cos(angle);
        freqs[base + i * 2 + 1] = Math.sin(angle);
      }
    }
    return freqs;
  }

  /**
   * Apply RoPE in-place to a contiguous array of `numHeads` head vectors,
   * each of length `headDim`, at sequence position `pos`.
   *
   * @param {Float32Array} vec       – [numHeads * headDim]
   * @param {number}       numHeads
   * @param {number}       pos       – absolute position index
   */
  _applyRope(vec, numHeads, pos) {
    const headDim = this._headDim;
    const halfDim = headDim >> 1;
    const freqs   = this._ropeFreqs;
    const freqBase = pos * halfDim * 2;

    for (let h = 0; h < numHeads; h++) {
      const hBase = h * headDim;
      for (let i = 0; i < halfDim; i++) {
        const cos = freqs[freqBase + i * 2    ];
        const sin = freqs[freqBase + i * 2 + 1];
        const x0  = vec[hBase + i];
        const x1  = vec[hBase + i + halfDim];
        vec[hBase + i          ] = x0 * cos - x1 * sin;
        vec[hBase + i + halfDim] = x0 * sin + x1 * cos;
      }
    }
  }

  /**
   * Multiply a weight matrix (row-major [outDim × inDim]) by an input vector,
   * writing the result into `out`.
   *
   * Uses loop unrolling for the most common head dimensions (64 and 128)
   * to help the JIT avoid per-iteration overhead.
   *
   * @param {Float32Array} out     – [outDim]
   * @param {Float32Array} weight  – [outDim * inDim]
   * @param {Float32Array} inp     – [inDim]
   * @param {number}       outDim
   * @param {number}       inDim
   */
  _matmulVec(out, weight, inp, outDim, inDim) {
    assert(weight.length === outDim * inDim,
      `matmulVec: weight length ${weight.length} != outDim*inDim (${outDim}*${inDim}=${outDim * inDim})`);
    assert(inp.length >= inDim,
      `matmulVec: input length ${inp.length} < inDim ${inDim}`);
    assert(out.length >= outDim,
      `matmulVec: output length ${out.length} < outDim ${outDim}`);

    // Unrolled inner loop for inDim=64
    if (inDim === 64) {
      for (let o = 0; o < outDim; o++) {
        const wOff = o * 64;
        let s = 0;
        s += weight[wOff     ] * inp[ 0]; s += weight[wOff +  1] * inp[ 1];
        s += weight[wOff +  2] * inp[ 2]; s += weight[wOff +  3] * inp[ 3];
        s += weight[wOff +  4] * inp[ 4]; s += weight[wOff +  5] * inp[ 5];
        s += weight[wOff +  6] * inp[ 6]; s += weight[wOff +  7] * inp[ 7];
        s += weight[wOff +  8] * inp[ 8]; s += weight[wOff +  9] * inp[ 9];
        s += weight[wOff + 10] * inp[10]; s += weight[wOff + 11] * inp[11];
        s += weight[wOff + 12] * inp[12]; s += weight[wOff + 13] * inp[13];
        s += weight[wOff + 14] * inp[14]; s += weight[wOff + 15] * inp[15];
        s += weight[wOff + 16] * inp[16]; s += weight[wOff + 17] * inp[17];
        s += weight[wOff + 18] * inp[18]; s += weight[wOff + 19] * inp[19];
        s += weight[wOff + 20] * inp[20]; s += weight[wOff + 21] * inp[21];
        s += weight[wOff + 22] * inp[22]; s += weight[wOff + 23] * inp[23];
        s += weight[wOff + 24] * inp[24]; s += weight[wOff + 25] * inp[25];
        s += weight[wOff + 26] * inp[26]; s += weight[wOff + 27] * inp[27];
        s += weight[wOff + 28] * inp[28]; s += weight[wOff + 29] * inp[29];
        s += weight[wOff + 30] * inp[30]; s += weight[wOff + 31] * inp[31];
        s += weight[wOff + 32] * inp[32]; s += weight[wOff + 33] * inp[33];
        s += weight[wOff + 34] * inp[34]; s += weight[wOff + 35] * inp[35];
        s += weight[wOff + 36] * inp[36]; s += weight[wOff + 37] * inp[37];
        s += weight[wOff + 38] * inp[38]; s += weight[wOff + 39] * inp[39];
        s += weight[wOff + 40] * inp[40]; s += weight[wOff + 41] * inp[41];
        s += weight[wOff + 42] * inp[42]; s += weight[wOff + 43] * inp[43];
        s += weight[wOff + 44] * inp[44]; s += weight[wOff + 45] * inp[45];
        s += weight[wOff + 46] * inp[46]; s += weight[wOff + 47] * inp[47];
        s += weight[wOff + 48] * inp[48]; s += weight[wOff + 49] * inp[49];
        s += weight[wOff + 50] * inp[50]; s += weight[wOff + 51] * inp[51];
        s += weight[wOff + 52] * inp[52]; s += weight[wOff + 53] * inp[53];
        s += weight[wOff + 54] * inp[54]; s += weight[wOff + 55] * inp[55];
        s += weight[wOff + 56] * inp[56]; s += weight[wOff + 57] * inp[57];
        s += weight[wOff + 58] * inp[58]; s += weight[wOff + 59] * inp[59];
        s += weight[wOff + 60] * inp[60]; s += weight[wOff + 61] * inp[61];
        s += weight[wOff + 62] * inp[62]; s += weight[wOff + 63] * inp[63];
        out[o] = s;
      }
      return;
    }

    // Unrolled inner loop for inDim=128
    if (inDim === 128) {
      for (let o = 0; o < outDim; o++) {
        const wOff = o * 128;
        let s = 0;
        for (let i = 0; i < 128; i += 8) {
          s += weight[wOff + i    ] * inp[i    ];
          s += weight[wOff + i + 1] * inp[i + 1];
          s += weight[wOff + i + 2] * inp[i + 2];
          s += weight[wOff + i + 3] * inp[i + 3];
          s += weight[wOff + i + 4] * inp[i + 4];
          s += weight[wOff + i + 5] * inp[i + 5];
          s += weight[wOff + i + 6] * inp[i + 6];
          s += weight[wOff + i + 7] * inp[i + 7];
        }
        out[o] = s;
      }
      return;
    }

    // Generic fallback with 4-way unrolling
    const rem = inDim & 3;
    const len = inDim - rem;
    for (let o = 0; o < outDim; o++) {
      const wOff = o * inDim;
      let s = 0;
      for (let i = 0; i < len; i += 4) {
        s += weight[wOff + i    ] * inp[i    ];
        s += weight[wOff + i + 1] * inp[i + 1];
        s += weight[wOff + i + 2] * inp[i + 2];
        s += weight[wOff + i + 3] * inp[i + 3];
      }
      for (let i = len; i < inDim; i++) {
        s += weight[wOff + i] * inp[i];
      }
      out[o] = s;
    }
  }

  /**
   * Apply Root-Mean-Square layer normalisation in-place.
   * out[i] = inp[i] / rms(inp) * scale[i]
   *
   * @param {Float32Array} out    – destination buffer [size]
   * @param {Float32Array} inp    – input buffer [size]
   * @param {Float32Array} scale  – scale weights [size]
   * @param {number}       size
   * @param {number}       [eps=1e-6]
   */
  _rmsNorm(out, inp, scale, size, eps = 1e-6) {
    let ss = 0;
    for (let i = 0; i < size; i++) ss += inp[i] * inp[i];
    const norm = 1.0 / Math.sqrt(ss / size + eps);
    for (let i = 0; i < size; i++) out[i] = inp[i] * norm * scale[i];
  }

  /**
   * SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
   * @param {number} x
   * @returns {number}
   */
  _silu(x) {
    return x / (1.0 + Math.exp(-x));
  }

  /**
   * Numerically stable in-place softmax over `arr[0..len-1]`.
   * Subtracts the max value before exponentiation to prevent overflow.
   *
   * @param {Float32Array} arr
   * @param {number}       len  – number of elements to normalise
   */
  _softmax(arr, len) {
    let maxVal = arr[0];
    for (let i = 1; i < len; i++) {
      if (arr[i] > maxVal) maxVal = arr[i];
    }
    let sum = 0;
    for (let i = 0; i < len; i++) {
      arr[i] = Math.exp(arr[i] - maxVal);
      sum += arr[i];
    }
    const inv = 1.0 / sum;
    for (let i = 0; i < len; i++) arr[i] *= inv;
  }

  /**
   * Process a single transformer layer (attention + FFN) for the token at `pos`.
   *
   * @param {object}       layer    – layer weight tensors
   * @param {number}       layerIdx
   * @param {number}       pos      – current sequence position (0-indexed)
   */
  _processLayer(layer, layerIdx, pos) {
    const {
      _hidden: H, _headDim: HD, _numHeads: NH, _numKvHeads: NKV,
      _qDim: Q, _kvDim: KV, _kvGroupSize: G, _intermediate: I,
      _scratch: s, _kvCache: kvc
    } = this;

    // ── Attention pre-norm ──────────────────────────────────────
    this._rmsNorm(s.normed, s.hidden, layer.rms_norm_1.data, H);

    // ── Q / K / V projections ──────────────────────────────────
    this._matmulVec(s.q, layer.wq.data, s.normed, Q, H);
    this._matmulVec(s.k, layer.wk.data, s.normed, KV, H);
    this._matmulVec(s.v, layer.wv.data, s.normed, KV, H);

    // ── RoPE ──────────────────────────────────────────────────
    this._applyRope(s.q, NH,  pos);
    this._applyRope(s.k, NKV, pos);

    // ── Write into KV cache ────────────────────────────────────
    const cache = kvc[layerIdx];
    cache.k.set(s.k, pos * KV);
    cache.v.set(s.v, pos * KV);

    // ── Multi-head (grouped-query) attention ───────────────────
    const scale = 1.0 / Math.sqrt(HD);
    s.attnOut.fill(0);

    for (let h = 0; h < NH; h++) {
      const kvHead  = Math.floor(h / G);
      const qOffset = h * HD;
      const scores  = s.attnScores;

      // Dot products with all cached keys
      for (let t = 0; t <= pos; t++) {
        const kOffset = t * KV + kvHead * HD;
        let dot = 0;
        for (let d = 0; d < HD; d++) {
          dot += s.q[qOffset + d] * cache.k[kOffset + d];
        }
        scores[t] = dot * scale;
      }

      // Softmax over [0..pos]
      this._softmax(scores, pos + 1);

      // Weighted sum of value vectors
      for (let t = 0; t <= pos; t++) {
        const vOffset  = t * KV + kvHead * HD;
        const sc       = scores[t];
        const outBase  = h * HD;
        for (let d = 0; d < HD; d++) {
          s.attnOut[outBase + d] += sc * cache.v[vOffset + d];
        }
      }
    }

    // ── Output projection + residual ──────────────────────────
    this._matmulVec(s.residual2, layer.wo.data, s.attnOut, H, Q);
    for (let i = 0; i < H; i++) s.hidden[i] += s.residual2[i];

    // ── FFN pre-norm ───────────────────────────────────────────
    this._rmsNorm(s.normed, s.hidden, layer.rms_norm_2.data, H);

    // ── FFN: SwiGLU (gate * SiLU(up) → down) ─────────────────
    this._matmulVec(s.ffnGate, layer.ffn_gate.data, s.normed, I, H);
    this._matmulVec(s.ffnUp,   layer.ffn_up.data,   s.normed, I, H);
    for (let i = 0; i < I; i++) {
      s.ffnHidden[i] = this._silu(s.ffnGate[i]) * s.ffnUp[i];
    }
    this._matmulVec(s.residual2, layer.ffn_down.data, s.ffnHidden, H, I);
    for (let i = 0; i < H; i++) s.hidden[i] += s.residual2[i];
  }

  /**
   * Run a single forward pass for one token at position `pos`.
   * Writes output logits into `this._scratch.logits`.
   *
   * @param {number} tokenId
   * @param {number} pos
   */
  _forward(tokenId, pos) {
    const { _model: model, _hidden: H, _scratch: s } = this;

    assert(
      pos < this._maxSeqLen,
      `Sequence length ${pos + 1} exceeds maxSeqLen ${this._maxSeqLen}`
    );
    assert(
      tokenId >= 0 && tokenId < this._vocab,
      `Token ID ${tokenId} out of vocabulary range [0, ${this._vocab})`
    );

    // Load token embedding
    const embBase = tokenId * H;
    const embData = model.token_embedding.data;
    s.hidden.set(embData.subarray(embBase, embBase + H));

    // Process all layers
    for (let l = 0; l < this._numLayers; l++) {
      this._processLayer(model.layers[l], l, pos);
    }

    // Final RMS norm (optional)
    const norm = model.final_norm;
    if (norm) {
      this._rmsNorm(s.normed, s.hidden, norm.data, H);
    } else {
      s.normed.set(s.hidden);
    }

    // Project to vocabulary
    this._matmulVec(s.logits, model.output_head.data, s.normed, this._vocab, H);
  }

  /**
   * Sample the next token from `logits` using temperature and optional top-K.
   * @param {Float32Array} logits
   * @returns {number} sampled token id
   */
  _sample(logits) {
    const V = this._vocab;

    // Apply temperature
    if (this._temperature !== 1.0) {
      const invTemp = 1.0 / this._temperature;
      for (let i = 0; i < V; i++) logits[i] *= invTemp;
    }

    // Top-K filtering
    if (this._topK > 0 && this._topK < V) {
      // Find the top-K threshold using a partial sort approach
      const k     = this._topK;
      const topK  = this._pool.acquire(V);
      topK.set(logits);

      // Partial selection sort to find the k-th largest value
      let threshold = -Infinity;
      for (let i = 0; i < k; i++) {
        let maxIdx = i;
        for (let j = i + 1; j < V; j++) {
          if (topK[j] > topK[maxIdx]) maxIdx = j;
        }
        // Swap
        const tmp = topK[i]; topK[i] = topK[maxIdx]; topK[maxIdx] = tmp;
        if (i === k - 1) threshold = topK[i];
      }

      this._pool.release(topK);

      // Zero out logits below threshold
      for (let i = 0; i < V; i++) {
        if (logits[i] < threshold) logits[i] = -Infinity;
      }
    }

    // Softmax
    this._softmax(logits, V);

    // Sample from the distribution
    let r = Math.random();
    for (let i = 0; i < V; i++) {
      r -= logits[i];
      if (r <= 0) return i;
    }
    return V - 1; // fallback
  }

  /**
   * Reset the KV cache and sequence length (call before a new generation).
   */
  reset() {
    for (const cache of this._kvCache) {
      cache.k.fill(0);
      cache.v.fill(0);
    }
    this._seqLen = 0;
    this._log("KV cache reset");
  }

  /**
   * Run autoregressive generation for a prompt, returning an array of token ids.
   *
   * @param {number[]}  promptTokens   – initial token sequence
   * @param {number}    maxNewTokens   – maximum tokens to generate
   * @param {Function}  [onToken]      – optional callback(tokenId: number) per generated token
   * @returns {number[]} generated token ids (excluding the prompt)
   */
  generate(promptTokens, maxNewTokens, onToken) {
    if (!Array.isArray(promptTokens) || promptTokens.length === 0) {
      throw new TypeError("promptTokens must be a non-empty array of token ids");
    }
    if (!Number.isInteger(maxNewTokens) || maxNewTokens <= 0) {
      throw new TypeError("maxNewTokens must be a positive integer");
    }

    this._timeStart("generate");
    this.reset();

    const generated = [];

    // Process prompt tokens (teacher-forcing)
    for (let i = 0; i < promptTokens.length; i++) {
      this._timeStart(`forward:${i}`);
      this._forward(promptTokens[i], i);
      this._timeEnd(`forward:${i}`);
    }
    this._seqLen = promptTokens.length;

    // Autoregressive generation
    let pos = this._seqLen - 1;
    let lastLogits = this._scratch.logits.slice(); // copy after last prompt token
    let nextToken = this._sample(lastLogits);

    for (let step = 0; step < maxNewTokens; step++) {
      generated.push(nextToken);
      if (onToken) onToken(nextToken);

      pos++;
      if (pos >= this._maxSeqLen) {
        this._log(`Reached maxSeqLen (${this._maxSeqLen}), stopping generation`);
        break;
      }

      this._timeStart(`forward:gen:${step}`);
      this._forward(nextToken, pos);
      this._timeEnd(`forward:gen:${step}`);

      lastLogits = this._scratch.logits.slice();
      nextToken  = this._sample(lastLogits);
    }

    this._seqLen = pos + 1;
    this._timeEnd("generate");
    this._log(`Generated ${generated.length} tokens`);
    return generated;
  }

  /**
   * Prepare a batch-processing context (placeholder for future SIMD/WebGPU backend).
   * Currently processes sequences sequentially using the single-token engine.
   *
   * @param {number[][]} batchPrompts   – array of token-id arrays
   * @param {number}     maxNewTokens
   * @returns {number[][]} array of generated token arrays
   */
  generateBatch(batchPrompts, maxNewTokens) {
    if (!Array.isArray(batchPrompts)) {
      throw new TypeError("batchPrompts must be an array of token arrays");
    }
    const results = [];
    for (let i = 0; i < batchPrompts.length; i++) {
      this._log(`Batch item ${i + 1}/${batchPrompts.length}`);
      results.push(this.generate(batchPrompts[i], maxNewTokens));
    }
    return results;
  }

  /**
   * Free all pre-allocated scratch memory and pool buffers.
   * Call when this engine instance is no longer needed.
   */
  dispose() {
    this._pool.clear();
    // Clear KV cache references to help GC
    for (const cache of this._kvCache) {
      cache.k = null;
      cache.v = null;
    }
    this._log("TransformerEngine disposed");
  }
}

/* =========================================================
 * §6  Exports (works in browser globals and CommonJS)
 * ========================================================= */

const _exports = {
  createManifest,
  loadModel,
  TransformerEngine,
  Float32ArrayPool,
  // Low-level utilities (useful for testing)
  normalizeRange,
  makeFloat32View,
  buildLayoutInfo,
  validateManifest
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = _exports;
} else if (typeof globalThis !== "undefined") {
  Object.assign(globalThis, _exports);
}

/* =========================================================
 * §7  Example generation flow
 * ========================================================= */
/*
 * Example usage (async context required):
 *
 *   const model = await loadModel(
 *     "/model/config.json",
 *     "/model/manifest.json",
 *     "/model/weights.bin"
 *   );
 *
 *   const engine = new TransformerEngine(model, {
 *     temperature: 0.8,
 *     topK:        40,
 *     maxSeqLen:   512,
 *     profiling:   false,
 *     logger:      (msg) => console.log("[model_runtime]", msg)
 *   });
 *
 *   // Tokenise your prompt however your tokeniser works:
 *   const promptTokens = [1, 15043, 29892, 3186, 29991]; // "Hello, world!"
 *
 *   const generated = engine.generate(
 *     promptTokens,
 *     128,                         // max new tokens
 *     (tok) => process.stdout.write(String(tok) + " ")
 *   );
 *
 *   console.log("\nGenerated tokens:", generated);
 *
 *   // Batch example (sequential under the hood, ready for SIMD/WebGPU upgrade):
 *   const results = engine.generateBatch(
 *     [[1, 29987], [1, 15043]],
 *     64
 *   );
 *
 *   engine.dispose(); // free memory when done
 */
