/**
 * model_runtime.js
 *
 * Includes:
 *  - createManifest(config)
 *  - loadModel(configUrl, manifestUrl, weightsUrl)
 *  - TransformerEngine
 *  - Example generation flow
 *
 * Assumptions:
 *  - Weights are stored as float32
 *  - Manifest offsets are in float32 ELEMENTS, not bytes
 *  - Matrix layout is row-major [outDim, inDim]
 *
 * Tensor shapes:
 *  - token_embedding: [vocab_size, hidden_size]
 *  - final_norm: [hidden_size] (optional)
 *  - wq: [qDim, hidden_size]
 *  - wk: [kvDim, hidden_size]
 *  - wv: [kvDim, hidden_size]
 *  - wo: [hidden_size, qDim]
 *  - ffn_gate: [intermediate_size, hidden_size]
 *  - ffn_up: [intermediate_size, hidden_size]
 *  - ffn_down: [hidden_size, intermediate_size]
 *  - output_head: [vocab_size, hidden_size]
 */

/* =========================================================
 * 0) Utility / validation helpers
 * ========================================================= */

/**
 * Assert that a value is a positive integer and return it.
 * Throws a descriptive error when the check fails.
 *
 * @param {*}      value - The value to validate.
 * @param {string} name  - Human-readable name used in the error message.
 * @returns {number} The validated positive integer.
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
 * Normalise a [start, end] offset range, validating that both values are
 * non-negative integers and that start <= end.
 *
 * @param {*} range - Expected to be a two-element numeric array.
 * @returns {[number, number]} Validated [start, end] tuple.
 */
function normalizeRange(range) {
  if (!Array.isArray(range) || range.length !== 2) {
    throw new Error(`Invalid range: ${JSON.stringify(range)}`);
  }

  const [start, end] = range;

  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    throw new TypeError(
      `Range values must be integers, got: [${start}, ${end}]`
    );
  }

  if (start < 0 || end < start) {
    throw new RangeError(
      `Range must satisfy 0 <= start <= end, got: [${start}, ${end}]`
    );
  }

  return [start, end];
}

/**
 * Create a Float32Array view into an ArrayBuffer from a [start, end] range.
 *
 * @param {ArrayBuffer} buffer     - The source buffer.
 * @param {[number, number]} range - Offset range [start, end].
 * @param {"elements"|"bytes"} offsetUnit - Unit for the offsets (default: "elements").
 * @returns {Float32Array} A typed-array view of the requested slice.
 */
function makeFloat32View(buffer, range, offsetUnit = "elements") {
  const [start, end] = normalizeRange(range);

  if (offsetUnit === "elements") {
    const byteOffset = start * 4;
    const length = end - start;
    return new Float32Array(buffer, byteOffset, length);
  }

  if (offsetUnit === "bytes") {
    const byteOffset = start;
    const byteLength = end - start;

    if (byteLength % 4 !== 0) {
      throw new Error(
        `Byte range ${start}-${end} is not aligned to float32 size.`
      );
    }

    return new Float32Array(buffer, byteOffset, byteLength / 4);
  }

  throw new Error(`Unknown offset_unit: ${offsetUnit}`);
}

/* =========================================================
 * 1) Layout computation + manifest validation
 * ========================================================= */

/**
 * Compute the expected tensor shapes for every layer from the model config.
 *
 * @param {object} config - Model configuration object.
 * @returns {{layerShapes: object, outputHeadShape: [number, number]}}
 */
function buildLayoutInfo(config) {
  const hidden       = mustBePositiveInt(config.hidden_size,       "hidden_size");
  const vocab        = mustBePositiveInt(config.vocab_size,        "vocab_size");
  const numHeads     = mustBePositiveInt(config.num_heads,         "num_heads");
  const numKvHeads   = mustBePositiveInt(config.num_kv_heads,      "num_kv_heads");
  const headDim      = mustBePositiveInt(config.head_dim,          "head_dim");
  const intermediate = mustBePositiveInt(config.intermediate_size, "intermediate_size");

  const qDim  = numHeads   * headDim;
  const kvDim = numKvHeads * headDim;

  return {
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
    outputHeadShape: [vocab, hidden]
  };
}

/**
 * Validate that every tensor range in the manifest is consistent with the
 * model config and fits within the loaded weight buffer.
 *
 * @param {object}      config   - Model configuration object.
 * @param {object}      manifest - Parsed manifest JSON.
 * @param {ArrayBuffer} buffer   - Loaded weights buffer.
 * @param {{layerShapes: object, outputHeadShape: number[]}} layout - Pre-computed layout info.
 */
function validateManifest(config, manifest, buffer, layout) {
  const offsetUnit    = manifest.metadata?.offset_unit || "elements";
  const bufferFloats  = buffer.byteLength / 4;
  const bufferBytes   = buffer.byteLength;

  /**
   * Assert that a [start, end] range fits within the buffer.
   * @param {string} tensorName
   * @param {[number, number]} range
   */
  function assertRangeInBuffer(tensorName, range) {
    const [start, end] = normalizeRange(range);
    const limit = offsetUnit === "bytes" ? bufferBytes : bufferFloats;
    if (end > limit) {
      throw new RangeError(
        `Tensor "${tensorName}" range [${start}, ${end}] exceeds buffer size (${limit} ${offsetUnit}).`
      );
    }
  }

  assertRangeInBuffer("token_embedding", manifest.token_embedding);

  if (manifest.final_norm) {
    assertRangeInBuffer("final_norm", manifest.final_norm);
  }

  if (!Array.isArray(manifest.layers)) {
    throw new Error("Manifest is missing a layers array.");
  }

  const expectedLayers = mustBePositiveInt(config.num_layers, "num_layers");
  if (manifest.layers.length !== expectedLayers) {
    throw new Error(
      `Manifest has ${manifest.layers.length} layers but config specifies ${expectedLayers}.`
    );
  }

  for (const layer of manifest.layers) {
    const o = layer.offsets;
    for (const key of Object.keys(layout.layerShapes)) {
      if (!o[key]) {
        throw new Error(
          `Layer ${layer.layer_id} is missing offset for "${key}".`
        );
      }
      assertRangeInBuffer(`layer[${layer.layer_id}].${key}`, o[key]);
    }
  }

  assertRangeInBuffer("output_head", manifest.output_head);
}

/* =========================================================
 * 2) Manifest generator
 * ========================================================= */

/**
 * Generate a complete weight-manifest object for a given model config.
 * All offsets are in float32 elements (not bytes).
 *
 * @param {object} config - Model configuration object with the following fields:
 *   @param {number} config.hidden_size
 *   @param {number} config.vocab_size
 *   @param {number} config.num_layers
 *   @param {number} config.num_heads
 *   @param {number} config.num_kv_heads
 *   @param {number} config.head_dim
 *   @param {number} config.intermediate_size
 * @returns {object} Manifest describing the layout of the flat weights file.
 */
function createManifest(config) {
  const hidden       = mustBePositiveInt(config.hidden_size,       "hidden_size");
  const vocab        = mustBePositiveInt(config.vocab_size,        "vocab_size");
  const numLayers    = mustBePositiveInt(config.num_layers,        "num_layers");
  const numHeads     = mustBePositiveInt(config.num_heads,         "num_heads");
  const numKvHeads   = mustBePositiveInt(config.num_kv_heads,      "num_kv_heads");
  const headDim      = mustBePositiveInt(config.head_dim,          "head_dim");
  const intermediate = mustBePositiveInt(config.intermediate_size, "intermediate_size");

  const qDim  = numHeads   * headDim;
  const kvDim = numKvHeads * headDim;

  let cursor = 0;

  /**
   * Allocate a contiguous float32 region and advance the cursor.
   * @param {number} numFloatElements
   * @returns {[number, number]} [start, end] in float32 elements.
   */
  function alloc(numFloatElements) {
    const start = cursor;
    cursor += numFloatElements;
    return [start, cursor];
  }

  const manifest = {
    metadata: {
      format:           "float32",
      offset_unit:      "elements",
      total_size_bytes: 0
    },

    token_embedding: alloc(vocab * hidden),
    final_norm:      alloc(hidden),

    layers: [],

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
        ffn_down:   alloc(hidden * intermediate),
        ffn_up:     alloc(intermediate * hidden)
      }
    });
  }

  manifest.output_head            = alloc(vocab * hidden);
  manifest.metadata.total_size_bytes = cursor * 4;

  return manifest;
}

/* =========================================================
 * 3) Network helpers
 * ========================================================= */

/**
 * Fetch and parse a JSON resource.
 *
 * @param {string} url
 * @returns {Promise<object>}
 */
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch JSON: ${url} (${res.status})`);
  }
  return res.json();
}

/**
 * Fetch a binary resource as an ArrayBuffer.
 *
 * @param {string} url
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchArrayBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch weights: ${url} (${res.status})`);
  }
  return res.arrayBuffer();
}

/* =========================================================
 * 4) Model loader
 * ========================================================= */

/**
 * Load a model from remote URLs.
 * Fetches config, manifest, and weight buffer in parallel, then builds
 * typed-array tensor views into the weight buffer (zero-copy).
 *
 * @param {string} configUrl   - URL of the model config JSON.
 * @param {string} manifestUrl - URL of the weight manifest JSON.
 * @param {string} weightsUrl  - URL of the raw float32 weights file.
 * @returns {Promise<object>} Model object with all tensors attached.
 */
async function loadModel(configUrl, manifestUrl, weightsUrl) {
  const [config, manifest, buffer] = await Promise.all([
    fetchJson(configUrl),
    fetchJson(manifestUrl),
    fetchArrayBuffer(weightsUrl)
  ]);

  const layout     = buildLayoutInfo(config);
  validateManifest(config, manifest, buffer, layout);

  const offsetUnit = manifest.metadata?.offset_unit || "elements";

  // Memoized tensor factory — wraps makeFloat32View with a name for debugging.
  const tensorCache = new Map();

  /**
   * Create (or return a cached) Float32Array tensor view.
   *
   * @param {string} name
   * @param {[number, number]} range
   * @param {number[]} shape
   * @returns {{ name: string, shape: number[], data: Float32Array }}
   */
  function makeTensor(name, range, shape) {
    const key = `${name}:${range[0]}-${range[1]}`;
    if (!tensorCache.has(key)) {
      tensorCache.set(key, {
        name,
        shape,
        data: makeFloat32View(buffer, range, offsetUnit)
      });
    }
    return tensorCache.get(key);
  }

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
      const o  = layer.offsets;
      const ls = layout.layerShapes;
      const id = layer.layer_id ?? index;

      return {
        layer_id:   id,
        rms_norm_1: makeTensor(`layer${id}.rms_norm_1`, o.rms_norm_1, ls.rms_norm_1),
        wq:         makeTensor(`layer${id}.wq`,         o.wq,         ls.wq),
        wk:         makeTensor(`layer${id}.wk`,         o.wk,         ls.wk),
        wv:         makeTensor(`layer${id}.wv`,         o.wv,         ls.wv),
        wo:         makeTensor(`layer${id}.wo`,         o.wo,         ls.wo),
        rms_norm_2: makeTensor(`layer${id}.rms_norm_2`, o.rms_norm_2, ls.rms_norm_2),
        ffn_gate:   makeTensor(`layer${id}.ffn_gate`,   o.ffn_gate,   ls.ffn_gate),
        ffn_down:   makeTensor(`layer${id}.ffn_down`,   o.ffn_down,   ls.ffn_down),
        ffn_up:     makeTensor(`layer${id}.ffn_up`,     o.ffn_up,     ls.ffn_up)
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
 * 5) Module exports (CommonJS + ESM compatible)
 * ========================================================= */

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    createManifest,
    loadModel,
    buildLayoutInfo,
    validateManifest,
    makeFloat32View,
    normalizeRange,
    mustBePositiveInt
  };
}
