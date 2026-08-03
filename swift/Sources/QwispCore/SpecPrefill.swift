// ─────────────────────────────────────────────────────────────────────────────
// SpecPrefill (OMLX-style attention-importance sparse prefill) — scoring side.
//
// Pipeline (mirrors omlx/patches/specprefill.py, algorithm from
// arxiv.org/abs/2502.02789):
//   1. Run the 6-layer DFlash draft as a plain autoregressive LM over the NEW
//      conversation tokens (embed from the target engine, causal self-attn
//      over its own KV, target lm_head for the lookahead).
//   2. 8 greedy lookahead decodes, capturing the post-RoPE query vectors at
//      all 6 layers.
//   3. Importance = mean over lookahead tokens of max over (layers × heads)
//      of avgpool13(softmax(Q_lookahead · K_promptᵀ / √d)).
//   4. select_chunks: 32-token chunk top-keep_pct → the token indices the
//      sparse target prefill must keep.
//
// NOTE on the draft conditioning: OMLX's SpecPrefill draft wrapper is
// unverified (its z-lab draft fails to load — rope_theta arg bug — so OMLX's
// specprefill never ran on this model). We score with the clean standard
// self-attention LM forward (no fc context projection — the fc maps the
// TARGET's hiddens, which do not exist at scoring time, and projecting the
// embed through it is a fixed linear map on top of the same stream).
// ─────────────────────────────────────────────────────────────────────────────

import MLX
import MLXFast

public enum SpecPrefill {
    public static let nHeads = DFlashDraftModel.nHeads       // 32
    public static let nKVHeads = DFlashDraftModel.nKVHeads   // 8
    public static let headDim = DFlashDraftModel.headDim     // 128
    public static let scale = DFlashDraftModel.scale
    public static let H = DFlashDraftModel.H                 // 2048

    /// Per-layer draft KV cache (post-RoPE keys at absolute positions 0..len).
    public struct LayerKV {
        public var k: MLXArray     // [1, 8, T, 128]
        public var v: MLXArray     // [1, 8, T, 128]
        public var len: Int
        public init() {
            k = MLXArray.zeros([1, SpecPrefill.nKVHeads, 0, SpecPrefill.headDim])
            v = MLXArray.zeros([1, SpecPrefill.nKVHeads, 0, SpecPrefill.headDim])
            len = 0
        }
    }

    static func rope(_ x: MLXArray, _ offset: Int) -> MLXArray {
        MLXFast.RoPE(x, dimensions: headDim, traditional: false,
                     base: DFlashDraftModel.ropeTheta, scale: 1.0, offset: offset)
    }

    /// One draft layer over a block of L tokens starting at absolute `offset`.
    /// Standard causal self-attention; the full layer (sliding == false) is
    /// bidirectional (dflash_mlx sliding_window=None → mask=None).
    static func forwardChunk(_ layer: DFlashDraftModel.LayerW, x: MLXArray, kv: inout LayerKV,
                             offset: Int, L: Int) -> MLXArray {
        let B = 1
        let normed = MLXFast.rmsNorm(x, weight: layer.inputLN, eps: DFlashDraftModel.eps)   // [L, 2048]
        let qo = MLX.matmul(normed, layer.qW.transposed()).reshaped([B, L, nHeads, headDim])
        var q = MLXFast.rmsNorm(qo, weight: layer.qNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
        q = rope(q, offset)
        let kn = MLX.matmul(normed, layer.kW.transposed()).reshaped([B, L, nKVHeads, headDim])
        var kNew = MLXFast.rmsNorm(kn, weight: layer.kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
        kNew = rope(kNew, offset)
        let vNew = MLX.matmul(normed, layer.vW.transposed()).reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        let keys = kv.len > 0 ? MLX.concatenated([kv.k, kNew], axis: 2) : kNew
        let values = kv.len > 0 ? MLX.concatenated([kv.v, vNew], axis: 2) : vNew

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if layer.sliding {
            let qPos = MLXArray((0 ..< L).map { Int32(offset + $0) }).reshaped([L, 1])
            let kPos = MLXArray((0 ..< offset + L).map { Int32($0) }).reshaped([1, offset + L])
            let causal = kPos .<= qPos
            let window = kPos .< (qPos + Int32(DFlashDraftModel.slidingWindow))
            mask = .array((causal & window).reshaped([1, 1, L, offset + L]))
        } else {
            mask = .none
        }

        let out = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values,
                                                    scale: scale, mask: mask)
        let o = out.transposed(0, 2, 1, 3).reshaped([B, L, nHeads * headDim])
        var h = x + MLX.matmul(o, layer.oW.transposed())
        let post = MLXFast.rmsNorm(h, weight: layer.postLN, eps: DFlashDraftModel.eps)
        let gate = MLX.matmul(post, layer.gateW.transposed())
        let up = MLX.matmul(post, layer.upW.transposed())
        let mlp = MLX.matmul(MLX.multiply(MLX.multiply(gate, MLX.sigmoid(gate)), up), layer.downW.transposed())
        h = h + mlp
        kv.k = keys; kv.v = values; kv.len += L
        return h
    }

    /// Prefill all tokens through the draft LM. Returns the per-layer KV caches
    /// and the final-row logits [1, vocab] (for the first lookahead step).
    static func prefill(tokens: [Int32], draft: DFlashDraftModel, engine: SeedlessEngine,
                        chunkLen: Int) -> ([LayerKV], MLXArray) {
        var kvs = (0 ..< DFlashDraftModel.nLayers).map { _ in LayerKV() }
        var offset = 0
        var lastNormed: MLXArray? = nil
        var i = 0
        while i < tokens.count {
            let L = min(chunkLen, tokens.count - i)
            let block = Array(tokens[i ..< i + L])
            var h = engine.embed(tokens: block).reshaped([1, L, H])
            for li in 0 ..< DFlashDraftModel.nLayers {
                h = forwardChunk(draft.layers[li], x: h, kv: &kvs[li], offset: offset, L: L)
            }
            h = MLXFast.rmsNorm(h, weight: draft.normW, eps: DFlashDraftModel.eps)   // [1, L, 2048]
            MLX.eval([h])
            lastNormed = h
            offset += L
            i += L
        }
        guard let h = lastNormed else { return (kvs, MLXArray.zeros([1, 1])) }
        let lastRow = h[0..., (h.dim(1) - 1)..., 0...].reshaped([1, H])
        guard let lg = engine.logits(lastRow, M: 1) else { return (kvs, MLXArray.zeros([1, 1])) }
        MLX.eval([lg])
        return (kvs, lg)
    }

    /// nSteps greedy lookahead decodes, capturing post-RoPE queries.
    /// Returns queriesByLayer[li] = [ [1, 32, 1, 128] × nSteps ].
    static func lookahead(kvs: inout [LayerKV], draft: DFlashDraftModel, engine: SeedlessEngine,
                          firstToken: Int32, nSteps: Int, offset: Int) -> [[MLXArray]] {
        var qbuf: [[MLXArray]] = (0 ..< DFlashDraftModel.nLayers).map { _ in [] }
        var tok = firstToken
        var pos = offset
        for _ in 0 ..< nSteps {
            var h = engine.embed(tokens: [tok]).reshaped([1, 1, H])
            for li in 0 ..< DFlashDraftModel.nLayers {
                let L = 1
                let layer = draft.layers[li]
                let normed = MLXFast.rmsNorm(h, weight: layer.inputLN, eps: DFlashDraftModel.eps)
                let qo = MLX.matmul(normed, layer.qW.transposed()).reshaped([1, 1, nHeads, headDim])
                var q = MLXFast.rmsNorm(qo, weight: layer.qNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                q = rope(q, pos)
                qbuf[li].append(q)
                let kn = MLX.matmul(normed, layer.kW.transposed()).reshaped([1, 1, nKVHeads, headDim])
                var kNew = MLXFast.rmsNorm(kn, weight: layer.kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                kNew = rope(kNew, pos)
                let vNew = MLX.matmul(normed, layer.vW.transposed()).reshaped([1, 1, nKVHeads, headDim]).transposed(0, 2, 1, 3)
                let keys = MLX.concatenated([kvs[li].k, kNew], axis: 2)
                let values = MLX.concatenated([kvs[li].v, vNew], axis: 2)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode
                if layer.sliding {
                    let qPos = MLXArray([Int32(pos)]).reshaped([1, 1])
                    let kPos = MLXArray((0 ... pos).map { Int32($0) }).reshaped([1, pos + 1])
                    let causal = kPos .<= qPos
                    let window = kPos .< (qPos + Int32(DFlashDraftModel.slidingWindow))
                    mask = .array((causal & window).reshaped([1, 1, 1, pos + 1]))
                } else {
                    mask = .none
                }
                let out = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values,
                                                            scale: scale, mask: mask)
                let o = out.transposed(0, 2, 1, 3).reshaped([1, 1, nHeads * headDim])
                h = h + MLX.matmul(o, layer.oW.transposed())
                let post = MLXFast.rmsNorm(h, weight: layer.postLN, eps: DFlashDraftModel.eps)
                let gate = MLX.matmul(post, layer.gateW.transposed())
                let up = MLX.matmul(post, layer.upW.transposed())
                let mlp = MLX.matmul(MLX.multiply(MLX.multiply(gate, MLX.sigmoid(gate)), up), layer.downW.transposed())
                h = h + mlp
                kvs[li].k = keys; kvs[li].v = values; kvs[li].len += 1
            }
            h = MLXFast.rmsNorm(h, weight: draft.normW, eps: DFlashDraftModel.eps)
            let row = h[0..., 0..., 0...].reshaped([1, H])
            guard let lg = engine.logits(row, M: 1) else { break }
            MLX.eval([lg])
            tok = Int32(MLX.argMax(lg, axis: -1).item(Int.self))
            pos += 1
        }
        return qbuf
    }

    /// avg_pool1d along the last axis (prefix-sum trick, mirrors the reference).
    static func avgPool1d(_ x: MLXArray, kernel: Int) -> MLXArray {
        let pad = kernel / 2
        let nd = x.ndim
        var widths: [IntOrPair] = Array(repeating: .init((0, 0)), count: nd)
        widths[nd - 1] = .init((pad, pad))
        let padded = MLX.padded(x, widths: widths)
        let zeros = MLXArray.zeros(Array(x.shape.dropLast()) + [1])
        let prefix = MLX.concatenated([zeros, padded.cumsum(axis: -1)], axis: -1)
        let T = x.dim(-1)
        let top = prefix[0..., 0..., kernel...]
        let bot = prefix[0..., 0..., 0 ..< T]
        return (top - bot) / Float(kernel)
    }

    /// Per-token importance [T] from the captured lookahead queries and the
    /// prompt KV (pooled softmax, max over layers×heads, mean over lookahead).
    static func importance(kvs: [LayerKV], qbuf: [[MLXArray]], T: Int, poolKernel: Int) -> MLXArray {
        var combined: [MLXArray] = []
        let group = nHeads / nKVHeads
        for li in 0 ..< DFlashDraftModel.nLayers {
            guard !qbuf[li].isEmpty else { continue }
            let qStack = MLX.concatenated(qbuf[li], axis: 2)                    // [1, 32, nL, 128]
            let pKeys = kvs[li].k[0..., 0..., 0 ..< T, 0...]                    // [1, 8, T, 128]
            let expanded = MLX.tiled(pKeys, repetitions: [1, group, 1, 1])      // [1, 32, T, 128]
            let scores = MLX.matmul(qStack, expanded.transposed(0, 1, 3, 2)) * scale
            let w = MLX.softMax(scores, axes: [-1]).squeezed(axis: 0)           // [32, nL, T]
            combined.append(w)
        }
        var c = MLX.concatenated(combined, axis: 0)                             // [L×32, nL, T]
        if poolKernel > 1 { c = avgPool1d(c, kernel: poolKernel) }
        let maxScores = c.max(axis: 0)                                          // [nL, T]
        let imp = maxScores.mean(axis: 0)                                       // [T]
        MLX.eval([imp])
        return imp
    }

    /// Chunk-based top-keep_pct selection (chunk_size=32, sorted output indices).
    public static func selectChunks(importance: MLXArray, keepPct: Float, chunkSize: Int = 32) -> [Int32] {
        let M = importance.dim(0)
        if keepPct >= 1.0 { return Array(0 ..< M).map(Int32.init) }
        let nChunks = (M + chunkSize - 1) / chunkSize
        let keepN = max(1, Int((Double(nChunks) * Double(keepPct)).rounded(.up)))
        let impArr = importance.asArray(Float.self)
        var chunkScores: [Float] = []
        chunkScores.reserveCapacity(nChunks)
        for ci in 0 ..< nChunks {
            let s = ci * chunkSize
            let e = min(s + chunkSize, M)
            var sum: Float = 0
            for t in s ..< e { sum += impArr[t] }
            chunkScores.append(sum / Float(e - s))
        }
        let top = chunkScores.enumerated().sorted { $0.element > $1.element }.prefix(keepN).map { $0.offset }.sorted()
        var out: [Int32] = []
        out.reserveCapacity(keepN * chunkSize)
        for ci in top {
            for t in ci * chunkSize ..< min((ci + 1) * chunkSize, M) { out.append(Int32(t)) }
        }
        return out
    }

    /// Full scoring pipeline. Returns the selected indices (absolute, within the
    /// scored token span), or nil on any failure.
    public static func score(tokens: [Int32], draft: DFlashDraftModel, engine: SeedlessEngine,
                             lookahead: Int = 8, poolKernel: Int = 13, keepPct: Float = 0.2,
                             chunkSize: Int = 32, prefillChunk: Int = 2048) -> [Int32]? {
        let T = tokens.count
        guard T > 1 else { return nil }
        let (kvs, lg) = prefill(tokens: tokens, draft: draft, engine: engine, chunkLen: prefillChunk)
        let firstTok = Int32(MLX.argMax(lg, axis: -1).item(Int.self))
        var kvsMutable = kvs
        let qbuf = SpecPrefill.lookahead(kvs: &kvsMutable, draft: draft, engine: engine,
                             firstToken: firstTok, nSteps: lookahead, offset: T)
        let imp = SpecPrefill.importance(kvs: kvsMutable, qbuf: qbuf, T: T, poolKernel: poolKernel)
        return selectChunks(importance: imp, keepPct: keepPct, chunkSize: chunkSize)
    }
}
