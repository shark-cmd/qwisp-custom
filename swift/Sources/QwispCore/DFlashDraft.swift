import Foundation
import MLX
import MLXFast

// ─────────────────────────────────────────────────────────────────────────────
// DFlash block-diffusion draft model (z-lab/Qwen3.6-35B-A3B-DFlash) — Swift port
// of dflash_mlx (oMLX.app). Gated by QWISP_DFLASH=1 (+ QWISP_DFLASH_DIR).
//
// The draft is a 6-layer qwen3-style model with NO embed/lm_head of its own: it
// reuses the TARGET model's token embedding and LM head. It conditions on the
// target's hidden states captured at layers [1,6,11,16,22,27,32,37] (1-based),
// concatenated → fc [2048, 16384] → rms_norm (the "projected context").
//
// Per cycle: block = [lastToken, mask×15] → embed via target → ONE draft forward
// (5 sliding + 1 full attention layers; context K/V from the projected features,
// noise K/V from the block) → target lm_head logits → greedy block of 15 tokens.
//
// Reference semantics (dflash_mlx/model.py, ContextOnlyDraftKVCache branch):
//   queries = rope(block, offset=query_offset)
//   context keys roped at their own absolute positions (window tail)
//   noise keys roped at query_offset
//   sliding layers: mask = (queryPos >= keyPos) & (queryPos < keyPos + 4096)
//   full layer: mask = None (block is jointly denoised, bidirectional)
// ─────────────────────────────────────────────────────────────────────────────

/// Rolling context window for the DFlash draft: the last ≤1024 tokens' target
/// hidden states at the 8 capture layers, plus the capture readback plumbing.
/// Lives on the backend; persists across requests (mirrors the arena KV).
public final class DFlashContext {
    public static let windowRows = 1024
    /// Target layers to capture, 0-based. The config's target_layer_ids are 1-based
    /// [1,6,11,16,22,27,32,37]; dflash_mlx's runtime captures `captured[layer_index+1]`
    /// (target_qwen_gdn.py forward_with_hidden_capture) and reads back `layer_id + 1`
    /// (extract_context_feature) → the effective 0-based capture layers are
    /// [1,6,11,16,22,27,32,37] (ONE LAYER LATER than a naive 1-based→0-based mapping).
    public static let captureTargets: [Int] = [1, 6, 11, 16, 22, 27, 32, 37]

    /// Gate for the fused-forward capture hook. Set by the prefill tail gate and
    /// left on for the spec loop (every verify captures).
    public var captureEnabled = false
    /// Readback of the LAST forward's captured hidden: li → [M, H] fp16.
    public var readback: [Int: MLXArray] = [:]

    // Rolling window: per-layer chunk list (append-only; trim front past windowRows).
    private var chunks: [Int: [MLXArray]] = [:]
    private var chunkRows: [Int: Int] = [:]
    /// Total tokens seen (= current KV length at the last append). The draft's
    /// query_offset = this value.
    public private(set) var count = 0

    public init() {}

    public func reset() {
        chunks = [:]; chunkRows = [:]; readback = [:]; captureEnabled = false; count = 0
    }

    /// Append the first `rowCount` rows of the current readback to the window
    /// (trimmed to the last windowRows). Caller guarantees the rows correspond to
    /// tokens actually committed to the KV.
    public func appendReadback(rowCount: Int) {
        let take = max(0, rowCount)
        guard take > 0 else { return }
        if ProcessInfo.processInfo.environment["QWISP_DFLASH_DEBUG"] == "1" {
            let keys = readback.keys.sorted().map { String($0) }.joined(separator: ",")
            let h0 = readback[Self.captureTargets[0]].flatMap { rb -> String in
                let flat = rb[0, 0 ..< 4].reshaped([-1]).asArray(Float16.self)
                return flat.map { String(format: "%.3f", Float($0)) }.joined(separator: ",")
            } ?? "?"
            FileHandle.standardError.write(Data("[dflash-capture] take=\(take) keys=[\(keys)] h0=[\(h0)]\n".utf8))
        }
        for li in Self.captureTargets {
            guard let rb = readback[li] else { continue }
            let m = rb.dim(0)
            guard m > 0 else { continue }
            let n = min(take, m)
            let part = (n < m) ? rb[0 ..< n, 0...] : rb
            var cs = chunks[li] ?? []
            var total = chunkRows[li] ?? 0
            cs.append(part)
            total += n
            while total > Self.windowRows, let f = cs.first {
                cs.removeFirst()
                total -= f.dim(0)
            }
            chunks[li] = cs
            chunkRows[li] = total
        }
        count += take
    }

    /// Roll back to a previous count (mirrors the KV snapshot/rollback in the
    /// spec loop's partial-reject path).
    public func truncate(to target: Int) {
        guard target < count else { return }
        let drop = count - target
        for li in Self.captureTargets {
            var cs = chunks[li] ?? []
            var total = chunkRows[li] ?? 0
            var rem = drop
            while rem > 0, let f = cs.first {
                if f.dim(0) <= rem {
                    cs.removeFirst()
                    total -= f.dim(0)
                    rem -= f.dim(0)
                } else {
                    cs[0] = f[rem..., 0...]
                    total -= rem
                    rem = 0
                }
            }
            chunks[li] = cs
            chunkRows[li] = total
        }
        count = target
    }

    /// The window features in capture-layer order, 8 × [T, 2048] fp16 (T ≤ 1024).
    /// Empty array if any layer is missing rows (context not ready).
    public func features() -> [MLXArray] {
        var out: [MLXArray] = []
        out.reserveCapacity(Self.captureTargets.count)
        for li in Self.captureTargets {
            guard let cs = chunks[li], !cs.isEmpty else { return [] }
            out.append(cs.count == 1 ? cs[0] : MLX.concatenated(cs, axis: 0))
        }
        return out
    }
}

/// Loaded DFlash draft weights + inference (pure MLX; runs ~once per spec cycle).
public struct DFlashDraftModel {
    public static let blockSize = 16
    public static let maskTokenID: Int32 = 248077
    public static let H = 2048
    public static let nHeads = 32
    public static let nKVHeads = 8
    public static let headDim = 128
    public static let ropeTheta: Float = 10_000_000
    public static let eps: Float = 1e-6
    public static let slidingWindow = 4096
    public static let nLayers = 6
    public static let layerSliding: [Bool] = [true, true, true, true, true, false]
    /// scale = headDim^-0.5 (matches dflash_mlx self.scale)
    public static let scale: Float = Float(pow(Double(headDim), -0.5))

    public struct LayerW {
        let inputLN, postLN: MLXArray      // [2048]
        let qW, kW, vW, oW: MLXArray       // [4096,2048] [1024,2048] [1024,2048] [2048,4096]
        let qNorm, kNorm: MLXArray         // [128]
        let gateW, upW, downW: MLXArray    // [6144,2048] [6144,2048] [2048,6144]
        let sliding: Bool

        fileprivate func rope(_ x: MLXArray, _ offset: Int) -> MLXArray {
            MLXFast.RoPE(x, dimensions: DFlashDraftModel.headDim, traditional: false,
                         base: DFlashDraftModel.ropeTheta, scale: 1.0, offset: offset)
        }

        /// One draft attention layer: q from block, context K/V from the projected
        /// target features, noise K/V from the block. Mirrors dflash_mlx/model.py
        /// `__call__` (ContextOnlyDraftKVCache branch).
        fileprivate func forwardAttention(_ normed: MLXArray, ctx: MLXArray,
                                          ctxPositions: MLXArray, queryOffset: Int) -> MLXArray {
            let B = 1, L = normed.dim(1), T = ctx.dim(1)
            let D = DFlashDraftModel.headDim

            let qo = MLX.matmul(normed, qW.transposed()).reshaped([B, L, DFlashDraftModel.nHeads, D])
            var queries = MLXFast.rmsNorm(qo, weight: qNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
            queries = rope(queries, queryOffset)

            // Context K/V: projected target features at their absolute positions
            // (contiguous tail window → single rope offset at the window start).
            let kc = MLX.matmul(ctx, kW.transposed()).reshaped([B, T, DFlashDraftModel.nKVHeads, D])
            var kCtx = MLXFast.rmsNorm(kc, weight: kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
            kCtx = rope(kCtx, queryOffset - T)
            let vCtx = MLX.matmul(ctx, vW.transposed()).reshaped([B, T, DFlashDraftModel.nKVHeads, D]).transposed(0, 2, 1, 3)

            // Noise K/V: the block rows, roped at query_offset.
            let kn = MLX.matmul(normed, kW.transposed()).reshaped([B, L, DFlashDraftModel.nKVHeads, D])
            var kNoise = MLXFast.rmsNorm(kn, weight: kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
            kNoise = rope(kNoise, queryOffset)
            let vNoise = MLX.matmul(normed, vW.transposed()).reshaped([B, L, DFlashDraftModel.nKVHeads, D]).transposed(0, 2, 1, 3)

            let keys = MLX.concatenated([kCtx, kNoise], axis: 2)    // [1, 8, T+L, 128]
            let values = MLX.concatenated([vCtx, vNoise], axis: 2)  // [1, 8, T+L, 128]

            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if sliding {
                // (queryPos >= keyPos) & (queryPos < keyPos + sliding_window)
                let qPos = MLXArray((0 ..< L).map { Int32(queryOffset + $0) }).reshaped([L, 1])
                let kPos = MLX.concatenated(
                    [ctxPositions, MLXArray((0 ..< L).map { Int32(queryOffset + $0) })],
                    axis: 0).reshaped([1, T + L])
                let causal = kPos .<= qPos                       // [L, T+L] broadcast
                let window = kPos .< (qPos + Int32(DFlashDraftModel.slidingWindow))
                mask = .array((causal & window).reshaped([1, 1, L, T + L]))
            } else {
                // Full-attention layer: the block is jointly denoised (mask = None).
                mask = .none
            }

            let out = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: DFlashDraftModel.scale, mask: mask)
            let o = out.transposed(0, 2, 1, 3).reshaped([B, L, DFlashDraftModel.nHeads * D])
            return MLX.matmul(o, oW.transposed())
        }

        fileprivate func forwardLayer(_ x: MLXArray, ctx: MLXArray, ctxPositions: MLXArray,
                                      queryOffset: Int) -> MLXArray {
            let normed = MLXFast.rmsNorm(x, weight: inputLN, eps: DFlashDraftModel.eps)
            let attn = forwardAttention(normed, ctx: ctx, ctxPositions: ctxPositions, queryOffset: queryOffset)
            let h = x + attn
            let post = MLXFast.rmsNorm(h, weight: postLN, eps: DFlashDraftModel.eps)
            let gate = MLX.matmul(post, gateW.transposed())
            let up = MLX.matmul(post, upW.transposed())
            let mlp = MLX.matmul(MLX.multiply(MLX.multiply(gate, MLX.sigmoid(gate)), up), downW.transposed())
            return h + mlp
        }
    }

    public let fcW: MLXArray          // [2048, 16384]
    public let hiddenNormW: MLXArray  // [2048]
    public let normW: MLXArray        // [2048]
    public let layers: [LayerW]

    /// Load the BF16 draft safetensors. Returns nil on any failure.
    public static func load(draftDir: String) -> DFlashDraftModel? {
        let url = URL(fileURLWithPath: draftDir).appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path),
              let w = try? loadArrays(url: url) else { return nil }
        func g(_ k: String) -> MLXArray? { w[k] }
        guard let fcW = g("fc.weight"), let hn = g("hidden_norm.weight"), let nrm = g("norm.weight") else { return nil }
        var layers: [LayerW] = []
        for i in 0 ..< nLayers {
            let p = "layers.\(i)."
            guard let inputLN = g(p + "input_layernorm.weight"),
                  let postLN = g(p + "post_attention_layernorm.weight"),
                  let qW = g(p + "self_attn.q_proj.weight"),
                  let kW = g(p + "self_attn.k_proj.weight"),
                  let vW = g(p + "self_attn.v_proj.weight"),
                  let oW = g(p + "self_attn.o_proj.weight"),
                  let qN = g(p + "self_attn.q_norm.weight"),
                  let kN = g(p + "self_attn.k_norm.weight"),
                  let gateW = g(p + "mlp.gate_proj.weight"),
                  let upW = g(p + "mlp.up_proj.weight"),
                  let downW = g(p + "mlp.down_proj.weight") else { return nil }
            layers.append(LayerW(inputLN: inputLN, postLN: postLN, qW: qW, kW: kW, vW: vW, oW: oW,
                                 qNorm: qN, kNorm: kN, gateW: gateW, upW: upW, downW: downW,
                                 sliding: layerSliding[i]))
        }
        return DFlashDraftModel(fcW: fcW, hiddenNormW: hn, normW: nrm, layers: layers)
    }

    /// Project the 8 captured target hidden states → [1, T, 2048] draft context
    /// (fc over the concatenated features + hidden rms_norm). Activations stay fp32
    /// (bf16 weights promote) — matches dflash_mlx exactly (its Linear weights are bf16).
    public func projectTargetFeatures(_ feats: [MLXArray]) -> MLXArray {
        let cat = MLX.concatenated(feats, axis: -1)                               // [T, 16384]
        let proj = MLX.matmul(cat, fcW.transposed())                              // [T, 2048]
        return MLXFast.rmsNorm(proj, weight: hiddenNormW, eps: DFlashDraftModel.eps)
            .reshaped([1, feats[0].dim(0), DFlashDraftModel.H])
    }

    /// One draft forward over the block (all 6 layers + final norm).
    /// - blockHidden: [1, L, 2048] (embed of [u, mask×15])
    /// - ctx: [1, T, 2048] (projected target features)
    /// - ctxPositions: [T] int32 absolute positions of the context rows
    /// - queryOffset: absolute position of the block's first row (= KV length)
    public func forwardBlock(_ blockHidden: MLXArray, ctx: MLXArray, ctxPositions: MLXArray,
                             queryOffset: Int) -> MLXArray {
        var h = blockHidden
        for li in 0 ..< DFlashDraftModel.nLayers {
            h = layers[li].forwardLayer(h, ctx: ctx, ctxPositions: ctxPositions, queryOffset: queryOffset)
        }
        return MLXFast.rmsNorm(h, weight: normW, eps: DFlashDraftModel.eps)
    }

    /// Draft a block of blockSize-1 tokens after `u`, using the current window.
    /// Returns nil when the context window isn't ready (caller falls back to greedy).
    public func draftBlock(u: Int32, ctx: DFlashContext, engine: SeedlessEngine) -> [Int]? {
        let baseLen = ctx.count
        let feats = ctx.features()
        guard !feats.isEmpty else { return nil }
        let T = feats[0].dim(0)
        guard T > 0 else { return nil }
        let dctx = projectTargetFeatures(feats)                                    // [1, T, 2048]
        let ctxPositions = MLXArray(Array(Int32(baseLen - T) ..< Int32(baseLen)))  // [T]
        let blockToks: [Int32] = [u] + Array(repeating: DFlashDraftModel.maskTokenID,
                                             count: DFlashDraftModel.blockSize - 1)
        let emb = engine.embed(tokens: blockToks)                                  // [16, 2048]
        let h0 = emb.reshaped([1, DFlashDraftModel.blockSize, DFlashDraftModel.H])
        let h = forwardBlock(h0, ctx: dctx, ctxPositions: ctxPositions, queryOffset: baseLen)
        let blockH = h[0..., 1..., 0...].reshaped([DFlashDraftModel.blockSize - 1, DFlashDraftModel.H])
        guard let lg = engine.logits(blockH, M: DFlashDraftModel.blockSize - 1) else { return nil }
        MLX.eval([lg])
        var out: [Int] = []
        out.reserveCapacity(DFlashDraftModel.blockSize - 1)
        for i in 0 ..< (DFlashDraftModel.blockSize - 1) {
            out.append(MLX.argMax(lg[i], axis: -1).item(Int.self))
        }
        // QWISP_DFLASH_DUMP=1: write the first cycle's features/block-embed/hidden for
        // cross-checking against the reference dflash_mlx/model.py forward.
        if ProcessInfo.processInfo.environment["QWISP_DFLASH_DUMP"] == "1",
           !FileManager.default.fileExists(atPath: "/tmp/dflash_dump/done") {
            try? FileManager.default.createDirectory(atPath: "/tmp/dflash_dump", withIntermediateDirectories: true)
            func dump(_ name: String, _ x: MLXArray) {
                MLX.eval([x])
                let a = x.asType(.float16).reshaped([-1]).asArray(Float16.self)
                var data = Data(capacity: a.count * 2)
                for v in a { var u = v.bitPattern; data.append(UnsafeBufferPointer(start: &u, count: 1)) }
                try? data.write(to: URL(fileURLWithPath: "/tmp/dflash_dump/\(name).bin"))
            }
            for (i, f) in feats.enumerated() { dump("feat_\(i)", f) }
            dump("block_embed", emb)
            dump("draft_hidden", h)
            dump("draft_logits", lg)
            let meta = "baseLen=\(baseLen) T=\(T) u=\(u) d=\(out)\n"
            try? meta.write(toFile: "/tmp/dflash_dump/meta.txt", atomically: true, encoding: .utf8)
            try? Data().write(to: URL(fileURLWithPath: "/tmp/dflash_dump/done"))
            FileHandle.standardError.write(Data("[dflash] dumped first cycle -> /tmp/dflash_dump/\n".utf8))
        }
        if ProcessInfo.processInfo.environment["QWISP_DFLASH_DEBUG"] == "1" {
            let mx = MLX.max(lg[0], axes: [-1])
            let mn = MLX.min(lg[0], axes: [-1])
            MLX.eval([mx, mn])
            FileHandle.standardError.write(Data(
                "[dflash] baseLen=\(baseLen) T=\(T) u=\(u) d0=\(out[0]) d1=\(out[1]) d2=\(out[2]) lgmax=\(String(format: "%.2f", mx.item(Float.self))) lgmin=\(String(format: "%.2f", mn.item(Float.self)))\n".utf8))
        }
        return out
    }
}

/// A loaded draft session: the model + its rolling context window.
public final class DFlashSession {
    public let model: DFlashDraftModel
    public let ctx = DFlashContext()
    public init(model: DFlashDraftModel) { self.model = model }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic forward check (qwisp dflash-check): run the draft forward on
// deterministic LCG inputs and print a hash, for comparison against the
// reference dflash_mlx/model.py with identical inputs. Verifies the port
// numerically without needing the target model or a GPU capture.
// ─────────────────────────────────────────────────────────────────────────────
public enum DFlashCheck {
    public struct LCG64 {
        public var s: UInt64
        public init(seed: UInt64) { s = seed }
        public mutating func next() -> Float {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let hi = Double(s >> 11) / Double(1 << 53)   // fp64 like the Python side
            return Float(hi * 2.0 - 1.0)                 // round ONCE to fp32
        }
    }

    public static func run(draftDir: String) -> String {
        guard let m = DFlashDraftModel.load(draftDir: draftDir) else { return "DFLASHCHECK FAIL: load\n" }
        let T = 8, L = DFlashDraftModel.blockSize
        var rng = LCG64(seed: 0xDEADBEEF)
        // Target features: 8 × [T, 2048]
        var feats: [MLXArray] = []
        for _ in 0 ..< 8 {
            let a = (0 ..< (T * DFlashDraftModel.H)).map { _ in rng.next() }
            feats.append(MLXArray(a, [T, DFlashDraftModel.H]))
        }
        let dctx = m.projectTargetFeatures(feats)          // [1, T, 2048]
        // Block hidden: [1, L, 2048]
        let ba = (0 ..< (L * DFlashDraftModel.H)).map { _ in rng.next() }
        let bh = MLXArray(ba, [1, L, DFlashDraftModel.H])
        let ctxPos = MLXArray(Array(0 ..< Int32(T)))
        var out = ""
        func dumpHash(_ label: String, _ x: MLXArray) {
            MLX.eval([x])
            let flat = x.reshaped([-1]).asArray(Float.self)
            var hash: UInt64 = 0xcbf29ce484222325
            for v in flat {
                hash ^= UInt64(v.bitPattern)
                hash = hash &* 0x100000001b3
            }
            out += "\(label) hash=\(String(hash, radix: 16)) head=[\(flat.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ","))]\n"
        }
        dumpHash("ctx0", feats[0])
        dumpHash("dctx", dctx)
        var h = bh
        for li in 0 ..< DFlashDraftModel.nLayers {
            let normed = MLXFast.rmsNorm(h, weight: m.layers[li].inputLN, eps: DFlashDraftModel.eps)
            if li == 0 {
                // attention internals: post-rope q / kCtx / kNoise
                let qo0 = MLX.matmul(normed, m.layers[li].qW.transposed()).reshaped([1, L, 32, 128])
                var qq = MLXFast.rmsNorm(qo0, weight: m.layers[li].qNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                qq = MLXFast.RoPE(qq, dimensions: 128, traditional: false, base: 10_000_000, scale: 1.0, offset: T)
                let kc0 = MLX.matmul(dctx, m.layers[li].kW.transposed()).reshaped([1, T, 8, 128])
                var kc = MLXFast.rmsNorm(kc0, weight: m.layers[li].kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                kc = MLXFast.RoPE(kc, dimensions: 128, traditional: false, base: 10_000_000, scale: 1.0, offset: 0)
                let kn0 = MLX.matmul(normed, m.layers[li].kW.transposed()).reshaped([1, L, 8, 128])
                var kn = MLXFast.rmsNorm(kn0, weight: m.layers[li].kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                kn = MLXFast.RoPE(kn, dimensions: 128, traditional: false, base: 10_000_000, scale: 1.0, offset: T)
                dumpHash("q0", qq[0, 0, 0 ..< 1, 0 ..< 8].reshaped([-1]))
                dumpHash("kCtx0", kc[0, 0, 0 ..< 1, 0 ..< 8].reshaped([-1]))
                dumpHash("kNoise0", kn[0, 0, 0 ..< 1, 0 ..< 8].reshaped([-1]))
                let vc0 = MLX.matmul(dctx, m.layers[li].vW.transposed()).reshaped([1, T, 8, 128]).transposed(0, 2, 1, 3)
                dumpHash("vCtx0", vc0[0, 0, 0 ..< 1, 0 ..< 8].reshaped([-1]))
                let vn0 = MLX.matmul(normed, m.layers[li].vW.transposed()).reshaped([1, L, 8, 128]).transposed(0, 2, 1, 3)
                dumpHash("vNoise0", vn0[0, 0, 0 ..< 1, 0 ..< 8].reshaped([-1]))
                let m0 = (MLXArray((0 ..< L).map { Int32(T + $0) }).reshaped([L, 1]) .>= MLXArray(Array(0 ..< Int32(T + L))).reshaped([1, T + L])).reshaped([-1])
                dumpHash("mask0", m0.asType(.float32))
            }
            let attn = m.layers[li].forwardAttention(normed, ctx: dctx, ctxPositions: ctxPos, queryOffset: T)
            if li == 0 {
                dumpHash("L0_attn", attn)
                // raw SDPA output: recompute q/k/v from the debug path and run SDPA directly
                let qo0 = MLX.matmul(normed, m.layers[li].qW.transposed()).reshaped([1, L, 32, 128])
                var qq = MLXFast.rmsNorm(qo0, weight: m.layers[li].qNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                qq = MLXFast.RoPE(qq, dimensions: 128, traditional: false, base: 10_000_000, scale: 1.0, offset: T)
                let kc0 = MLX.matmul(dctx, m.layers[li].kW.transposed()).reshaped([1, T, 8, 128])
                var kc = MLXFast.rmsNorm(kc0, weight: m.layers[li].kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                kc = MLXFast.RoPE(kc, dimensions: 128, traditional: false, base: 10_000_000, scale: 1.0, offset: 0)
                let kn0 = MLX.matmul(normed, m.layers[li].kW.transposed()).reshaped([1, L, 8, 128])
                var kn = MLXFast.rmsNorm(kn0, weight: m.layers[li].kNorm, eps: DFlashDraftModel.eps).transposed(0, 2, 1, 3)
                kn = MLXFast.RoPE(kn, dimensions: 128, traditional: false, base: 10_000_000, scale: 1.0, offset: T)
                let keysD = MLX.concatenated([kc, kn], axis: 2)
                let valsD = MLX.concatenated([MLX.matmul(dctx, m.layers[li].vW.transposed()).reshaped([1, T, 8, 128]).transposed(0, 2, 1, 3),
                                              MLX.matmul(normed, m.layers[li].vW.transposed()).reshaped([1, L, 8, 128]).transposed(0, 2, 1, 3)], axis: 2)
                let mD = (MLXArray((0 ..< L).map { Int32(T + $0) }).reshaped([L, 1]) .>= MLXArray(Array(0 ..< Int32(T + L))).reshaped([1, T + L])).reshaped([1, 1, L, T + L])
                let outD = MLXFast.scaledDotProductAttention(queries: qq, keys: keysD, values: valsD,
                    scale: DFlashDraftModel.scale, mask: .array(mD))
                dumpHash("L0_rawSDPA", outD.transposed(0, 2, 1, 3).reshaped([1, L, -1]))
                dumpHash("oW", m.layers[li].oW)
                dumpHash("L0_oproj_raw", MLX.matmul(outD.transposed(0, 2, 1, 3).reshaped([1, L, -1]), m.layers[li].oW.transposed()))}
            let h2 = h + attn
            let post = MLXFast.rmsNorm(h2, weight: m.layers[li].postLN, eps: DFlashDraftModel.eps)
            let gate = MLX.matmul(post, m.layers[li].gateW.transposed())
            let up = MLX.matmul(post, m.layers[li].upW.transposed())
            let mlp = MLX.matmul(MLX.multiply(MLX.multiply(gate, MLX.sigmoid(gate)), up), m.layers[li].downW.transposed())
            if li == 0 { dumpHash("L0_mlp", mlp) }
            h = h2 + mlp
            dumpHash("L\(li)", h)
        }
        h = MLXFast.rmsNorm(h, weight: m.normW, eps: DFlashDraftModel.eps)
        dumpHash("norm", h)
        return "DFLASHCHECK ok\n" + out
    }
}
