# AGENT.md — Qwisp Custom Fork

Operational reference for the `shark-cmd/qwisp-custom` fork.  
See README.md for build instructions and configuration reference.

---

## Runtime layout

| Path | Purpose |
|---|---|
| `/opt/homebrew/opt/qwisp/libexec/qwisp` | Live binary (overwritten after each build) |
| `/opt/homebrew/opt/qwisp/libexec/default.metallib` | MLX Metal library — must stay adjacent to binary |
| `/Users/apple/qwisp-fork/` | Fork source + startup script (tracked in git) |
| `/Users/apple/qwisp-src/` | Original upstream source (reference only) |
| `/Users/apple/start-qwisp.sh` | Symlink/copy of `qwisp-fork/start-qwisp.sh` |
| `/Users/apple/qwisp_script.log` | Server stdout/stderr (nohup log) |

---

## Build → deploy cycle

```bash
cd /Users/apple/qwisp-fork/swift
swift build -c release                          # ~90s
cp .build/arm64-apple-macosx/release/qwisp \
   /opt/homebrew/opt/qwisp/libexec/qwisp        # deploy
pkill -f "qwisp serve" && sleep 2              # restart
./start-qwisp.sh                               # or: bash /Users/apple/start-qwisp.sh
```

Build output is `arm64-apple-macosx/release/qwisp` (~52 MB stripped).  
Do **not** use `swift run` — it doesn't set `RPATH` the same way and the binary won't find `default.metallib`.

---

## Endpoints

| | URL |
|---|---|
| Local | `http://127.0.0.1:9080/v1` |
| Tailscale | `http://100.93.131.76:9080/v1` |
| Health | `curl http://127.0.0.1:9080/v1/models` |

---

## Environment variables

```bash
QWISP_PREFIX_MAX=81920        # KV cache arena: 80K tokens
QWISP_SLIDING_WINDOW=75000    # compact when context hits ~69K tokens
QWISP_WINDOW_HEADROOM=6000    # keep last ~69K tokens after compact
QWISP_PORT=9080               # listen port
```

Set `QWISP_SLIDING_WINDOW=0` to disable compacting entirely (zero overhead).

---

## How sliding window compacting works

### Code path

All streaming non-sampling requests route through `generateCached()` in `LLMBackend.swift`, not `generate()`. The sliding window check runs at the TOP of `generateCached()` — BEFORE any prefill — so the retained context is prefilled exactly once.

### Logic (simplified)

```
full = all prompt tokens (system + history + current user turn)
content = full[0..<contentLen]        # cached content boundary
if full.count >= QWISP_SLIDING_WINDOW - QWISP_WINDOW_HEADROOM:
    drop = full.count - QWISP_SLIDING_WINDOW + QWISP_WINDOW_HEADROOM
    content = content[drop...]        # drop oldest tokens from the CONTENT
    reset KV cache to empty snapshot  # via prefixEmptySnap / kv.len = 0
    compacted = true                  # arena restart: restoreLen=0, slots cleared
# (only if !compacted) multi-slot / RAM / disk restore of the cached arena
# delta prefill loop prefills `content` ONCE (0..<content.count)
Tell.runSpecLoop(promptIds: compacted ? content+genSuffix : full,
                 prefillTokens: genSuffix, ...)   # genSuffix is UNCHANGED by compacting
```

**Why check before prefill (not after):** the old implementation prefilled the full prompt, then restored to empty and re-prefilled the truncated sequence — doubling prefill cost (~2x TTFT on 50K-token Hermes prompts). Checking first means the truncated content is prefilled exactly once. Measured on a 9610-token prompt with a 4096 window: 37.5s → 8.8s (4.3x). Note `genSuffix` is untouched by compacting: dropping from the front shifts contentLen by `drop`, so the suffix (which starts at contentLen) stays where it is.

### Why re-prefill (not KV shift)

RoPE encodes absolute token position into each key vector before it is written to the cache:

```
key_at_slot_i = rope(raw_key, position=i)
```

Shifting the cache bytes left by `drop` slots moves `key_at_slot_drop` to `slot_0`, but the embedded position is still `drop`, not `0`. When the model runs SDPA it sees keys with wrong positions → incoherent attention scores → garbage output or immediate EOS.

Re-prefilling writes `rope(raw_key, position=0)` into `slot_0`, which is correct.

### Metal shift_kv kernel

The `shift_kv` MSL kernel and `shiftKVCache` Swift API exist in `SeedlessMetalForward.swift` and are compiled into every build. They are **not called** from the compacting path — kept for potential future use (e.g. decode-only streaming where the model appends one token at a time and never re-prefills).

### Observed behaviour

```
[qwisp] context compacting: dropping 6358 oldest tokens (context: 35030/32768)
[qwisp] prefill 1024/2048 (50%) · 434 tok/s      # single prefill of the retained 28.6K
[qwisp] complete prompt=35030 prefill=214 tok/s gen=172 ttft=134876ms decode=27.1 tok/s (141.17s)
```

The retained context is prefilled exactly once (pre-compaction check). Coherent output verified after compacting at scale (9×8 → '72').

---

## Modifications reference

### Server.swift — `0.0.0.0` binding

```swift
// qwisp/Server.swift
configuration: .init(address: .hostname("0.0.0.0", port: port))
```

Allows Tailscale peers to reach the server directly. No Nginx required.

### Tools.swift — NSNull fix

```swift
// qwisp/Tools.swift
case .null: return Optional<any Sendable>.none as any Sendable
```

`NSNull()` survives `as? [String: Any]` casts but becomes a non-nil value in Swift's type system, breaking Jinja's `if value` checks. `Optional.none as any Sendable` correctly tunnels as genuine nil.

### SeedlessFusedVerify.swift — allKVCaches

```swift
// QwispCore/SeedlessFusedVerify.swift  (SeedlessFusedForward)
public var allKVCaches: [KVCacheBufs] {
    layers.compactMap { $0.kvCache }
}
```

Exposes all layer KV buffers so `LLMBackend` can zero their `.len` when `prefixEmptySnap` is unavailable.

### TellRuntime.swift — kvCaches wiring

```swift
// QwispCore/TellRuntime.swift  (fusedBackendWithFwd + streamingBackendGlue)
backend.kvCaches = fwd.allKVCaches
```

Passes the layer KV buffer list into `SpecBackend` so it reaches `LLMBackend.generateCached`.

### LLMBackend.swift — sliding window in generateCached

Added after `genSuffix`/`maxK` computation, before `Tell.runSpecLoop`:

```swift
let slidingWindow = Tell.envInt("QWISP_SLIDING_WINDOW", 0)
let windowHeadroom = Tell.envInt("QWISP_WINDOW_HEADROOM", 4096)
var effectiveFull = full
var effectiveGenSuffix = genSuffix
if slidingWindow > 0 && full.count >= slidingWindow - windowHeadroom {
    let dropTokens = full.count - slidingWindow + windowHeadroom
    // ... drop + reset + re-prefill ...
}
```

Also present in `generate()` (non-cached / bolt path) for completeness, but that path is not used on this machine for the streaming tier.

---

## Commit history (key changes)

| Commit | Description |
|---|---|
| initial | Server.swift 0.0.0.0 binding, Tools.swift NSNull fix |
| `77c4f3a` | Add sliding window: Metal shift_kv kernel + Swift API + LLMBackend integration |
| `aee8f17` | Fix: move sliding window check into generateCached() (actual streaming code path) |
| `1812b3d` | Fix: replace KV shift with reset+reprefill (correct RoPE position alignment) |
| `48cbc8b`…`e75ddc0` | perf: cache env vars at init (hot-path lookups eliminated) |
| `030569f` | **Fix: eliminate double prefill** — compact BEFORE prefill so retained context is prefilled once (37.5s → 8.8s @ 9610 tok). Also fixes the Swift-6 closure `self` build breakage that silently blocked all builds after the env-caching commits. |

## Upstream comparison (penta2himajin/qwisp, v0.3.11)

Checked 2026-08-01. The fork already carries all of upstream v0.3.11's `LLMBackend`
work (`isStreamingTier`, `prefixArenaMaxDefault`, `cachedGenBudget`, the 64K-cliff
BYPASS warning, client-abort check in the delta prefill) and is AHEAD on:

- **Tools.swift NSNull fix** — upstream v0.3.11 still returns `NSNull()` (crashes the
  `content` field decoding on `messages: [{"content": null}]`); the fork ships the fix.
- **Streaming robustness** — StreamDetok multibyte fix, `<tool_call>` buffering, generate
  loop error logging (upstream has only a `prompt(req)` wrapper).
- **Sliding window compacting** — upstream has no such feature.

Adopted from upstream: `Memory.clearCache()` on KV-arena GROWTH rebuild (4a90737;
measured −22.5GB peak + ~10% faster on 64GB). A/B on the 35K workload: 141.9s → 139.4s
(noise), free RAM 1.0GB → 2.9GB. Kill switch: `QWISP_CLEAR_CACHE_ON_GROWTH=0`.
NOT adopted: LaneServe token-budget scheduler / ctx-adaptive arenas (QWISP_LANES path
only, not used by the Hermes streaming tier) and the MMA-prefill kernel (WS-A NO-GO
verdict, flag-off default is byte-identical anyway).

---

## Known limitations

- **Compacting cuts conversation history.** After dropping oldest tokens, the model loses early conversation context. For typical agentic workloads the most recent context matters most, so this is acceptable.
- **Re-prefill latency.** Each compacting event re-prefills the retained context (~28.6K tokens ≈ 70–135s on a 35B model). This replaces the old double-prefill cost; the retained context is prefilled exactly once.
- **Prefix cache invalidated on compact.** All cross-turn prefix snapshots are cleared. The first turn after compacting pays a full prefill cost; subsequent turns rebuild the cache normally.
- **Thinking tokens.** Qwen3 emits `reasoning_content` before `content`. Many clients ignore `reasoning_content`. Use `max_tokens >= 500` or `enable_thinking: false` to get the final answer token.

---

# OMLX efficiency stack — notes & qwisp port plan (2026-08-02)

Source studied: `/Applications/oMLX.app/Contents/Resources/omlx/` + embedded
`dflash_mlx`, `mlx_vlm/turboquant.py`, `mlx_lm` (Python/MLX). OMLX runs the
SAME model family (Qwen3.6-35B-A3B) on this same Mac, port 7866, and is
actively used (117K-token requests). Its server holds 20.6GB RSS — qwisp
MUST coexist with it on 64GB (this is what OOM-killed qwisp on 2026-08-02;
fixed by the HostMemory guard, see commits).

## What OMLX does (settings.json / model_profiles.json / source)

| Technique | OMLX config | What it does | qwisp status |
|---|---|---|---|
| **SpecPrefill** (draft-guided sparse MoE prefill) | `specprefill_enabled=true, draft=DFlash, threshold=8192, keep_pct=0.1–0.2` | DFlash (6-layer, 737MB) prefills the prompt + 8 lookahead decodes; importance = pooled softmax(Q_lookahead·K_promptᵀ/√d) max over layers×heads; top 10–20% of conversation tokens get full target compute, the rest SKIP experts (KV holes at real RoPE positions, `_OffsetAdjustedRoPE` patches decode) | NOT IMPLEMENTED — the 5–10x prefill win (cold 52K ≈ 290s → ~60–90s; new-conversation delta prefill also slashed) |
| **DFlash block-diffusion spec decode** | `dflash_enabled` + `dflash_mlx` engine | Draft predicts 16-token blocks in one diffusion pass; target verifies; AdaptiveBlockPolicy shrinks block on low acceptance; draft KV = sink 64 + window 1024 | qwisp has SuffixSpec (N-gram, 3–5% accept — useless at 52K); DFlash replaces the draft |
| **TurboQuant KV cache** | `turboquant_kv_bits=2/4/8`, MSE codec + RHT rotation, fused quant kernel, keys=floor(bits) values=ceil(bits) | 4x smaller KV → decode at long context is KV-memory-bound; also 4x smaller disk states | NOT IMPLEMENTED (fp16 KV). Our decode window (QWISP_DECODE_WINDOW=16384, commit 7ad6ae4) already caps decode attention reads, same decode win WITHOUT lossiness; KV quant additionally shrinks PrefixPersist files 4x (1.9GB → 500MB) |
| **SSD/hot KV cache** | 120GB SSD + 10GB hot, paged block tables, LRU | Cross-process prefix reuse at scale; stats show 83M/90M tokens cached (92% hit) over 1021 requests | PrefixPersist (stable tier default-on; #89 tier `QWISP_PREFIX_PERSIST=1`; caps now 4096MB/8192MB). VERIFIED 2026-08-02: restart → same 14K request 83.1s → 7.4s (restore, prefill=4 tok/s) |
| **Memory guards** | `prefill_memory_guard=true, soft=0.85, hard=0.95, safe_zone=0.8, min_chunk=32` | Refuses/degrades when the WHOLE MACHINE (not just MLX) is OOM-tight | **ADOPTED** (commit 3c00cb1): `HostMemory.freeGB()` (host_statistics64 free+inactive), gates both persist save sites on `canAllocate(3.0)`. OOM root cause: omlx(20.6GB)+Docker(3.1GB)+qwisp(25.7GB)+1.2GB blob copy → killed |
| **GDN chunked/blocked-sequential Metal kernels** | `custom_kernels/qwen35_prefill/gdn.py` (chunked-parallel WY, blocked-seq S) | Gated DeltaNet prefill ~half the FLOPs of sequential recurrence, staged in 4K-token segments | qwisp already models GDN (persistentStateData carries GDN states); kernel-level optimization is deep Metal work, deferred |
| **qwen35 fused MoE kernels** | `qwen35_moe_weighted_sum` + NAX (M5 tensor-unit) dispatch detect | Fused expert-weighting reduction | Not ported (qwisp's own MoE path is MLX; NAX is M5-only — we're M1 Max) |
| **Chunked prefill** | `chunked_prefill=true, step=2048` | Chunked prefill with memory-aware chunking | qwisp hybrid prefill already chunks (min(1024,maxM)); OMLX's memory-guarded chunk sizing is the missing piece |
| **Burst decode** | `burst_decode_mode=balanced` | Decode-ahead scheduling | Not ported |

## Port plan (priority order, impact × feasibility)

1. **DFlash speculative DECODE** (replaces SuffixSpec 3% accept → 60%+): load the
   6-layer DFlash draft (config: block_size=16, mask_token_id=248077, target
   layer ids 1,6,11,16,22,27,32,37, 5×sliding+1×full attn, head_dim=128,
   hidden 2048/6144, 32 heads/8 KV) + block-diffusion denoising loop in
   Swift/MLX; feed drafted blocks into the existing SeedlessFusedVerify path.
   Decode floor is ~80 tok/s fixed; at 52K ctx (21→39 tok/s windowed) spec
   decode could reach 60–100+ tok/s. ALSO enables SpecPrefill scoring.
   **PORTED + VERIFIED (2026-08)**: pure-MLX draft forward (fc→hidden→6×
   layers→norm→lm_head) matches dflash_mlx bit-for-bit on synthetic LCG
   inputs and reproduces the reference argmax on qwisp's real captured
   features. Capture: 0-based layers [1,6,11,16,22,27,32,37] (OMLX adds +1
   to the 1-based target_layer_ids). MEASURED on the INT4 MTPLX target:
   acceptance 20–38% (15-token prompt 33–38%; 5.8K ctx 21%; 12.5K 20% —
   plateaus, the 4-bit target's hiddens diverge from the draft's bf16
   teacher). tok/step 4.0–6.7, verify amortizes ~2x at M=17 (157ms vs 378ms
   sequential at 12.5K) but draft KV rebuild is 123–146ms/step at T=1024
   (fc over the whole window). NET: 13–24 tok/s vs 40–72 tok/s fused baseline
   → net-negative on this model; matches OMLX's own dflash_enabled:false.
   The draft KV cache (incremental dctx/k/v projection) would cut the draft
   to ~10ms but acceptance must reach ~50% for parity — INT4 caps it below.
   Keep for bf16 targets / SpecPrefill scoring; OFF by default here.
2. **SpecPrefill sparse prefill**: draft scores importance → target prefills
   only top 10–20% of NEW-conversation tokens (system prefix stays dense).
   The 52K cold-start killer (290s → ~60–90s). Needs: draft prefill fwd,
   Q·K importance kernel, sparse attention mask + per-token MoE routing mask,
   RoPE position-map + offset patch, correctness gate (PREFIXE2E-style).
3. **TurboQuant KV**: quantize KV on write (per-head-vector norms + packed
   midpoints, MSE codec), dequant inline in the sdpa kernel read path;
   shrinks disk states 4x (faster restores, fits default caps) and arena
   footprint; could relax/disable the decode window (lossless attention).
4. **OMLX-scale SSD cache**: PrefixPersist already covers this; extend with
   block-granular spans (save only changed suffix) to cut the 1.9GB full
   rewrite per Hermes turn.
5. **Memory-guarded prefill chunking** (OMLX min_chunk/safe-zone): scale the
   hybrid prefill chunk by free RAM when the machine is tight.

## Key numbers from OMLX stats (this Mac, 64GB, M1 Max)

- mlx-community Qwen3.6-35B-A3B-4bit: 51.1M prompt tokens / 24.2K s ≈ 2114 tok/s
  avg (includes SSD-cache restores + specprefill sparsity; dense ceiling for
  3B-active ≈ 1.6K tok/s → OMLX EXCEEDS it via sparse prefill)
- Ornith-1.0-35B-6bit: 2372 tok/s avg; 92% cache-hit rate
- MTPLX model (ours): tried once, 346 tok/s (no cache, no specprefill)
- Decode: 117K-token prompt → 0.8 tok/s (OMLX, no KV quant at that load);
  ours at 52K with decode window: 39 tok/s — already ahead at long context

## What OMLX does NOT do for us

- The 20.6GB resident server competes with qwisp for RAM (hence the memory
  guard). Its DFlash/specprefill machinery is Python+MLX; porting the
  ALGORITHMS (not kernels) to Swift/MLX is the strategy. mlx-swift exposes
  the same ops (mlx.fast.metal_kernel, etc.) so GDN/kernels are portable.
