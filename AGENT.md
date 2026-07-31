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
