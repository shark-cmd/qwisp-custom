# Qwisp Custom Fork

A custom fork of [qwisp](https://github.com/penta2himajin/qwisp) — an Apple Silicon inference server for Qwen3 MoE models — with modifications for extended context, direct network access, and long-conversation stability.

**Fork repo:** https://github.com/shark-cmd/qwisp-custom  
**Upstream:** https://github.com/penta2himajin/qwisp  
**Target model:** `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`

---

## What's changed

| Modification | File(s) | Why |
|---|---|---|
| Bind to `0.0.0.0` instead of `127.0.0.1` | `Server.swift` | Direct Tailscale access without Nginx |
| Fix `NSNull` → `Optional<any Sendable>.none` | `Tools.swift` | Jinja null values in tool schemas crashed inference |
| **Sliding window context compacting** | `LLMBackend.swift`, `SeedlessFusedVerify.swift`, `SeedlessMetalForward.swift`, `TellRuntime.swift` | Keeps long conversations alive past the 80K limit |
| Startup script with all env vars | `start-qwisp.sh` | One-command launch |

---

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- ~70 GB unified memory for the FP16 35B model
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew + original `qwisp` installed via tap (for the MLX resource bundle)

---

## Build & install

```bash
# Clone this fork
git clone https://github.com/shark-cmd/qwisp-custom.git
cd qwisp-custom/swift

# Build release binary (~90s on M2 Ultra)
swift build -c release

# Overwrite the Homebrew-installed binary.
# Must target libexec/ — the binary resolves default.metallib relative to itself.
cp .build/arm64-apple-macosx/release/qwisp /opt/homebrew/opt/qwisp/libexec/qwisp
```

> **Why not replace `/opt/homebrew/bin/qwisp`?**  
> The binary loads `default.metallib` via a path relative to its own location.  
> Homebrew's `libexec/qwisp` is the real binary; `bin/qwisp` is a shim.  
> Replacing `libexec/qwisp` keeps the Metal library lookup intact.

---

## Running

### Quick start

```bash
./start-qwisp.sh
```

This starts qwisp on port `9080`, bound to `0.0.0.0`, with an 80K context window and sliding window compacting enabled (fires near the 80K limit).

### Manual launch

```bash
QWISP_PREFIX_MAX=81920 \
QWISP_SLIDING_WINDOW=75000 \
QWISP_WINDOW_HEADROOM=6000 \
QWISP_PORT=9080 \
qwisp serve
```

### Verify

```bash
curl http://127.0.0.1:9080/v1/models
```

---

## Configuration

All settings are environment variables. None are required — defaults work for the standard setup.

| Variable | Default | Description |
|---|---|---|
| `QWISP_PORT` | `8080` | Server listen port |
| `QWISP_PREFIX_MAX` | `65536` | Max KV cache tokens (arena size). Set to `81920` for 80K context. |
| `QWISP_SLIDING_WINDOW` | `0` (off) | Context compacting window in tokens. `0` = disabled. Set to e.g. `75000` to compact when context approaches 80K. |
| `QWISP_WINDOW_HEADROOM` | `4096` | Tokens to keep after compacting. After a compacting event, the retained context is `QWISP_SLIDING_WINDOW - QWISP_WINDOW_HEADROOM` tokens. |
| `QWISP_PREFIX_CACHE` | on | Prefix KV cache (cross-turn reuse). |

### Recommended production settings

```bash
QWISP_PREFIX_MAX=81920          # 80K arena
QWISP_SLIDING_WINDOW=75000      # compact when context hits ~69K
QWISP_WINDOW_HEADROOM=6000      # keep last ~69K tokens after compact
QWISP_PORT=9080
```

### Sliding window compacting explained

Without compacting, inference fails (or degrades) once the conversation exceeds `QWISP_PREFIX_MAX` tokens. With compacting enabled:

1. When `context_tokens >= QWISP_SLIDING_WINDOW - QWISP_WINDOW_HEADROOM`, compacting fires.
2. The oldest `drop_tokens = context - QWISP_SLIDING_WINDOW + QWISP_WINDOW_HEADROOM` tokens are discarded from the front of the sequence.
3. The KV cache is reset to empty.
4. The truncated sequence is re-prefilled from scratch.
5. Generation continues normally.

**Why re-prefill instead of shifting KV entries?**  
RoPE (Rotary Position Embedding) bakes absolute token positions directly into each key vector before it is written to the KV cache. Physically shifting cache bytes left doesn't change those embedded positions — the model would attend over keys with wrong positions and generate incoherent output or immediate EOS. Re-prefilling gives clean, positionally-consistent KV entries.

**Cost:** one extra prefill pass (~1–5 seconds depending on retained context length). Subsequent turns resume at normal speed.

**Server log on compacting:**
```
[qwisp] context compacting: dropping 6144 oldest tokens (context: 75264/75000)
```

---

## Network access

qwisp binds to `0.0.0.0:9080` and is accessible on:

| Interface | URL |
|---|---|
| Loopback | `http://127.0.0.1:9080/v1` |
| LAN / Tailscale | `http://<your-tailscale-ip>:9080/v1` |

Set this as the `base_url` in any OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://100.93.131.76:9080/v1", api_key="x")
```

---

## API

Implements the OpenAI-compatible HTTP API:

```
GET  /v1/models
POST /v1/chat/completions   (streaming supported)
POST /v1/completions
```

### Streaming example

```bash
curl -N http://127.0.0.1:9080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16",
    "messages": [{"role": "user", "content": "Explain TCP in one sentence."}],
    "max_tokens": 200,
    "stream": true
  }'
```

> **Note on Qwen3 thinking tokens:** This model emits `reasoning_content` chunks (thinking tokens) before the final `content` chunk. Most OpenAI clients ignore `reasoning_content` — pass a large enough `max_tokens` (500+) to let the model finish thinking before the content token appears, or use `enable_thinking: false` in `extra_body` if your client supports it.

---

## Modified source files

All changes are relative to `swift/Sources/`:

### `qwisp/Server.swift`
Bind address changed from `127.0.0.1` to `0.0.0.0`:
```swift
// before
.init(address: .hostname("127.0.0.1", port: port))
// after
.init(address: .hostname("0.0.0.0", port: port))
```

### `qwisp/Tools.swift`
Fix `NSNull` crashing Jinja null rendering in tool schemas:
```swift
// before
case .null: return NSNull()
// after
case .null: return Optional<any Sendable>.none as any Sendable
```

### `QwispCore/LLMBackend.swift`
Sliding window compacting in `generateCached()` (the live streaming code path):
- Checks `full.count >= QWISP_SLIDING_WINDOW - QWISP_WINDOW_HEADROOM` before each `runSpecLoop` call.
- On trigger: drops oldest tokens from `effectiveFull`, resets KV cache via `prefixEmptySnap`, re-prefills truncated content, invalidates prefix snapshot slots.
- Also present in `generate()` for the non-cached path (bolt / sampling).

### `QwispCore/SeedlessFusedVerify.swift`
Added `allKVCaches` accessor on `SeedlessFusedForward`:
```swift
public var allKVCaches: [KVCacheBufs] {
    layers.compactMap { $0.kvCache }
}
```

### `QwispCore/TellRuntime.swift`
Wired `backend.kvCaches = fwd.allKVCaches` in both `fusedBackendWithFwd` and `streamingBackendGlue` so `LLMBackend` can reset individual layer KV lengths directly when `prefixEmptySnap` is unavailable.

### `QwispCore/SeedlessMetalForward.swift`
Added Metal `shift_kv` kernel (MSL) and `shiftKVCache` / `getQueue` Swift API. The kernel is compiled and the pipeline is initialised alongside the existing write-KV pipeline. **Currently unused** in the compacting path (re-prefill is correct; shift was the original approach and is kept for potential future use on decode-only paths).

---

## Troubleshooting

**Port already in use:**
```bash
lsof -i :9080
pkill -f "qwisp serve"
```

**Model not found:**
```bash
qwisp pull Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16
```

**Build fails — missing Xcode tools:**
```bash
xcode-select --install
```

**Replies are empty / very short:**  
The model uses thinking tokens (`reasoning_content`). If `max_tokens` is too low, the model exhausts the budget during the thinking phase and never emits `content`. Use `max_tokens >= 500` for simple questions, `>= 2000` for complex ones.

**Compacting fires too often:**  
Raise `QWISP_SLIDING_WINDOW` or lower `QWISP_WINDOW_HEADROOM`. Each compacting event costs one extra prefill pass.

**Compacting fires but answers degrade:**  
This shouldn't happen with the re-prefill approach. If it does, check that `QWISP_PREFIX_MAX` is large enough to hold `QWISP_SLIDING_WINDOW` tokens (`QWISP_PREFIX_MAX >= QWISP_SLIDING_WINDOW`).

---

## License

Same as the original qwisp repository.
