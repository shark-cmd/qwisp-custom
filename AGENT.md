# Qwisp Custom Fork - AGENT.md

This fork contains custom modifications for running Qwisp on Apple Silicon with extended context window and direct Tailscale network access.

## Overview

Qwisp is a local inference server for Qwen3.6-35B-A3B models on Apple Silicon. This fork includes:

- **80K token context window** (up from default)
- **Direct network binding** (0.0.0.0 instead of 127.0.0.1)
- **NSNull fix** for Jinja template rendering with null values in tool schemas
- **Extended timeouts** for long prefill operations
- **No Nginx dependency** - direct Tailscale network access

## Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew
- Tailscale (optional, for network access)
- Swift 5.9+

## Quick Start

### 1. Install Dependencies

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Swift dependencies
brew install swift
```

### 2. Build from Source

```bash
cd swift
swift build -c release
```

### 3. Install the Binary

```bash
# Copy to Homebrew path (preserves MLX resource bundle paths)
cp .build/release/qwisp /opt/homebrew/opt/qwisp/libexec/qwisp
```

### 4. Start the Server

```bash
# Using the startup script
./start-qwisp.sh

# Or manually
QWISP_PREFIX_MAX=81920 QWISP_PORT=9080 qwisp serve
```

### 5. Verify

```bash
curl http://127.0.0.1:9080/v1/models
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWISP_PORT` | 8080 | Server port |
| `QWISP_PREFIX_MAX` | 65536 | Maximum context tokens (set to 81920 for 80K) |
| `QWISP_MODEL` | - | Path to model checkpoint |

### Model Path

Default model location:
```
/Users/apple/Documents/huggingface/models/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16
```

## Custom Modifications

### 1. Network Binding (Server.swift)

Changed from `127.0.0.1` to `0.0.0.0` for direct Tailscale access:

```swift
// Before
configuration: .init(address: .hostname("127.0.0.1", port: port))

// After
configuration: .init(address: .hostname("0.0.0.0", port: port))
```

### 2. NSNull Fix (Tools.swift)

Fixed Jinja template rendering for null values in tool schemas:

```swift
// Before
case .null: return NSNull()

// After
case .null: return Optional<any Sendable>.none as any Sendable
```

### 3. Startup Script (start-qwisp.sh)

Located at `/Users/apple/start-qwisp.sh`:
- Kills Nginx (no longer needed)
- Starts qwisp with 80K context
- Binds to port 9080
- Accessible from Tailscale network

## Network Access

### Local
```
http://127.0.0.1:9080/v1
```

### Tailscale
```
http://<tailscale-ip>:9080/v1
```

## Troubleshooting

### Server won't start

1. Check if port is in use:
   ```bash
   lsof -i :9080
   ```

2. Kill existing process:
   ```bash
   pkill -f "qwisp serve"
   ```

### Model not found

Ensure model is downloaded to the correct path:
```bash
qwisp pull Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16
```

### Build errors

Ensure Xcode Command Line Tools are installed:
```bash
xcode-select --install
```

### Memory issues

The 35B FP16 model requires ~70GB RAM. For smaller memory machines, use a quantized model.

## API Compatibility

Qwisp implements the OpenAI-compatible API:

- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/completions` - Text completions

## Source Files Modified

| File | Modification |
|------|--------------|
| `swift/Sources/qwisp/Server.swift` | Network binding to 0.0.0.0 |
| `swift/Sources/qwisp/Tools.swift` | NSNull fix for Jinja rendering |
| `start-qwisp.sh` | Startup script |

## Original Repository

Forked from: https://github.com/penta2himajin/qwisp

## License

Same as original qwisp repository.
