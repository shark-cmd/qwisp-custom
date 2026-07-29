# Qwisp Custom Fork

A custom fork of [qwisp](https://github.com/penta2himajin/qwisp) with modifications for Apple Silicon deployment with extended context window and direct Tailscale network access.

## Features

- **80K token context window** (up from default 64K)
- **Direct network binding** (0.0.0.0) for Tailscale access without Nginx
- **NSNull fix** for Jinja template rendering with null values in tool schemas
- **Extended timeouts** for long prefill operations on large contexts

## Quick Start

```bash
# Build
cd swift && swift build -c release

# Install
cp .build/release/qwisp /opt/homebrew/opt/qwisp/libexec/qwisp

# Run
./start-qwisp.sh
```

## Endpoints

- Local: `http://127.0.0.1:9080/v1`
- Tailscale: `http://<tailscale-ip>:9080/v1`

## Documentation

See [AGENT.md](./AGENT.md) for detailed setup and configuration instructions.

## Modifications

| File | Change |
|------|--------|
| `swift/Sources/qwisp/Server.swift` | Bind to 0.0.0.0 instead of 127.0.0.1 |
| `swift/Sources/qwisp/Tools.swift` | Fix NSNull to Optional<any Sendable>.none |
| `start-qwisp.sh` | Startup script with 80K context |

## Requirements

- macOS 14+ Apple Silicon
- 70GB+ RAM for FP16 model
- Xcode Command Line Tools

## License

Same as original qwisp repository.
