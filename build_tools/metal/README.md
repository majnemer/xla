# Metal backend build notes

This directory holds build/toolchain notes for the XLA Metal (Apple GPU)
backend. The full implementation plan lives outside the repo; this README
documents the build-system-relevant pieces only.

## Toolchain expectations

XLA's Metal backend does **not** invoke `xcrun metal` at build time. MSL
sources (`*.metal`) are embedded as text via `cc_embed_data` and compiled
to `MTLLibrary` objects at runtime via
`[MTLDevice newLibraryWithSource:options:error:]`. Distribution therefore
does not require Xcode or Command Line Tools on user systems — only
`Metal.framework` and `MetalPerformanceShaders.framework`, both of which
ship with macOS.

The Xcode pin documented below is for **test-time MSL syntax checking**,
not for build correctness or distribution.

### Pinned Xcode

- Minimum: Xcode 15.0 (Command Line Tools 15.0+)
- Verified via `xcrun metal --version` in CI smoke tests
- Bumped deliberately when a newer Metal language feature is required

## Deployment target

- `--macos_minimum_os=12.0` (macOS 12 Monterey)
- Metal 2.4 is the floor language standard. Metal 3 fast paths are gated
  at JIT time via `MTLCompileOptions.languageVersion`, with single MSL
  sources using `#if __METAL_VERSION__ >= 30000` guards for
  capability-specific code (e.g., SIMDgroup matrix intrinsics).

## CI matrix

Self-hosted macOS runners exercise the Metal targets:

| macOS         | Hardware | Role                                         |
| ------------- | -------- | -------------------------------------------- |
| 12 Monterey   | M1       | Floor — binding for Metal 2.4 codegen path   |
| 13 Ventura    | M2       | Coverage                                     |
| 14 Sonoma     | M3       | Metal 3 + bf16 hardware coverage             |
| current       | M4       | Newest stack                                 |

Linux contributors can build and test non-Metal parts of XLA freely;
Metal runtime targets under `xla/stream_executor/metal/` are
`select()`-gated to macOS via `Metal.framework` linkage. Pure-C++ pieces
(e.g., `kMetalPlatformId`) build on any host.

## Code-signing & notarization

Not required for unsigned development builds — ad-hoc unsigned binaries
can use Metal. The PJRT plugin distribution (see plan phase P3)
additionally requires Developer ID signing and `xcrun notarytool`
notarization; release pipelines handle that step separately.
