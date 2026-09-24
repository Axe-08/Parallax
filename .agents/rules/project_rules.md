# Project Constitution: Parallax

## Non-Negotiable Engineering Invariants
- **Package Management:** Use idiomatic package manager (`uv` for Python, `cargo` for Rust, `pnpm`/`npm` for TS).
- **Strict Quality Gate:** All code must pass `make gate` before completion.
- **Observability:** Emit structured trace spans for multi-step logic.
- **Separation of Concerns:** Keep core domain contracts in `contracts/` and deterministic logic isolated from external APIs.
