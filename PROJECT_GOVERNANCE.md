# ⚖️ Project Governance & Legislative Charter: Parallax

> Formalized during Workflow 1.5: Project Bootstrap.

---

## 🏛️ 1. Legislative Invariants & Non-Negotiable Rules

1. **Deterministic Bounds:** Probabilistic model calls must remain isolated from critical transactional execution paths.
2. **Strict Typing:** 100% type annotations enforced via strict type checking. Zero unvalidated dictionary payloads in core business logic.
3. **Trace-Based Observability:** Every multi-step workflow must emit a structured trace ID with latency, provider, and error spans.
4. **Eval-Driven Development (EDD):** Build against acceptance criteria defined in `docs/golden_tests/cases.yaml`.
5. **Data Privacy & Security:** Zero unhashed PII or plaintext credentials in logs or memory state.

---

## 🎛️ 2. Active Workflow Customization Matrix

| Workflow | Active Status | Execution Rigor & Scope | Trigger Cadence |
|---|---|---|---|
| **Workflow 1: Inception Board** | Completed | Full specification bundle | Pre-Flight |
| **Workflow 1.5: Project Bootstrap**| Completed | Toolchain scaffolding & AI constitution | Initial Setup |
| **Workflow 2: HIVE Implementation** | Active | Multi-agent Pods with Git worktrees | Build Time |
| **Workflow 4: PR Gatekeeper** | Active | SAST + Contract Guardian + Taint review | Pre-Merge |
| **Workflow 5: Codebase Scribe** | Active | CFG mapping + AST dead-code pruning + Vault Sync | Post-Coding |
| **Workflow 6: Continuous Evals** | Active | Golden benchmark suite & regression gate | Release / CI |
| **Workflow 7: Incident Commander**| Standby | 5-Whys RCA + Minimal Failing Reproduction | On Crash |

---

## 🛠️ 3. Verification Rigor Standard: STANDARD
- **Quality Gate:** `make gate` (Runs linter, strict types, and unit tests in <10s).
- **Pre-commit Gate:** Mandatory commit hooks enforcing clean format and secret scanning.
