# Karl — Offline LLM Lab

An offline LLM workspace for prompt experiments, session branching, local document retrieval, and inspectable generation traces.

**Status:** Clean source snapshot  
**Built with:** Python, PyQt6, llama-cpp-python, local embeddings

## Reading order

1. Read the root README for project purpose and setup navigation.
2. Follow the entry points below to find the implementation.
3. Use the file guide to locate individual modules, tests, assets, and configuration.
4. Check component licenses before redistributing dependencies or assets.

## Important paths

- [PROJECT_NOTES.md](../PROJECT_NOTES.md) — Preserved detailed setup, usage, architecture, and development notes from the earlier README.
- [app](../app) — Application modules and user interface
- [core](../core) — Shared domain logic and processing
- [requirements.txt](../requirements.txt) — Python dependency requirements and optional dependency groups; install with the project’s documented interpreter and environment.

## Structure

| Path | Role |
| --- | --- |
| [.github/](../.github/) | GitHub workflow and contributor configuration (2 tracked entries). |
| [app/](../app/) | Application modules and user interface (187 tracked entries). |
| [core/](../core/) | Shared domain logic and processing (9 tracked entries). |
| [data/](../data/) | Structured reference or sample data (48 tracked entries). |
| [docs/](../docs/) | Project documentation, decisions, and reference material (26 tracked entries). |
| [eval/](../eval/) | Project-specific eval components (9 tracked entries). |
| [helm/](../helm/) | Project-specific helm components (9 tracked entries). |
| [k8s/](../k8s/) | Project-specific k8s components (7 tracked entries). |
| [neovim/](../neovim/) | Project-specific neovim components (1 tracked entries). |
| [oss/](../oss/) | Project-specific oss components (22 tracked entries). |
| [tests/](../tests/) | Automated checks and test fixtures (86 tracked entries). |
| [tools/](../tools/) | Developer utilities and supporting automation (10 tracked entries). |
| [training/](../training/) | Project-specific training components (3 tracked entries). |

## Scope

Model weights, local documents, conversations, runtime keys, and earlier credential-bearing history are excluded from this snapshot. PyQt6 and PyMuPDF have separate GPL/AGPL or commercial licensing choices.

## Maintaining the guide

The file catalog is a source-inspection snapshot, not proof that every runtime path works. Update it when moving files or changing module responsibilities. The 2026-10-12 documentation review did not run every application, deploy services, or validate all historical performance claims.
