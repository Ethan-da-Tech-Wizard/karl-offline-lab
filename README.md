# Karl — Offline LLM Lab

An offline LLM workspace for prompt experiments, session branching, local document retrieval, and inspectable generation traces.

**Status:** Clean source snapshot  
**Built with:** Python, PyQt6, llama-cpp-python, local embeddings

[Project guide](docs/PROJECT_GUIDE.md) · [Every file explained](docs/FILE_GUIDE.md) · [Licenses and dependencies](docs/LICENSE_REVIEW.md)

## Start here

- [PROJECT_NOTES.md](PROJECT_NOTES.md) — Preserved detailed setup, usage, architecture, and development notes from the earlier README.
- [app](app) — Application modules and user interface
- [core](core) — Shared domain logic and processing
- [requirements.txt](requirements.txt) — Python dependency requirements and optional dependency groups; install with the project’s documented interpreter and environment.

[Detailed setup, usage, and development notes](PROJECT_NOTES.md) preserve the previous project documentation.

## Repository layout

| Path | Role |
| --- | --- |
| [.github/](.github/) | GitHub workflow and contributor configuration (2 tracked entries). |
| [app/](app/) | Application modules and user interface (187 tracked entries). |
| [core/](core/) | Shared domain logic and processing (9 tracked entries). |
| [data/](data/) | Structured reference or sample data (48 tracked entries). |
| [docs/](docs/) | Project documentation, decisions, and reference material (26 tracked entries). |
| [eval/](eval/) | Project-specific eval components (9 tracked entries). |
| [helm/](helm/) | Project-specific helm components (9 tracked entries). |
| [k8s/](k8s/) | Project-specific k8s components (7 tracked entries). |
| [neovim/](neovim/) | Project-specific neovim components (1 tracked entries). |
| [oss/](oss/) | Project-specific oss components (22 tracked entries). |
| [tests/](tests/) | Automated checks and test fixtures (86 tracked entries). |
| [tools/](tools/) | Developer utilities and supporting automation (10 tracked entries). |
| [training/](training/) | Project-specific training components (3 tracked entries). |

## Scope and current limits

Model weights, local documents, conversations, runtime keys, and earlier credential-bearing history are excluded from this snapshot. PyQt6 and PyMuPDF have separate GPL/AGPL or commercial licensing choices.

## License

Original project code uses **AGPL-3.0** terms in [LICENSE](LICENSE). Third-party libraries, media, data, models, and tools retain their own terms. See the [license review](docs/LICENSE_REVIEW.md) and [dependency inventory](docs/DEPENDENCIES.csv).
