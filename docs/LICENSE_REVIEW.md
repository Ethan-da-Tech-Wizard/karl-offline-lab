# License review — Karl — Offline LLM Lab

Reviewed 2026-10-12. Original project code retains **MIT** terms in [LICENSE](../LICENSE).

The owner selected MIT for previously unlicensed original code. Existing licenses were preserved. This choice does not relicense copied code, dependencies, datasets, model weights, fonts, logos, or game assets.

## Preserved notices

- [LICENSE](../LICENSE)
- [oss/vss_extension/LICENSE](../oss/vss_extension/LICENSE)

## Dependency evidence

[DEPENDENCIES.csv](DEPENDENCIES.csv) lists 121 manifest and lockfile records. Repeated packages can appear under different manifests, versions, or development scopes. Registry metadata is evidence, not a replacement for the actual license text. For unpinned ranges, `metadata_version` is the version inspected and is not a claim about every install. Go and system/build tools can require notices beyond package-manager metadata.

| Component | Declared version | License evidence |
| --- | --- | --- |
| npm: ws | ^8.21.0 | [MIT](https://registry.npmjs.org/ws/latest) |
| npm: eslint | ^8.57.0 | [MIT](https://registry.npmjs.org/eslint/latest) |
| PyPI: llama-cpp-python | unpinned | [MIT](https://pypi.org/pypi/llama-cpp-python/json) |
| PyPI: PyQt6 | unpinned | [GPL-3.0-only](https://pypi.org/pypi/PyQt6/json) |
| PyPI: sentence-transformers | unpinned | [Apache-2.0](https://pypi.org/pypi/sentence-transformers/json) |
| PyPI: faiss-cpu | unpinned | [MIT](https://pypi.org/pypi/faiss-cpu/json) |
| PyPI: numpy | unpinned | [BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0](https://pypi.org/pypi/numpy/json) |
| PyPI: PyMuPDF | unpinned | [Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License](https://pypi.org/pypi/PyMuPDF/json) |
| PyPI: python-docx | unpinned | [MIT](https://pypi.org/pypi/python-docx/json) |
| PyPI: psutil | unpinned | [BSD-3-Clause](https://pypi.org/pypi/psutil/json) |
| PyPI: GPUtil | unpinned | [MIT](https://pypi.org/pypi/GPUtil/json) |
| PyPI: tqdm | unpinned | [MPL-2.0 AND MIT](https://pypi.org/pypi/tqdm/json) |
| PyPI: requests | unpinned | [Apache-2.0](https://pypi.org/pypi/requests/json) |
| PyPI: keyring | unpinned | [MIT](https://pypi.org/pypi/keyring/json) |
| PyPI: cryptography | unpinned | [Apache-2.0 OR BSD-3-Clause](https://pypi.org/pypi/cryptography/json) |
| PyPI: websockets | unpinned | [BSD-3-Clause](https://pypi.org/pypi/websockets/json) |
| PyPI: mcp | unpinned | [MIT](https://pypi.org/pypi/mcp/json) |
| PyPI: peft | unpinned | [Apache](https://pypi.org/pypi/peft/json) |
| PyPI: trl | unpinned | [Apache-2.0](https://pypi.org/pypi/trl/json) |
| PyPI: transformers | unpinned | [Apache 2.0 License](https://pypi.org/pypi/transformers/json) |
| PyPI: datasets | unpinned | [Apache 2.0](https://pypi.org/pypi/datasets/json) |

## Project-specific considerations

PyQt6 is GPL-3.0-only or commercially licensed, and PyMuPDF is AGPL/commercial ([PyQt terms](https://www.riverbankcomputing.com/commercial/license-faq), [PyMuPDF terms](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)). An MIT source license does not erase those obligations. Choose and satisfy a compatible distribution route before shipping a combined installer or hosted service. Model licenses also apply separately.

## Remaining work before binary distribution

No unresolved registry-license lookup remains in the listed package records. System packages, downloaded tools/models, optional plugins, release binaries, and externally obtained assets need a build-specific inventory. This source review is not a blanket legal certification or a claim that every release artifact has been audited.
