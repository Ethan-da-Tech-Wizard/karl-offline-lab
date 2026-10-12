# Karl

An offline lab for working with local language models. Karl runs a model inside your own Python process, shows you its reasoning while it generates, and keeps a full trace of every run, so you can see exactly what a prompt did.

Nothing leaves your machine. There are no cloud calls and no telemetry, and the optional editor bridge only listens on localhost.

## What it does

- **Live reasoning view.** DeepSeek-R1 style `<think>` blocks are split into their own panel, apart from the final answer.
- **Session branching.** Branch from any message in a conversation to try a different prompt without losing the original.
- **Generation traces.** Every run is logged to JSONL with the prompt, settings, reasoning and response.
- **A hackable core.** `core/interaction_loop.py` and `core/cognitive_parser.py` are plain Python files that reload on every generation. Edit one, press Generate, no restart needed.
- **Local document search.** Add PDFs, Word files, code, markdown or CSVs to the knowledge base and Karl chunks, embeds and retrieves them on your machine.
- **Extras.** A VS Code / Code OSS extension, a Neovim client, an eval suite, and LoRA training tools.

## What you need

- **Python 3.10 or newer.** The project targets 3.12.
- **A C/C++ compiler and CMake.** The `llama-cpp-python` package is compiled when you install it. On Arch: `sudo pacman -S base-devel cmake python`. On Debian or Ubuntu: `sudo apt install build-essential cmake python3-venv`.
- **A desktop session for the window.** The interface is PyQt6, which needs the usual Qt system libraries on Linux (for example `libegl1` and `libxkbcommon0` on Debian or Ubuntu). If you only want the backend, run it headless (see below).
- **Enough memory for your model.** The default 1.5B model needs about 3 GB of RAM and a download of roughly 1 GB. The 7B and 8B models want about 8 GB, the 14B about 16 GB, and the 70B about 48 GB.
- **A normal user account.** Karl refuses to start as root.
- **Internet for setup only.** You need it once to install packages, download a model, and let document search fetch its small embedding and reranking models the first time you use it. After that Karl runs offline.

Optional extras:

- An **NVIDIA GPU with CUDA drivers** for GPU inference and LoRA training.
- **Docker with the NVIDIA Container Toolkit** if you would rather run the backend in a container.
- **Node.js** if you want to build the VS Code extension.

## Getting started

~~~bash
git clone https://github.com/Ethan-da-Tech-Wizard/karl-offline-lab.git
cd karl-offline-lab
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
~~~

The install builds `llama-cpp-python` from source, so it can take a few minutes. For a build tuned to your CPU you can instead run `CMAKE_ARGS="-DGGML_NATIVE=ON" pip install llama-cpp-python --no-binary llama-cpp-python`.

Download the default model (DeepSeek-R1-Distill-Qwen-1.5B, about 1 GB) and start Karl:

~~~bash
python download_test_model.py
python main.py
~~~

The model lands in `data/models/`. You can drop other GGUF files there too; `data/model_registry.json` lists the models Karl knows about.

**Headless mode.** `python main.py --headless` starts the backend without a window, for the editor extensions.

**GPU setup.** If you have an NVIDIA card, `sudo ./setup_gpu.sh` installs CUDA-enabled PyTorch and the training tools into your virtual environment.

Docker and VS Code extension instructions are in [PROJECT_NOTES.md](PROJECT_NOTES.md).

## Running the tests

~~~bash
pip install pytest
python -m pytest
~~~

The suite does not need a model or a GPU. One timing-sensitive fuzz test can fail on a slow or busy machine.

## Learn more

- [Project guide](docs/PROJECT_GUIDE.md) for what the project is and how it fits together
- [PROJECT_NOTES.md](PROJECT_NOTES.md) for Docker, the VS Code extension and the project layout
- [docs/](docs) for the architecture, requirements and design notes
- [Every file explained](docs/FILE_GUIDE.md)

## License

Karl is released under the [AGPL-3.0](LICENSE). Two libraries it uses, PyQt6 and PyMuPDF, are GPL or AGPL (with commercial options), so follow their terms if you redistribute Karl together with them. Models are not included and each has its own license; some, such as the Llama-based ones, use the Llama community license, so check the model page before you use or share one. The VS Code extension in `oss/vss_extension/` keeps its own MIT license. More detail is in the [license review](docs/LICENSE_REVIEW.md).
