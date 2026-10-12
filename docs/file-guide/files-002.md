# Files 2 of 3 — Karl — Offline LLM Lab

[All pages](../FILE_GUIDE.md) · [Project guide](../PROJECT_GUIDE.md)

| File | Role and useful details |
| --- | --- |
| [app/utils/keychain_manager.py](../../app/utils/keychain_manager.py) | Python implementation of keychain manager. Defines `_os_keychain_locked`, `save_cached_token`, `load_cached_token`, `_call_keyring_with_timeout`, `_get_password_with_timeout`, `_set_password_with_timeout` and 12 more top-level symbols. |
| [app/utils/memory_manager.py](../../app/utils/memory_manager.py) | Python implementation of memory manager. Defines `MemoryManager`. |
| [app/utils/rag_pipeline.py](../../app/utils/rag_pipeline.py) | RAG Pipeline — Karl Workbench (Hardened) ========================================= Changes from original: - Persistent FAISS index: saved/loaded from data/vector_db/ - SQLite metad… Defines `_ParsedFile`, `RAGPipeline`. |
| [app/utils/session_tree.py](../../app/utils/session_tree.py) | Python implementation of session tree. Defines `_strip_thoughts`, `SessionTreeStats`, `SessionNode`, `SessionTree`. |
| [app/utils/swarm_agent_profiles.py](../../app/utils/swarm_agent_profiles.py) | Persistent specialist profile registry for Karl's swarm agents. Defines `_default_profiles`, `_normalize_profile`, `load_agent_profiles`, `save_agent_profiles`, `save_agent_profile`, `active_profile_map`. |
| [app/utils/swarm_replay.py](../../app/utils/swarm_replay.py) | Swarm Replay Manager — Karl Workbench ====================================== Reconstructs historical swarm runs for the interactive debugger/replay UI. Defines `SwarmReplayManager`. |
| [app/utils/topic_graph.py](../../app/utils/topic_graph.py) | Topic-Aware Sampling Graph for the Karl self-improvement flywheel. Defines `TopicNode`, `DynamicTopicGraph`. |
| [app/utils/trace_logger.py](../../app/utils/trace_logger.py) | Python implementation of trace logger. Defines `TraceLogger`. |
| [app/utils/tracing.py](../../app/utils/tracing.py) | Local span tracing for Karl. Defines `Tracer`, `Span`. |
| [app/utils/training_curator.py](../../app/utils/training_curator.py) | Training Data Curator — M11 Defines `_ensure_dir`, `save_example`, `get_all_examples`, `get_stats`, `_is_degraded`, `_classify_example` and 6 more top-level symbols. |
| [app/vision/__init__.py](../../app/vision/__init__.py) | Offline vision helpers for Karl. |
| [app/vision/image_preprocess.py](../../app/vision/image_preprocess.py) | Python implementation of image preprocess. Defines `image_info`, `make_thumbnail`, `prepare_for_ocr`. |
| [app/vision/image_store.py](../../app/vision/image_store.py) | Python implementation of image store. Defines `ImageStore`. |
| [app/vision/ocr_engine.py](../../app/vision/ocr_engine.py) | Python implementation of ocr engine. Defines `TesseractOcrEngine`. |
| [app/vision/schemas.py](../../app/vision/schemas.py) | Python implementation of schemas. Defines `OcrResult`, `VisionResult`, `ImageRecord`. |
| [app/vision/vision_analyzer.py](../../app/vision/vision_analyzer.py) | Python implementation of vision analyzer. Defines `AnalysisSuggestion`, `VisionAnalyzer`, `classify_image`, `_layout_hint`, `_dedupe`. |
| [app/vision/vision_model_loader.py](../../app/vision/vision_model_loader.py) | Python implementation of vision model loader. Defines `VisionModelEntry`, `read_vision_registry`, `active_vision_model_id`, `set_active_vision_model`, `installed_vision_models`, `VisionModelLoader` and 2 more top-level symbols. |
| [auto_train.py](../../auto_train.py) | Karl One-Click Auto-Train Pipeline ================================== A unified, easy-to-understand end-to-end pipeline that: 1. Generates synthetic coding/math/reasoning tasks for… Defines `generate_tasks_for_topic`, `generate_fallback_tasks`, `solve_task`, `verify_solution`, `reflect_and_correct`, `train_adapter` and 2 more top-level symbols. |
| [core/agentic_loop.py](../../core/agentic_loop.py) | Python implementation of agentic loop. Defines `should_continue`, `build_next_prompt`. |
| [core/cognitive_parser.py](../../core/cognitive_parser.py) | Python implementation of cognitive parser. Defines `parse_thought_stream`. |
| [core/default_prompts.py](../../core/default_prompts.py) | Karl's default system prompt and prompt preset registry — single source of truth. Defines `load_prompt_presets`, `save_user_preset`, `delete_user_preset`. |
| [core/hardware_scout.py](../../core/hardware_scout.py) | Python implementation of hardware scout. Defines `get_cpu_flags`, `get_hardware_uuid`, `get_hardware_profile`. |
| [core/interaction_loop.py](../../core/interaction_loop.py) | Python implementation of interaction loop. Defines `strip_html_tags`, `matches_keyword`, `_get_codex_context`, `_apply_vocab_leak_bypass`, `_truthy`, `_machine_speak_enabled` and 2 more top-level symbols. |
| [core/prompt_optimizer.py](../../core/prompt_optimizer.py) | Prompt Optimizer Engine — Karl Workbench ======================================== Handles automated prompt mutations and evolution loops using the local LLM. Defines `PromptOptimizer`. |
| [core/prompt_templates.py](../../core/prompt_templates.py) | Python implementation of prompt templates. Defines `get_template`, `list_templates`. |
| [core/security.py](../../core/security.py) | Small, dependency-light security helpers shared across every independently launched Karl entrypoint (the PyQt app in main.py, and standalone scripts like auto_train.py that are spa… Defines `assert_not_privileged`. |
| [core/workflows.py](../../core/workflows.py) | Python implementation of workflows. Defines `get_workflow`, `list_workflows`. |
| [data/agent_profiles.json](../../data/agent_profiles.json) | Structured agent profiles data; top-level fields: `architect`, `coder`, `tester`. |
| [data/codex_library/.version](../../data/codex_library/.version) | Tracked .version resource in data/codex_library; 3 bytes. |
| [data/codex_library/AI Steering.html](../../data/codex_library/AI%20Steering.html) | Browser document for AI Steering; provides page structure and may include inline styles or application logic. |
| [data/codex_library/APIs.html](../../data/codex_library/APIs.html) | Browser document for APIs; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Agile.html](../../data/codex_library/Agile.html) | Browser document for Agile; provides page structure and may include inline styles or application logic. |
| [data/codex_library/C.html](../../data/codex_library/C.html) | Browser document for C; provides page structure and may include inline styles or application logic. |
| [data/codex_library/CLI  Testing Tools.html](../../data/codex_library/CLI%20%20Testing%20Tools.html) | Browser document for CLI  Testing Tools; provides page structure and may include inline styles or application logic. |
| [data/codex_library/CSS.html](../../data/codex_library/CSS.html) | Browser document for CSS; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Codex Search.html](../../data/codex_library/Codex%20Search.html) | Browser document for Codex Search; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Docker.html](../../data/codex_library/Docker.html) | Browser document for Docker; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Eval Suite.html](../../data/codex_library/Eval%20Suite.html) | Browser document for Eval Suite; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Extension Scripts.html](../../data/codex_library/Extension%20Scripts.html) | Browser document for Extension Scripts; provides page structure and may include inline styles or application logic. |
| [data/codex_library/FastAPI.html](../../data/codex_library/FastAPI.html) | Browser document for FastAPI; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Fortran.html](../../data/codex_library/Fortran.html) | Browser document for Fortran; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Go.html](../../data/codex_library/Go.html) | Browser document for Go; provides page structure and may include inline styles or application logic. |
| [data/codex_library/HTML.html](../../data/codex_library/HTML.html) | Browser document for HTML; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Java.html](../../data/codex_library/Java.html) | Browser document for Java; provides page structure and may include inline styles or application logic. |
| [data/codex_library/JavaScript.html](../../data/codex_library/JavaScript.html) | Browser document for JavaScript; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Karl Architecture.html](../../data/codex_library/Karl%20Architecture.html) | Browser document for Karl Architecture; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Knowledge Base.html](../../data/codex_library/Knowledge%20Base.html) | Browser document for Knowledge Base; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Kubernetes.html](../../data/codex_library/Kubernetes.html) | Browser document for Kubernetes; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Mathematical Introspection.html](../../data/codex_library/Mathematical%20Introspection.html) | Browser document for Mathematical Introspection; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Nodejs.html](../../data/codex_library/Nodejs.html) | Browser document for Nodejs; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Prompt Lab.html](../../data/codex_library/Prompt%20Lab.html) | Browser document for Prompt Lab; provides page structure and may include inline styles or application logic. |
| [data/codex_library/PySide6.html](../../data/codex_library/PySide6.html) | Browser document for PySide6; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Python.html](../../data/codex_library/Python.html) | Browser document for Python; provides page structure and may include inline styles or application logic. |
| [data/codex_library/RAG Customization.html](../../data/codex_library/RAG%20Customization.html) | Browser document for RAG Customization; provides page structure and may include inline styles or application logic. |
| [data/codex_library/React.html](../../data/codex_library/React.html) | Browser document for React; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Rust.html](../../data/codex_library/Rust.html) | Browser document for Rust; provides page structure and may include inline styles or application logic. |
| [data/codex_library/SQL.html](../../data/codex_library/SQL.html) | Browser document for SQL; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Steering Tactics.html](../../data/codex_library/Steering%20Tactics.html) | Browser document for Steering Tactics; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Swarm Studio.html](../../data/codex_library/Swarm%20Studio.html) | Browser document for Swarm Studio; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Swift.html](../../data/codex_library/Swift.html) | Browser document for Swift; provides page structure and may include inline styles or application logic. |
| [data/codex_library/System.html](../../data/codex_library/System.html) | Browser document for System; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Training Studio.html](../../data/codex_library/Training%20Studio.html) | Browser document for Training Studio; provides page structure and may include inline styles or application logic. |
| [data/codex_library/TypeScript.html](../../data/codex_library/TypeScript.html) | Browser document for TypeScript; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Uvicorn.html](../../data/codex_library/Uvicorn.html) | Browser document for Uvicorn; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Workbench.html](../../data/codex_library/Workbench.html) | Browser document for Workbench; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Workflows  Modes.html](../../data/codex_library/Workflows%20%20Modes.html) | Browser document for Workflows  Modes; provides page structure and may include inline styles or application logic. |
| [data/codex_library/Xcode.html](../../data/codex_library/Xcode.html) | Browser document for Xcode; provides page structure and may include inline styles or application logic. |
| [data/feature_flags.json](../../data/feature_flags.json) | Structured feature flags data; top-level fields: `experimental_speculative_decoding`, `multimodal_vision_ocr`, `agentic_swarming_loops`. |
| [data/flywheel/active_learner.py](../../data/flywheel/active_learner.py) | Python implementation of active learner. Defines `ActiveLearner`. |
| [data/flywheel/agent1_generator.py](../../data/flywheel/agent1_generator.py) | Python implementation of agent1 generator. Defines `_ngrams`, `_jaccard_ngram`, `_is_duplicate`, `generate_math_problem_3var`, `generate_coding_problem_extended`, `generate_symbolic_problem_extended` and 2 more top-level symbols. |
| [data/flywheel/agent3_curator.py](../../data/flywheel/agent3_curator.py) | Agent 3: Automated Curator & Training Grader Part of the Karl self-improvement flywheel. Defines `_load_stats`, `_save_stats`, `run_verification`, `_verify_subprocess`, `_verify_sympy`, `generate_correction` and 4 more top-level symbols. |
| [data/flywheel/executor_sandbox.py](../../data/flywheel/executor_sandbox.py) | Python implementation of executor sandbox. Defines `SafePythonSandbox`. |
| [data/flywheel/multilang_sandbox.py](../../data/flywheel/multilang_sandbox.py) | Multi-Language Sandboxed Execution Runtime =========================================== Sets up temporary, resource-bounded compilation environments to run and verify TypeScript (vi… Defines `SafeMultiLangSandbox`. |
| [data/model_registry.json](../../data/model_registry.json) | Structured model registry list with 6 records. |
| [data/vision_model_registry.json](../../data/vision_model_registry.json) | Structured vision model registry list with 5 records. |
| [docker-compose.yml](../../docker-compose.yml) | Configuration or build metadata for docker compose; read alongside the application or tool that loads it. |
| [docker-entrypoint.sh](../../docker-entrypoint.sh) | Command-line automation for docker entrypoint; review prerequisites and side effects before executing. |
| [docs/01_problem_statement.md](../../docs/01_problem_statement.md) | Documentation: Problem Statement: The Prompt Engineering Bottleneck. Covers 1. Executive Summary; 2. Core Deficiencies in Existing Solutions; 2.1 Security and Privacy Compromises. |
| [docs/02_prd.md](../../docs/02_prd.md) | Documentation: Product Requirements Document (PRD) — Karl v2. Covers 1. Product Vision & Philosophy; 2. Target Persona; 3. Strict Constraints & Acceptance Criteria. |
| [docs/02_rag_pipeline.md](../../docs/02_rag_pipeline.md) | Documentation: RAG Pipeline — Technical Reference. Covers Storage Architecture; Schema; PRAGMA journal_mode = WAL. |
| [docs/03_frd.md](../../docs/03_frd.md) | Documentation: Functional Requirements Document (FRD) — Karl v2. Covers 1. User Interface Architecture (PyQt6); 1.1 Three-Column Layout; 1.2 Panel Responsibilities. |
| [docs/03_training_curator.md](../../docs/03_training_curator.md) | Documentation: Training Curator — Technical Reference. Covers Storage Paths; Curated Example Format; SFT Export (`export_unsloth`). |
| [docs/04_architecture.md](../../docs/04_architecture.md) | Documentation: Karl — System Architecture. Covers 1. High-Level Component Diagram; 2. Workspace Architecture; 3. The Hackable Core. |
| [docs/05_multi_agent_swarm.md](../../docs/05_multi_agent_swarm.md) | Documentation: Karl — Multi-Agent Swarm. Covers Architecture; Shared Blackboard — SwarmSessionState; Execution Pipeline. |
| [docs/05_scope_and_milestones.md](../../docs/05_scope_and_milestones.md) | Documentation: Karl — Scope Lock & Milestones. Covers Scope Statement; Completed Milestones; ✅ M1 — Headless Introspection Engine. |
| [docs/06_repo_structure.md](../../docs/06_repo_structure.md) | Documentation: Repository Structure & File Specifications — Karl v2. Covers Full Tree; Architectural Principles; 1. The Hackable Core. |
| [docs/06_ui_architecture.md](../../docs/06_ui_architecture.md) | Documentation: Karl UI Architecture. Covers Shell Coordinates; Active Workspaces; Theme System. |
| [docs/07_evaluation_suite.md](../../docs/07_evaluation_suite.md) | Documentation: Karl — Evaluation Suite. Covers Architecture; Dataset JSONL Format; EvalHarness. |
| [docs/07_risk_register.md](../../docs/07_risk_register.md) | Documentation: Risk Register — Karl. Covers Active Risk Table; Resolved Risk Detail; Open Risk Summary. |
| [docs/08_vscode_extension.md](../../docs/08_vscode_extension.md) | Documentation: Karl VS Code Extension. Covers Purpose; Current Extension Surface; How To Use Locally. |
| [docs/09_vision_implementation_plan.md](../../docs/09_vision_implementation_plan.md) | Documentation: Karl Vision System Implementation Plan. Covers 1. Goal; 2. Product Requirements; Required. |
| [docs/10_security_and_safety.md](../../docs/10_security_and_safety.md) | Documentation: Security and Safety Guide. Covers 1. The Security Model; Zero-Remote Invariants; The Bridge (WebSocket). |
| [docs/DEPENDENCIES.csv](../../docs/DEPENDENCIES.csv) | Machine-readable inventory of declared and locked dependencies, versions, reported licenses, and evidence URLs. |
| [docs/FILE_GUIDE.md](../../docs/FILE_GUIDE.md) | Index of the complete tracked-file guide, with one entry for every file and links to the paginated descriptions. |
| [docs/LICENSE_REVIEW.md](../../docs/LICENSE_REVIEW.md) | Original-code license, preserved component notices, dependency-license evidence, and unresolved distribution questions. |
| [docs/PROJECT_GUIDE.md](../../docs/PROJECT_GUIDE.md) | Maintainer orientation: purpose, important entry points, source layout, commands, and project-specific boundaries. |
| [docs/audits/architecture_walkthrough_2026-07.md](../../docs/audits/architecture_walkthrough_2026-07.md) | Documentation: Karl Architecture Audit Walkthrough. Covers Scope; Technical Debt Removed; Model teardown race. |
| [docs/audits/repo_audit_findings_2026-07.md](../../docs/audits/repo_audit_findings_2026-07.md) | Documentation: NPUWU — Karl Audit Findings & Fix Tracker. Covers Critical; High; Medium. |
| [docs/audits/security_audit_2026-07.md](../../docs/audits/security_audit_2026-07.md) | Documentation: Security & Data Integrity Audit Report — Agent 3. Covers Data Pipelines, Security, and Verification (RAG, Cryptography, & Sandbox); 1. Data Flow Map; 2. Findings. |
| [docs/audits/swarm_phase2_task_specs_2026-07.md](../../docs/audits/swarm_phase2_task_specs_2026-07.md) | Documentation: Swarm Perfecting Tasks: Phase 2 Prompts (Detailed Specifications). Covers Agent 1: Machine-Speak Prompt Integration & Speculative Decoding (Prompt 3); Objective; Task 1: Native Machine-Speak Prompting. |
| [docs/file-guide/files-001.md](../../docs/file-guide/files-001.md) | One page of the tracked-file catalog; links each listed path to its role and source-derived details. |
| [docs/file-guide/files-002.md](../../docs/file-guide/files-002.md) | One page of the tracked-file catalog; links each listed path to its role and source-derived details. |
| [docs/file-guide/files-003.md](../../docs/file-guide/files-003.md) | One page of the tracked-file catalog; links each listed path to its role and source-derived details. |
| [download_all_models.py](../../download_all_models.py) | Python implementation of download all models. Defines `download_file`, `main`. |
| [download_test_model.py](../../download_test_model.py) | Python implementation of download test model. Defines `download_file`. |
| [engine_test.py](../../engine_test.py) | engine_test.py — Karl headless introspection engine smoke test. Defines `test_introspection_engine`. |
| [eval/__init__.py](../../eval/__init__.py) | Initializes the eval Python package and its public imports. |
| [eval/benchmark_rag.py](../../eval/benchmark_rag.py) | RAG Retrieval Benchmark — Karl Workbench ========================================== Compares retrieval quality metrics on a fixed query set. Now includes generation-level BLEU and… Defines `_tokenize`, `_get_ngrams`, `compute_bleu`, `compute_rouge_1`, `compute_rouge_l`, `QueryResult` and 5 more top-level symbols. |
| [eval/datasets/code_review.jsonl](../../eval/datasets/code_review.jsonl) | Line-delimited code review records, with one event or object per line. |
| [eval/datasets/document_extractor.jsonl](../../eval/datasets/document_extractor.jsonl) | Line-delimited document extractor records, with one event or object per line. |
| [eval/datasets/grounded_answer.jsonl](../../eval/datasets/grounded_answer.jsonl) | Line-delimited grounded answer records, with one event or object per line. |
| [eval/graders.py](../../eval/graders.py) | Eval Graders — Karl Workbench ============================== Collection of pure functions that score a model output against an expected answer. All graders return a dict: Defines `exact_match`, `json_valid`, `keyword_hit`, `groundedness`, `not_in_context`, `run_grader`. |
| [eval/harness.py](../../eval/harness.py) | Eval Harness — Karl Workbench ============================== Runs a JSONL dataset through the local LLM using a specified workflow and prompt template, then grades each output and… Defines `_run_model_worker`, `CaseResult`, `EvalReport`, `EvalHarness`. |
| [eval/perplexity_bench.py](../../eval/perplexity_bench.py) | Perplexity & Throughput Benchmarking Engine — Karl Workbench ============================================================= Evaluates GGUF quantization variants across three axes: Defines `_query_free_vram_mb`, `_vram_used_since`, `_softmax`, `compute_perplexity`, `measure_throughput`, `_release_model` and 5 more top-level symbols. |
| [eval/run_eval.py](../../eval/run_eval.py) | Karl Eval CLI — run_eval.py ============================ Command-line entry point for the eval harness. Defines `parse_args`, `dry_run_mode`, `main`. |
| [flywheel_runner.py](../../flywheel_runner.py) | Python implementation of flywheel runner. Defines `write_json_atomic`, `process_task`, `main`. |
| [helm/karl/Chart.yaml](../../helm/karl/Chart.yaml) | Configuration or build metadata for Chart; read alongside the application or tool that loads it. |
| [helm/karl/templates/certificate.yaml](../../helm/karl/templates/certificate.yaml) | Configuration or build metadata for certificate; read alongside the application or tool that loads it. |
| [helm/karl/templates/configmap.yaml](../../helm/karl/templates/configmap.yaml) | Configuration or build metadata for configmap; read alongside the application or tool that loads it. |
| [helm/karl/templates/deployment.yaml](../../helm/karl/templates/deployment.yaml) | Configuration or build metadata for deployment; read alongside the application or tool that loads it. |
| [helm/karl/templates/ingress.yaml](../../helm/karl/templates/ingress.yaml) | Configuration or build metadata for ingress; read alongside the application or tool that loads it. |
| [helm/karl/templates/pvc.yaml](../../helm/karl/templates/pvc.yaml) | Configuration or build metadata for pvc; read alongside the application or tool that loads it. |
| [helm/karl/templates/secrets.yaml](../../helm/karl/templates/secrets.yaml) | Configuration or build metadata for secrets; read alongside the application or tool that loads it. |
| [helm/karl/templates/service.yaml](../../helm/karl/templates/service.yaml) | Configuration or build metadata for service; read alongside the application or tool that loads it. |
| [helm/karl/values.yaml](../../helm/karl/values.yaml) | Configuration or build metadata for values; read alongside the application or tool that loads it. |
| [k8s/certificate.yaml](../../k8s/certificate.yaml) | Configuration or build metadata for certificate; read alongside the application or tool that loads it. |
| [k8s/configmap.yaml](../../k8s/configmap.yaml) | Configuration or build metadata for configmap; read alongside the application or tool that loads it. |
| [k8s/deployment.yaml](../../k8s/deployment.yaml) | Configuration or build metadata for deployment; read alongside the application or tool that loads it. |
| [k8s/ingress.yaml](../../k8s/ingress.yaml) | Configuration or build metadata for ingress; read alongside the application or tool that loads it. |
| [k8s/pvc.yaml](../../k8s/pvc.yaml) | Configuration or build metadata for pvc; read alongside the application or tool that loads it. |
| [k8s/secret.yaml](../../k8s/secret.yaml) | Configuration or build metadata for secret; read alongside the application or tool that loads it. |
| [k8s/service.yaml](../../k8s/service.yaml) | Configuration or build metadata for service; read alongside the application or tool that loads it. |
| [karl.sh](../../karl.sh) | Command-line automation for karl; review prerequisites and side effects before executing. |
| [main.py](../../main.py) | Karl -- entry point. All library noise is suppressed here, before any third-party import runs. Defines `_NullFD`, `_run_headless`, `main`. |
| [neovim/karl.lua](../../neovim/karl.lua) | Tracked karl resource in neovim; 15,063 bytes. |
| [oss/vss_extension/.eslintrc.json](../../oss/vss_extension/.eslintrc.json) | Structured .eslintrc data; top-level fields: `env`, `parserOptions`, `rules`, `overrides`, `globals`. |
| [oss/vss_extension/.vscodeignore](../../oss/vss_extension/.vscodeignore) | Tracked .vscodeignore resource in oss/vss_extension; 91 bytes. |
| [oss/vss_extension/BRIDGE.md](../../oss/vss_extension/BRIDGE.md) | Documentation: Karl WebSocket Bridge Protocol Specification. Covers 1. Request/Response Methods (Client to Server); `get_runtime_status`; `list_models`. |
| [oss/vss_extension/LICENSE](../../oss/vss_extension/LICENSE) | Preserved license, copyright, or attribution terms; read the full text before reusing the covered component. |
| [oss/vss_extension/README.md](../../oss/vss_extension/README.md) | Documentation: Karl VS/OSS Code Extension. Covers Main Surfaces (11-Tab Webview); Look / Aesthetics System; Safety Model. |
| [oss/vss_extension/extension.js](../../oss/vss_extension/extension.js) | Source module for extension; defines `readServiceDiscovery`, `activate`, `_getSelectionOrFile`, `_sendToKarlWebview`, `MiniGptInlineCompletionProvider`, `deactivate`, `register` and 2 more symbols. |
| [oss/vss_extension/jsconfig.json](../../oss/vss_extension/jsconfig.json) | Structured jsconfig data; top-level fields: `compilerOptions`, `exclude`. |
| [oss/vss_extension/media/icon.png](../../oss/vss_extension/media/icon.png) | Image or interface asset for icon. Source/asset rights remain separate from software-license terms. |
| [oss/vss_extension/media/icon.svg](../../oss/vss_extension/media/icon.svg) | Image or interface asset for icon. Source/asset rights remain separate from software-license terms. |
| [oss/vss_extension/media/karl.css](../../oss/vss_extension/media/karl.css) | Presentation rules for karl; controls layout, typography, color, states, and responsive behavior. |
| [oss/vss_extension/media/karl.js](../../oss/vss_extension/media/karl.js) | Source module for karl; defines `bindEvents`, `startWorkflow`, `askWorkspace`, `connectAndRun`, `runSwarm`, `stopSwarm`, `loadAgentProfiles` and 55 more symbols. |
| [oss/vss_extension/media/karl_render.js](../../oss/vss_extension/media/karl_render.js) | Source module for karl render; defines `escapeHtml`, `_triggerScroll`, `_getOrCreateCursor`, `attachStreamingCursor`, `removeStreamingCursor`, `_appendChatText`, `appendThoughtToken` and 32 more symbols. |
| [oss/vss_extension/media/karl_socket.js](../../oss/vss_extension/media/karl_socket.js) | Source module for karl socket; defines `_directSend`, `_startTokenRefreshTimer`, `_clearTokenRefreshTimer`, `_checkTokenRefresh`, `setConnectionState`, `teardownSocket`, `handleDisconnect` and 15 more symbols. |
| [oss/vss_extension/media/karl_state.js](../../oss/vss_extension/media/karl_state.js) | Source module for karl state; defines `hydrate`, `persist`, `initializeAppearance`, `themeById`, `layoutById`, `applyAppearance`, `switchWorkspace`. |
| [oss/vss_extension/media/themes.js](../../oss/vss_extension/media/themes.js) | Source module for themes. |
| [oss/vss_extension/package-lock.json](../../oss/vss_extension/package-lock.json) | Dependency lockfile recording resolved package versions and integrity/source metadata; update through the project’s package manager. |
| [oss/vss_extension/package.json](../../oss/vss_extension/package.json) | Node package manifest for karl; scripts: test; declares package dependencies and metadata. |
| [oss/vss_extension/src/commands.js](../../oss/vss_extension/src/commands.js) | Source module for commands; defines `currentWorkspacePath`, `groupedDiagnostics`, `revealKarlPanel`, `runWorkflow`, `runWorkflowById`, `buildWorkflowContext`, `sendActiveFileToKb` and 1 more symbols. |
| [oss/vss_extension/src/fileOps.js](../../oss/vss_extension/src/fileOps.js) | Source module for fileOps; defines `packageContext`, `writeTempFileAndDiff`. |
| [oss/vss_extension/src/gitOps.js](../../oss/vss_extension/src/gitOps.js) | Source module for gitOps; defines `execGit`, `getGitBranch`. |
| [oss/vss_extension/src/sidebarProvider.js](../../oss/vss_extension/src/sidebarProvider.js) | Source module for sidebarProvider; defines `getDiagnosticsStats`, `sendActiveStateToWebview`, `KarlSidebarProvider`, `clearCache`. |
| [oss/vss_extension/src/tests/run.js](../../oss/vss_extension/src/tests/run.js) | Test implementation for run. |
| [pyproject.toml](../../pyproject.toml) | Configuration or build metadata for pyproject; read alongside the application or tool that loads it. |
| [pytest.ini](../../pytest.ini) | Configuration or build metadata for pytest; read alongside the application or tool that loads it. |
| [raw_test.py](../../raw_test.py) | Quick raw model test -- bypasses all UI/threading. Run: python raw_test.py Connects `os`, `app.engine.model_loader`. |
| [requirements.txt](../../requirements.txt) | Python dependency requirements and optional dependency groups; install with the project’s documented interpreter and environment. |
| [setup_gpu.sh](../../setup_gpu.sh) | Command-line automation for setup gpu; review prerequisites and side effects before executing. |
| [setup_karl.py](../../setup_karl.py) | setup_karl.py — Cross-platform bootstrapper for Karl. Defines `_enable_windows_ansi`, `_ok`, `_info`, `_warn`, `_err`, `_sep` and 22 more top-level symbols. |
| [smoke_test.py](../../smoke_test.py) | Python implementation of smoke test. Connects `sys`, `core.prompt_templates`, `core.workflows`, `eval.graders`. |
| [tests/__init__.py](../../tests/__init__.py) | Initializes the tests Python package and its public imports. Connects `os`, `sys`, `PyQt6.QtWidgets`. |
| [tests/conftest.py](../../tests/conftest.py) | Shared test helpers: skip markers for optional runtime dependencies. Defines `module_available`, `embedding_model_available`, `requires_embedding_model`. |
| [tests/mock_mcp_server.py](../../tests/mock_mcp_server.py) | Mock MCP Server for Testing — Karl Workbench =========================================== A standard stdio-based MCP server using FastMCP. Registers basic tools for integration test… Defines `hello`, `get_version`. |
| [tests/qt_test_helper.py](../../tests/qt_test_helper.py) | Python implementation of qt test helper. Connects `os`, `sys`, `PyQt6.QtWidgets`. |
| [tests/test_adapter_sandbox.py](../../tests/test_adapter_sandbox.py) | Unit Tests for Adapter Sandbox & Speculative Evaluation ========================================================= Verifies evaluate_model_performance and visual PyQt6 sandbox compo… Defines `TestAdapterSandbox`. |
| [tests/test_agent_memory.py](../../tests/test_agent_memory.py) | Automated checks for agent memory. Defines `test_codebase_memory_indexes_functions_classes_and_methods`, `test_query_memory_returns_matching_signature_text`, `test_coder_agent_injects_codebase_signature_reference`. |
| [tests/test_agent_profile_rpc.py](../../tests/test_agent_profile_rpc.py) | Dispatch-level tests for the swarm agent-profile RPC endpoints: get_agent_profiles / save_agent_profile. Defines `_rpc`, `TestAgentProfileRpc`. |
| [tests/test_agent_profile_studio.py](../../tests/test_agent_profile_studio.py) | Tests for app/ui/workspaces/agent_profile_studio.py — the PyQt6 swarm agent profile manager widget. Defines `TestAgentProfileStudioWorkspace`. |
| [tests/test_agentic_loop.py](../../tests/test_agentic_loop.py) | Automated checks for agentic loop. Defines `test_should_continue_stops_on_explicit_final_answer`, `test_should_continue_does_not_stop_on_ordinary_summary_phrase`. |
| [tests/test_ai_lab.py](../../tests/test_ai_lab.py) | Automated checks for ai lab. Defines `TestAILabWorkspace`. |
| [tests/test_app_state_persistence.py](../../tests/test_app_state_persistence.py) | Focused tests for AppState.load_from_disk() and AppState.save_to_disk(). Defines `_start_patches`, `_stop_patches`, `TestAppStatePersistence`, `TestConfigCorruptionRecovery`, `TestConfigVersioning`, `TestConfigFieldValidation`. |
| [tests/test_architecture_audit.py](../../tests/test_architecture_audit.py) | Automated checks for architecture audit. Defines `test_model_loader_blocks_reset_during_active_inference`, `test_generation_threads_use_guarded_model_acquisition`, `test_app_state_queues_cross_thread_signal_emissions`, `test_websocket_broadcast_prunes_failed_clients`, `test_websocket_manager_exposes_thread_safe_client_count`. |
| [tests/test_auto_train.py](../../tests/test_auto_train.py) | Unit Tests for Karl One-Click Auto-Train Flywheel ================================================= Verifies task generation logic, sandboxed verification helper mocking, PyQt6 bac… Defines `_running_under_bwrap`, `TestAutoTrain`. |
| [tests/test_auto_train_lora_config.py](../../tests/test_auto_train_lora_config.py) | Automated checks for auto train lora config. Defines `FakeModel`, `test_detect_target_modules_keeps_known_projection_order`, `test_detect_target_modules_has_attention_fallback`, `test_build_sft_config_enables_packing_and_cosine_schedule`, `test_sft_config_accepts_packing_and_cosine_kwargs_when_trl_available`. |
| [tests/test_codex_integration.py](../../tests/test_codex_integration.py) | Automated checks for codex integration. Defines `TestCodexIntegration`. |
| [tests/test_cognitive_compression.py](../../tests/test_cognitive_compression.py) | Automated checks for cognitive compression. Defines `MockLlama`, `test_cognitive_compression_and_live_stats`, `test_agentic_cognitive_compression_and_live_stats`. |
