# Files 1 of 3 — Karl — Offline LLM Lab

[All pages](../FILE_GUIDE.md) · [Project guide](../PROJECT_GUIDE.md)

| File | Role and useful details |
| --- | --- |
| [.dockerignore](../../.dockerignore) | Tracked .dockerignore resource in .; 438 bytes. |
| [.github/workflows/ci.yml](../../.github/workflows/ci.yml) | GitHub Actions workflow for ci; defines automation triggers, jobs, and build/check steps. |
| [.github/workflows/release.yml](../../.github/workflows/release.yml) | GitHub Actions workflow for release; defines automation triggers, jobs, and build/check steps. |
| [.gitignore](../../.gitignore) | Defines local data, credentials, dependencies, and generated outputs that Git should leave untracked. |
| [.python-version](../../.python-version) | Tracked .python version resource in .; 5 bytes. |
| [AGENTS.md](../../AGENTS.md) | Repository-specific development instructions and continuity rules for coding assistants. |
| [Dockerfile](../../Dockerfile) | Tracked Dockerfile resource in .; 3,691 bytes. |
| [LICENSE](../../LICENSE) | Preserved license, copyright, or attribution terms; read the full text before reusing the covered component. |
| [PROJECT_NOTES.md](../../PROJECT_NOTES.md) | Preserved detailed setup, usage, architecture, and development notes from the earlier README. |
| [README.md](../../README.md) | Project overview, current status, entry points, setup navigation, and documentation links. |
| [app/engine/__init__.py](../../app/engine/__init__.py) | Karl backend package initialization. Connects `app.engine.offline_guard`. |
| [app/engine/agent_memory.py](../../app/engine/agent_memory.py) | Offline code-schema memory for swarm agents. Defines `CodebaseMemory`, `keywords_from_task`. |
| [app/engine/agentic_thread.py](../../app/engine/agentic_thread.py) | Python implementation of agentic thread. Defines `_clamp_max_tokens_to_context`, `_trim_memory`, `_get_gpu_temp`, `_json_object_complete`, `_salvage_compact_trace_fields`, `_emit_compacted_generation` and 1 more top-level symbols. |
| [app/engine/config_store.py](../../app/engine/config_store.py) | Centralized JSON configuration store for Karl. Defines `read_json`, `write_json_atomic`, `get_active_model`, `set_active_model`, `get_active_draft_model`, `set_active_draft_model` and 14 more top-level symbols. |
| [app/engine/event_broker.py](../../app/engine/event_broker.py) | Python implementation of event broker. Defines `EventBroker`. |
| [app/engine/feature_flags.py](../../app/engine/feature_flags.py) | Local Feature Flag Registry and Boot Guard utilities. Defines `FeatureFlagStore`, `check_boot_lock`, `create_boot_lock`, `release_boot_lock`, `run_boot_guard`. |
| [app/engine/hot_reload.py](../../app/engine/hot_reload.py) | Python implementation of hot reload. Defines `HotReloadSignals`, `compile_and_reload`. |
| [app/engine/image_analysis_thread.py](../../app/engine/image_analysis_thread.py) | Python implementation of image analysis thread. Defines `ImageAnalysisThread`. |
| [app/engine/inference_service.py](../../app/engine/inference_service.py) | Inference Service — Karl Engine Defines `InferenceService`. |
| [app/engine/kv_cache.py](../../app/engine/kv_cache.py) | KV-cache statistics and logging helpers. Defines `kv_cache_stats`, `log_cache_stats`. |
| [app/engine/llm_thread.py](../../app/engine/llm_thread.py) | Python implementation of llm thread. Defines `_clamp_max_tokens_to_context`, `_trim_memory`, `_free_vram_mb`, `_write_performance_telemetry`, `_get_gpu_temp`, `_json_object_complete` and 3 more top-level symbols. |
| [app/engine/mcp_client.py](../../app/engine/mcp_client.py) | MCP Client Manager — Karl Workbench ==================================== Manages the lifecycle of multiple stdio-based Model Context Protocol (MCP) servers configured in data/mcp_c… Defines `MCPClientManager`. |
| [app/engine/mini_train_thread.py](../../app/engine/mini_train_thread.py) | MiniGPT Training Thread — Educational Sandbox ============================================= This module handles training the MiniGPT model in a background thread, allowing Karl's U… Defines `MiniTrainThread`. |
| [app/engine/mini_transformer.py](../../app/engine/mini_transformer.py) | Educational Decoder-Only Transformer (Mini-GPT) in PyTorch ========================================================== This module implements a standard GPT-style character-level la… Defines `CharTokenizer`, `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `MiniGPT`. |
| [app/engine/model_load_thread.py](../../app/engine/model_load_thread.py) | Python implementation of model load thread. Defines `ModelLoadThread`. |
| [app/engine/model_loader.py](../../app/engine/model_loader.py) | Python implementation of model loader. Defines `ModelMemoryError`, `CircuitBreakerOpenException`, `ModelCircuitBreaker`, `_LlamaDraftModelAdapter`, `ModelLoader`. |
| [app/engine/offline_guard.py](../../app/engine/offline_guard.py) | Runtime network guard for Karl's offline-first backend. Defines `OfflineNetworkError`, `_truthy`, `online_execution_enabled`, `_is_local_url`, `assert_online_allowed`, `install` and 3 more top-level symbols. |
| [app/engine/quantizer_thread.py](../../app/engine/quantizer_thread.py) | QuantizerThread — background GGUF quantization via llama-quantize CLI ====================================================================== Wraps the `llama-quantize` binary in a… Defines `_locate_llama_quantize`, `_parse_progress`, `QuantizerThread`. |
| [app/engine/remote_rpc_client.py](../../app/engine/remote_rpc_client.py) | Python implementation of remote rpc client. Defines `RemoteRPCError`, `_is_private_host`, `_ssl_context_for_url`, `_with_token_query`, `RemoteRPCModel`. |
| [app/engine/streaming_parser.py](../../app/engine/streaming_parser.py) | Python implementation of streaming parser. Defines `_strip_special_token_leaks`, `ParserFeedResult`, `StreamingThoughtParser`. |
| [app/engine/swarm_agents.py](../../app/engine/swarm_agents.py) | Swarm Agents — Karl Workbench ============================== Defines the specialized ArchitectAgent, CoderAgent, and TesterAgent subclasses orchestrating local LLM calls and system… Defines `_safe_workspace_path`, `register_tool`, `get_tool_schema_block`, `_tool_write_file`, `_tool_read_file`, `_tool_grep_workspace` and 9 more top-level symbols. |
| [app/engine/swarm_judge.py](../../app/engine/swarm_judge.py) | SwarmJudge — heuristic multi-signal scoring for "Multiverse" candidate selection. Defines `_check_syntax`, `_lint_violation_count`, `_diff_size`, `_signature_alignment`, `score_candidate`, `select_winner`. |
| [app/engine/swarm_memory.py](../../app/engine/swarm_memory.py) | SwarmMemory — persistent cross-run failure/fix learning for the swarm. Defines `_keywords`, `fingerprint_failure`, `SwarmMemory`. |
| [app/engine/swarm_orchestrator.py](../../app/engine/swarm_orchestrator.py) | Swarm Orchestrator — Karl Workbench ==================================== Coordinates the multi-agent planning, coding, and testing execution loop. Runs in a background QThread to e… Defines `get_python_imports`, `SwarmSessionState`, `rollback_swarm_run`, `SwarmOrchestratorThread`. |
| [app/engine/swarm_specialists.py](../../app/engine/swarm_specialists.py) | Swarm Specialists — adaptive, LLM-free static-analysis reviewers. Defines `classify_task`, `SecurityAuditorAgent`, `PerformanceAuditorAgent`, `CriticAgent`. |
| [app/engine/task_supervisor.py](../../app/engine/task_supervisor.py) | TaskSupervisor — unified long-running task registry for Karl. Defines `TaskStatus`, `TaskRecord`, `TaskSupervisor`. |
| [app/engine/tool_executor.py](../../app/engine/tool_executor.py) | tool_executor.py — MCP tool call execution for LLMThread. Provides build_tool_schema_prompt() and execute_tool_call() for the chat loop. Defines `build_tool_schema_prompt`, `parse_tool_calls`, `execute_tool_calls`. |
| [app/engine/websocket_server.py](../../app/engine/websocket_server.py) | WebSocket Server Manager — Karl Workbench ========================================== Hosts a local WebSocket server to bridge communication between Karl's Multi-Agent Swarm and edi… Defines `TokenBucket`, `WebSocketServerManager`. |
| [app/repository/session_repository.py](../../app/repository/session_repository.py) | Python implementation of session repository. Defines `SessionRepository`. |
| [app/state.py](../../app/state.py) | Python implementation of state. Defines `AppState`. |
| [app/ui/main_window.py](../../app/ui/main_window.py) | Karl MainWindow — sidebar shell + workspace router. All heavy logic lives in the individual workspace widgets. Defines `_ModelInitThread`, `MainWindow`. |
| [app/ui/sidebar.py](../../app/ui/sidebar.py) | Python implementation of sidebar. Defines `_mono_font`, `_SidebarButton`, `Sidebar`. |
| [app/ui/styles/neutral.qss](../../app/ui/styles/neutral.qss) | Presentation rules for neutral; controls layout, typography, color, states, and responsive behavior. |
| [app/ui/themes.py](../../app/ui/themes.py) | Karl design system — customizable theme engine. Palettes are compiled dynamically into QSS. Defines `darken_hex_color`, `_tint_hex_color`, `hex_to_rgba`, `get_theme_colors`, `get_theme_stylesheet`, `stylesheet`. |
| [app/ui/widgets/__init__.py](../../app/ui/widgets/__init__.py) | Initializes the app/ui/widgets Python package and its public imports. Connects `status_bar`, `tracing_panel`, `glow_panel`. |
| [app/ui/widgets/command_palette.py](../../app/ui/widgets/command_palette.py) | Python implementation of command palette. Defines `CommandPalette`. |
| [app/ui/widgets/glow_panel.py](../../app/ui/widgets/glow_panel.py) | Python implementation of glow panel. Defines `GlowPanel`. |
| [app/ui/widgets/model_combo.py](../../app/ui/widgets/model_combo.py) | Model combo box — reusable model and adapter selector widget. Defines `ModelComboBox`. |
| [app/ui/widgets/shortcuts_overlay.py](../../app/ui/widgets/shortcuts_overlay.py) | Floating, semi-transparent keyboard shortcuts reference overlay. Defines `_kbd`, `ShortcutsOverlay`. |
| [app/ui/widgets/status_bar.py](../../app/ui/widgets/status_bar.py) | Python implementation of status bar. Defines `_lbl`, `_sep`, `StatusBar`. |
| [app/ui/widgets/symbolic_icon.py](../../app/ui/widgets/symbolic_icon.py) | Python implementation of symbolic icon. Defines `symbol_to_qicon`, `BaseSymbol`, `HamburgerIcon`, `BrainIcon`, `ThumbsUpIcon`, `ThumbsDownIcon` and 5 more top-level symbols. |
| [app/ui/widgets/toast.py](../../app/ui/widgets/toast.py) | Python implementation of toast. Defines `ToastOverlay`. |
| [app/ui/widgets/tracing_panel.py](../../app/ui/widgets/tracing_panel.py) | Python implementation of tracing panel. Defines `TracingPanel`. |
| [app/ui/workspaces/__init__.py](../../app/ui/workspaces/__init__.py) | Empty marker or reserved file; contains no executable behavior or data. |
| [app/ui/workspaces/agent_profile_studio.py](../../app/ui/workspaces/agent_profile_studio.py) | Karl Swarm Agent Profile Studio ================================ Provides a PyQt6 user interface to visual customize Specialist Agent roles, system prompt instructions, allowed per… Defines `AgentProfileStudioWorkspace`. |
| [app/ui/workspaces/ai_lab.py](../../app/ui/workspaces/ai_lab.py) | AI Lab Workspace — Interactive pipeline visualizer and agent composer. Defines `_hline`, `_section`, `_label`, `_row`, `ExecutionTraceFlowchart`, `AILabWorkspace`. |
| [app/ui/workspaces/docs.py](../../app/ui/workspaces/docs.py) | Python implementation of docs. Defines `_section`, `_hline`, `DocsWorkspace`. |
| [app/ui/workspaces/docs_data.py](../../app/ui/workspaces/docs_data.py) | Exhaustive reference guides for Karl's Codex workspace. Provides complete, self-contained documentation of grammar, compilers, runtime loops, APIs, and optimization tactics for off… |
| [app/ui/workspaces/eval_suite.py](../../app/ui/workspaces/eval_suite.py) | Eval Suite — run the harness, view results. Defines `_section`, `_hline`, `_EvalThread`, `EvalSuiteWorkspace`. |
| [app/ui/workspaces/flywheel_studio.py](../../app/ui/workspaces/flywheel_studio.py) | Flywheel Studio — Model self-improvement and optimization dashboard. Defines `_section`, `_label`, `CustomLineChart`, `_FlywheelDashboardLoader`, `_LogDecryptThread`, `_TokenLookupThread` and 1 more top-level symbols. |
| [app/ui/workspaces/knowledge_base.py](../../app/ui/workspaces/knowledge_base.py) | Knowledge Base — RAG management workspace. Defines `_hline`, `_section`, `_label`, `_IngestThread`, `_RagOperationThread`, `_SearchThread` and 2 more top-level symbols. |
| [app/ui/workspaces/prompt_lab.py](../../app/ui/workspaces/prompt_lab.py) | Prompt Lab — side-by-side prompt engineering workspace. Defines `_section`, `_hline`, `_scan_model_files`, `generate_char_diff_html`, `_RunThread`, `_PromptColumn` and 4 more top-level symbols. |
| [app/ui/workspaces/swarm_studio.py](../../app/ui/workspaces/swarm_studio.py) | Python implementation of swarm studio. Defines `_TpsChart`, `SwarmStudioWorkspace`. |
| [app/ui/workspaces/system_config/__init__.py](../../app/ui/workspaces/system_config/__init__.py) | Initializes the app/ui/workspaces/system_config Python package and its public imports. Connects `workspace`. |
| [app/ui/workspaces/system_config/adapter_sandbox.py](../../app/ui/workspaces/system_config/adapter_sandbox.py) | Adapter Sandbox Mixin Panel =========================== Provides UI controls to list, load, and benchmark trained LoRA adapters. Defines `BenchmarkWorker`, `GenWorker`, `AdapterSandboxMixin`. |
| [app/ui/workspaces/system_config/appearance_panel.py](../../app/ui/workspaces/system_config/appearance_panel.py) | Python implementation of appearance panel. Defines `AppearancePanelMixin`. |
| [app/ui/workspaces/system_config/appearance_runtime.py](../../app/ui/workspaces/system_config/appearance_runtime.py) | Python implementation of appearance runtime. Defines `AppearanceRuntimeMixin`. |
| [app/ui/workspaces/system_config/common.py](../../app/ui/workspaces/system_config/common.py) | Python implementation of common. Defines `_section`, `_hline`, `_row`. |
| [app/ui/workspaces/system_config/defaults_panel.py](../../app/ui/workspaces/system_config/defaults_panel.py) | Python implementation of defaults panel. Defines `DefaultsPanelMixin`. |
| [app/ui/workspaces/system_config/download_threads.py](../../app/ui/workspaces/system_config/download_threads.py) | Python implementation of download threads. Defines `QuantizationThread`, `QuantizationDialog`, `DownloadThread`. |
| [app/ui/workspaces/system_config/mcp_panel.py](../../app/ui/workspaces/system_config/mcp_panel.py) | Python implementation of mcp panel. Defines `McpPanelMixin`. |
| [app/ui/workspaces/system_config/model_panel.py](../../app/ui/workspaces/system_config/model_panel.py) | Python implementation of model panel. Defines `ModelPanelMixin`. |
| [app/ui/workspaces/system_config/model_preflight.py](../../app/ui/workspaces/system_config/model_preflight.py) | Python implementation of model preflight. Defines `ModelPreflightMixin`. |
| [app/ui/workspaces/system_config/observability_tab.py](../../app/ui/workspaces/system_config/observability_tab.py) | Python implementation of observability tab. Defines `ObservabilityTab`. |
| [app/ui/workspaces/system_config/quantization_panel.py](../../app/ui/workspaces/system_config/quantization_panel.py) | Python implementation of quantization panel. Defines `QuantizationPanelMixin`. |
| [app/ui/workspaces/system_config/registry_panel.py](../../app/ui/workspaces/system_config/registry_panel.py) | Python implementation of registry panel. Defines `RegistryPanelMixin`. |
| [app/ui/workspaces/system_config/vision_hardware_panel.py](../../app/ui/workspaces/system_config/vision_hardware_panel.py) | Python implementation of vision hardware panel. Defines `VisionHardwarePanelMixin`. |
| [app/ui/workspaces/system_config/workspace.py](../../app/ui/workspaces/system_config/workspace.py) | System — hardware info, model management, generation defaults, about. Defines `SystemConfigWorkspace`. |
| [app/ui/workspaces/training_studio/__init__.py](../../app/ui/workspaces/training_studio/__init__.py) | Initializes the app/ui/workspaces/training_studio Python package and its public imports. Defines `TrainingStudioWorkspace`. |
| [app/ui/workspaces/training_studio/auto_train_tab.py](../../app/ui/workspaces/training_studio/auto_train_tab.py) | Python implementation of auto train tab. Defines `_section`, `_hline`, `AutoTrainTab`. |
| [app/ui/workspaces/training_studio/dataset_tab.py](../../app/ui/workspaces/training_studio/dataset_tab.py) | Python implementation of dataset tab. Defines `DatasetListModel`, `_section`, `_MergeThread`, `DatasetTab`. |
| [app/ui/workspaces/training_studio/export_tab.py](../../app/ui/workspaces/training_studio/export_tab.py) | Python implementation of export tab. Defines `_section`, `ExportTab`. |
| [app/ui/workspaces/training_studio/flywheel_tab.py](../../app/ui/workspaces/training_studio/flywheel_tab.py) | Python implementation of flywheel tab. Defines `_section`, `_hline`, `FlywheelTab`. |
| [app/ui/workspaces/training_studio/mini_gpt_tab.py](../../app/ui/workspaces/training_studio/mini_gpt_tab.py) | Python implementation of mini gpt tab. Defines `_section`, `_hline`, `LossChartWidget`, `MiniGptTab`. |
| [app/ui/workspaces/training_studio/threads.py](../../app/ui/workspaces/training_studio/threads.py) | Python implementation of threads. Defines `TrainingThread`, `AutoTrainThread`, `_FlywheelStatsThread`, `MiniTrainThread`. |
| [app/ui/workspaces/training_studio/train_tab.py](../../app/ui/workspaces/training_studio/train_tab.py) | Python implementation of train tab. Defines `_section`, `_hline`, `TrainTab`. |
| [app/ui/workspaces/vision_workbench.py](../../app/ui/workspaces/vision_workbench.py) | Python implementation of vision workbench. Defines `VisionWorkbench`. |
| [app/ui/workspaces/workbench/__init__.py](../../app/ui/workspaces/workbench/__init__.py) | Workbench package — primary interaction space. Connects `app.ui.workspaces.workbench.chat_view`, `app.ui.workspaces.workbench.profiles`, `app.ui.workspaces.workbench.workspace`. |
| [app/ui/workspaces/workbench/chat_view.py](../../app/ui/workspaces/workbench/chat_view.py) | Scrollable conversation display with streaming support. Defines `_escape`, `_escape_pre`, `ChatView`. |
| [app/ui/workspaces/workbench/profiles.py](../../app/ui/workspaces/workbench/profiles.py) | Agent profile definitions injected into the active system prompt. Defines `_load_overrides`, `_save_overrides`, `save_profile_override`, `delete_profile_override`, `get_default_prompt`, `is_overridden` and 1 more top-level symbols. |
| [app/ui/workspaces/workbench/workspace.py](../../app/ui/workspaces/workbench/workspace.py) | Workbench — primary interaction space. Defines `_hline`, `_label`, `_load_persisted_system_prompt`, `_save_system_prompt`, `WorkbenchWorkspace`. |
| [app/utils/codebase_search.py](../../app/utils/codebase_search.py) | Python implementation of codebase search. Defines `codebase_search`. |
| [app/utils/compactor.py](../../app/utils/compactor.py) | Trace compaction for model-facing "Machine Speak" datasets. Defines `compact_trace_for_ai`, `decompact_trace`, `_compact_value`, `_decompact_value`, `_compact_text`, `_strip_boilerplate` and 6 more top-level symbols. |
| [app/utils/conversion/__init__.py](../../app/utils/conversion/__init__.py) | Initializes the app/utils/conversion Python package and its public imports. Defines `load_all_models`, `get_model_class`, `print_registered_models`. |
| [app/utils/conversion/afmoe.py](../../app/utils/conversion/afmoe.py) | Python implementation of afmoe. Defines `AfmoeModel`. |
| [app/utils/conversion/arctic.py](../../app/utils/conversion/arctic.py) | Python implementation of arctic. Defines `ArcticModel`. |
| [app/utils/conversion/baichuan.py](../../app/utils/conversion/baichuan.py) | Python implementation of baichuan. Defines `BaichuanModel`. |
| [app/utils/conversion/bailingmoe.py](../../app/utils/conversion/bailingmoe.py) | Python implementation of bailingmoe. Defines `BailingMoeModel`, `BailingMoeV2Model`, `SarvamMoEModel`. |
| [app/utils/conversion/base.py](../../app/utils/conversion/base.py) | Python implementation of base. Defines `SentencePieceTokenTypes`, `ModelType`, `ModelBase`, `TextModel`, `MmprojModel`, `LazyTorchTensor` and 1 more top-level symbols. |
| [app/utils/conversion/bert.py](../../app/utils/conversion/bert.py) | Python implementation of bert. Defines `BertModel`, `DistilBertModel`, `RobertaModel`, `NomicBertModel`, `NeoBert`, `EuroBertModel` and 3 more top-level symbols. |
| [app/utils/conversion/bitnet.py](../../app/utils/conversion/bitnet.py) | Python implementation of bitnet. Defines `BitnetModel`. |
| [app/utils/conversion/bloom.py](../../app/utils/conversion/bloom.py) | Python implementation of bloom. Defines `BloomModel`. |
| [app/utils/conversion/chameleon.py](../../app/utils/conversion/chameleon.py) | Python implementation of chameleon. Defines `ChameleonModel`. |
| [app/utils/conversion/chatglm.py](../../app/utils/conversion/chatglm.py) | Python implementation of chatglm. Defines `ChatGLMModel`. |
| [app/utils/conversion/codeshell.py](../../app/utils/conversion/codeshell.py) | Python implementation of codeshell. Defines `CodeShellModel`. |
| [app/utils/conversion/cogvlm.py](../../app/utils/conversion/cogvlm.py) | Python implementation of cogvlm. Defines `CogVLMVisionModel`, `CogVLMModel`. |
| [app/utils/conversion/command_r.py](../../app/utils/conversion/command_r.py) | Python implementation of command r. Defines `CommandR2Model`, `Cohere2Model`. |
| [app/utils/conversion/dbrx.py](../../app/utils/conversion/dbrx.py) | Python implementation of dbrx. Defines `DbrxModel`. |
| [app/utils/conversion/deci.py](../../app/utils/conversion/deci.py) | Python implementation of deci. Defines `DeciModel`. |
| [app/utils/conversion/deepseek.py](../../app/utils/conversion/deepseek.py) | Python implementation of deepseek. Defines `DeepseekOCRVisionModel`, `DeepseekOCR2VisionModel`, `DeepseekModel`, `DeepseekV2Model`, `DeepseekV32Model`. |
| [app/utils/conversion/dots1.py](../../app/utils/conversion/dots1.py) | Python implementation of dots1. Defines `Dots1Model`. |
| [app/utils/conversion/dotsocr.py](../../app/utils/conversion/dotsocr.py) | Python implementation of dotsocr. Defines `DotsOCRVisionModel`. |
| [app/utils/conversion/dream.py](../../app/utils/conversion/dream.py) | Python implementation of dream. Defines `DreamModel`. |
| [app/utils/conversion/ernie.py](../../app/utils/conversion/ernie.py) | Python implementation of ernie. Defines `Ernie4_5Model`, `Ernie4_5MoeModel`, `PaddleOCRModel`, `PaddleOCRVisionModel`. |
| [app/utils/conversion/exaone.py](../../app/utils/conversion/exaone.py) | Python implementation of exaone. Defines `ExaoneModel`, `Exaone4Model`, `ExaoneMoEModel`. |
| [app/utils/conversion/falcon.py](../../app/utils/conversion/falcon.py) | Python implementation of falcon. Defines `FalconModel`. |
| [app/utils/conversion/falcon_h1.py](../../app/utils/conversion/falcon_h1.py) | Python implementation of falcon h1. Defines `FalconH1Model`. |
| [app/utils/conversion/gemma.py](../../app/utils/conversion/gemma.py) | Python implementation of gemma. Defines `GemmaModel`, `Gemma2Model`, `Gemma3Model`, `EmbeddingGemma`, `Gemma3VisionModel`, `ConformerAudioModel` and 4 more top-level symbols. |
| [app/utils/conversion/glm.py](../../app/utils/conversion/glm.py) | Python implementation of glm. Defines `Glm4Model`, `GlmOCRModel`, `Glm4MoeModel`, `Glm4MoeLiteModel`, `GlmMoeDsaModel`, `SolarOpenModel`. |
| [app/utils/conversion/gpt2.py](../../app/utils/conversion/gpt2.py) | Python implementation of gpt2. Defines `GPT2Model`, `RuGPT3XLModel`. |
| [app/utils/conversion/gpt_oss.py](../../app/utils/conversion/gpt_oss.py) | Python implementation of gpt oss. Defines `GptOssModel`. |
| [app/utils/conversion/gptneox.py](../../app/utils/conversion/gptneox.py) | Python implementation of gptneox. Defines `GPTNeoXModel`. |
| [app/utils/conversion/granite.py](../../app/utils/conversion/granite.py) | Python implementation of granite. Defines `GraniteModel`, `GraniteMoeModel`, `GraniteHybridModel`, `GraniteSpeechMmprojModel`. |
| [app/utils/conversion/grok.py](../../app/utils/conversion/grok.py) | Python implementation of grok. Defines `GrokModel`. |
| [app/utils/conversion/grovemoe.py](../../app/utils/conversion/grovemoe.py) | Python implementation of grovemoe. Defines `GroveMoeModel`. |
| [app/utils/conversion/hunyuan.py](../../app/utils/conversion/hunyuan.py) | Python implementation of hunyuan. Defines `HunYuanMoEModel`, `HunYuanModel`, `HunyuanVLVisionModel`, `HunyuanVLTextModel`. |
| [app/utils/conversion/internlm.py](../../app/utils/conversion/internlm.py) | Python implementation of internlm. Defines `InternLM2Model`, `InternLM3Model`. |
| [app/utils/conversion/internvl.py](../../app/utils/conversion/internvl.py) | Python implementation of internvl. Defines `InternVisionModel`. |
| [app/utils/conversion/jais.py](../../app/utils/conversion/jais.py) | Python implementation of jais. Defines `Jais2Model`, `JaisModel`. |
| [app/utils/conversion/jamba.py](../../app/utils/conversion/jamba.py) | Python implementation of jamba. Defines `JambaModel`. |
| [app/utils/conversion/januspro.py](../../app/utils/conversion/januspro.py) | Python implementation of januspro. Defines `JanusProModel`, `JanusProVisionModel`. |
| [app/utils/conversion/kimi_linear.py](../../app/utils/conversion/kimi_linear.py) | Python implementation of kimi linear. Defines `KimiLinearModel`. |
| [app/utils/conversion/kimivl.py](../../app/utils/conversion/kimivl.py) | Python implementation of kimivl. Defines `KimiVLModel`, `KimiK25Model`. |
| [app/utils/conversion/lfm2.py](../../app/utils/conversion/lfm2.py) | Python implementation of lfm2. Defines `LFM2Model`, `LFM2ColBertModel`, `LFM2MoeModel`, `LFM2VLModel`, `LFM2AudioModel`, `LFM25AudioTokenizer`. |
| [app/utils/conversion/lighton_ocr.py](../../app/utils/conversion/lighton_ocr.py) | Python implementation of lighton ocr. Defines `LightOnOCRVisionModel`. |
| [app/utils/conversion/llada.py](../../app/utils/conversion/llada.py) | Python implementation of llada. Defines `LLaDAModel`, `LLaDAMoEModel`. |
| [app/utils/conversion/llama.py](../../app/utils/conversion/llama.py) | Python implementation of llama. Defines `LlamaModel`, `ArceeModel`, `Llama4Model`, `LlamaEmbedNemotronModel`, `SmolLM3Model`, `ApertusModel`. |
| [app/utils/conversion/llama4.py](../../app/utils/conversion/llama4.py) | Python implementation of llama4. Defines `Llama4VisionModel`. |
| [app/utils/conversion/llava.py](../../app/utils/conversion/llava.py) | Python implementation of llava. Defines `LlavaVisionModel`. |
| [app/utils/conversion/maincoder.py](../../app/utils/conversion/maincoder.py) | Python implementation of maincoder. Defines `MaincoderModel`. |
| [app/utils/conversion/mamba.py](../../app/utils/conversion/mamba.py) | Python implementation of mamba. Defines `MambaModel`, `Mamba2Model`. |
| [app/utils/conversion/mimo.py](../../app/utils/conversion/mimo.py) | Python implementation of mimo. Defines `MimoV2Model`, `MiMoV2VisionModel`. |
| [app/utils/conversion/minicpm.py](../../app/utils/conversion/minicpm.py) | Python implementation of minicpm. Defines `MiniCPMModel`, `MiniCPM3Model`, `MiniCPMV4_6TextModel`, `MiniCPMV4_6VisionModel`. |
| [app/utils/conversion/minimax.py](../../app/utils/conversion/minimax.py) | Python implementation of minimax. Defines `MiniMaxM2Model`. |
| [app/utils/conversion/mistral.py](../../app/utils/conversion/mistral.py) | Python implementation of mistral. Defines `MistralModel`, `MistralMoeModel`. |
| [app/utils/conversion/mistral3.py](../../app/utils/conversion/mistral3.py) | Python implementation of mistral3. Defines `Mistral3Model`. |
| [app/utils/conversion/mpt.py](../../app/utils/conversion/mpt.py) | Python implementation of mpt. Defines `MPTModel`. |
| [app/utils/conversion/nemotron.py](../../app/utils/conversion/nemotron.py) | Python implementation of nemotron. Defines `NemotronNanoV2VLModel`, `NemotronModel`, `NemotronHModel`. |
| [app/utils/conversion/olmo.py](../../app/utils/conversion/olmo.py) | Python implementation of olmo. Defines `OlmoModel`, `SeedOssModel`, `Olmo2Model`, `OlmoeModel`. |
| [app/utils/conversion/openelm.py](../../app/utils/conversion/openelm.py) | Python implementation of openelm. Defines `OpenELMModel`. |
| [app/utils/conversion/orion.py](../../app/utils/conversion/orion.py) | Python implementation of orion. Defines `OrionModel`. |
| [app/utils/conversion/pangu.py](../../app/utils/conversion/pangu.py) | Python implementation of pangu. Defines `PanguEmbeddedModel`. |
| [app/utils/conversion/phi.py](../../app/utils/conversion/phi.py) | Python implementation of phi. Defines `Phi2Model`, `Phi3MiniModel`, `Phi4VisionMmprojModel`, `PhiMoeModel`. |
| [app/utils/conversion/pixtral.py](../../app/utils/conversion/pixtral.py) | Python implementation of pixtral. Defines `PixtralModel`. |
| [app/utils/conversion/plamo.py](../../app/utils/conversion/plamo.py) | Python implementation of plamo. Defines `PlamoModel`, `Plamo2Model`, `Plamo3Model`. |
| [app/utils/conversion/plm.py](../../app/utils/conversion/plm.py) | Python implementation of plm. Defines `PLMModel`. |
| [app/utils/conversion/qwen.py](../../app/utils/conversion/qwen.py) | Python implementation of qwen. Defines `QwenModel`, `Qwen2Model`, `Qwen2MoeModel`, `Qwen3Model`, `Qwen3MoeModel`, `Qwen3NextModel` and 6 more top-level symbols. |
| [app/utils/conversion/qwen3vl.py](../../app/utils/conversion/qwen3vl.py) | Python implementation of qwen3vl. Defines `Qwen3VLVisionModel`, `Qwen3OmniMmprojModel`, `Qwen3ASRMmprojModel`, `Glm4VVisionModel`, `Qwen3VLTextModel`, `Qwen3VLMoeTextModel` and 2 more top-level symbols. |
| [app/utils/conversion/qwenvl.py](../../app/utils/conversion/qwenvl.py) | Python implementation of qwenvl. Defines `Qwen2VLModel`, `Qwen2VLVisionModel`, `Qwen25AudioModel`, `Qwen25OmniModel`. |
| [app/utils/conversion/refact.py](../../app/utils/conversion/refact.py) | Python implementation of refact. Defines `RefactModel`. |
| [app/utils/conversion/rwkv.py](../../app/utils/conversion/rwkv.py) | Python implementation of rwkv. Defines `Rwkv6Model`, `RWKV6Qwen2Model`, `Rwkv7Model`, `ARwkv7Model`. |
| [app/utils/conversion/sarashina2.py](../../app/utils/conversion/sarashina2.py) | Python implementation of sarashina2. Defines `Sarashina2VLTextModel`, `Sarashina2VLVisionModel`. |
| [app/utils/conversion/smallthinker.py](../../app/utils/conversion/smallthinker.py) | Python implementation of smallthinker. Defines `SmallThinkerModel`. |
| [app/utils/conversion/smolvlm.py](../../app/utils/conversion/smolvlm.py) | Python implementation of smolvlm. Defines `SmolVLMModel`. |
| [app/utils/conversion/stablelm.py](../../app/utils/conversion/stablelm.py) | Python implementation of stablelm. Defines `StableLMModel`. |
| [app/utils/conversion/starcoder.py](../../app/utils/conversion/starcoder.py) | Python implementation of starcoder. Defines `StarCoderModel`, `StarCoder2Model`. |
| [app/utils/conversion/step3.py](../../app/utils/conversion/step3.py) | Python implementation of step3. Defines `Step3VLVisionModel`, `Step3VLTextModel`, `Step35Model`. |
| [app/utils/conversion/t5.py](../../app/utils/conversion/t5.py) | Python implementation of t5. Defines `T5Model`, `T5EncoderModel`. |
| [app/utils/conversion/talkie.py](../../app/utils/conversion/talkie.py) | Python implementation of talkie. Defines `TalkieModel`. |
| [app/utils/conversion/ultravox.py](../../app/utils/conversion/ultravox.py) | Python implementation of ultravox. Defines `UltravoxModel`, `GlmASRWhisperEncoderModel`, `WhisperEncoderModel`, `UltravoxWhisperEncoderModel`, `MERaLiONWhisperEncoderModel`, `VoxtralWhisperEncoderModel` and 1 more top-level symbols. |
| [app/utils/conversion/wavtokenizer.py](../../app/utils/conversion/wavtokenizer.py) | Python implementation of wavtokenizer. Defines `WavTokenizerDecModel`. |
| [app/utils/conversion/xverse.py](../../app/utils/conversion/xverse.py) | Python implementation of xverse. Defines `XverseModel`. |
| [app/utils/conversion/youtuvl.py](../../app/utils/conversion/youtuvl.py) | Python implementation of youtuvl. Defines `YoutuVLVisionModel`. |
| [app/utils/convert_lora_to_gguf.py](../../app/utils/convert_lora_to_gguf.py) | Python implementation of convert lora to gguf. Defines `PartialLoraTensor`, `LoraTorchTensor`, `get_base_tensor_name`, `parse_args`, `load_hparams_from_hf`. |
| [app/utils/correlation_logger.py](../../app/utils/correlation_logger.py) | Correlation ID context tracking and structured logging filter. Defines `set_correlation_id`, `get_correlation_id`, `reset_correlation_id`, `new_correlation_id`, `CorrelationFilter`. |
| [app/utils/custom_embeddings.py](../../app/utils/custom_embeddings.py) | Custom Embeddings Engine — Pure Python & NumPy TF-IDF Vectorizer =============================================================== This module implements a transparent, lightweight T… Defines `TfidfEmbedder`. |
| [app/utils/dataset_merger.py](../../app/utils/dataset_merger.py) | Dataset Merger — Multi-User Trace Import ========================================= Merges curated.jsonl files from multiple developers without corrupting the local training records… Defines `_assistant_content`, `_is_anomalous`, `_exact_hash`, `_prompt_key`, `_parse_ts`, `_load_jsonl` and 2 more top-level symbols. |
| [app/utils/db_pool.py](../../app/utils/db_pool.py) | Thread-safe SQLite connection pool for Karl's metadata store. Defines `SQLiteConnectionPool`. |
| [app/utils/diagnostics.py](../../app/utils/diagnostics.py) | Karl Startup Diagnostics Engine ================================= Invoked via: python main.py --diagnose Defines `_c`, `_green`, `_red`, `_yellow`, `_cyan`, `_bold` and 13 more top-level symbols. |
| [app/utils/ipc_helper.py](../../app/utils/ipc_helper.py) | Karl IPC Helper — secure shared memory allocation and sanitization. Defines `_process_termination_sanitization`, `SanitizedSharedMemory`, `SharedMemoryManager`. |
