# ==============================================================================
# Program: __init__.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Weight conversion and architectural mapping implementation for __INIT__.
#          Transforms HuggingFace checkpoint tensors, normalizes layer weights,
#          and writes compliant GGUF data structures for llama.cpp.
# ==============================================================================

# Package overview: this package is a split-up copy of llama.cpp's
# convert_hf_to_gguf.py. Each architecture lives in its own module and registers
# its converter class with ModelBase via a decorator. This __init__ keeps two
# lookup tables (HF architecture name -> module name) so that only the module
# needed for a given checkpoint is imported, and re-exports the public base API.
# How conversion is dispatched
#   1. A caller (convert_lora_to_gguf.py, which auto_train.py and the Training Studio threads run) reads the base model's
#      config.json and takes architectures[0].
#   2. get_model_class(name) looks that string up in TEXT_MODEL_MAP (or MMPROJ_MODEL_MAP for vision/audio encoders) to find
#      the module, imports ONLY that module (cheap start-up), and returns the class the module registered.
#   3. The class (a TextModel or MmprojModel subclass from base.py) writes the GGUF file.
# Adding an architecture: create <module>.py with a class decorated @ModelBase.register("<HF architecture>"), then add the
# architecture string to the right map below. A missing map entry raises NotImplementedError in get_model_class.
# This package is excluded from ruff (pyproject.toml) because it mirrors upstream llama.cpp code style.
from __future__ import annotations

# Re-export the shared base classes/helpers so callers can import everything
# from `conversion` without knowing about conversion.base.
from .base import (
    ModelBase, TextModel, MmprojModel, ModelType, SentencePieceTokenTypes,
    logger, _mistral_common_installed, _mistral_import_error_msg,
    get_model_architecture, LazyTorchTensor,
)
import importlib
from typing import Type


# Public surface of the package (underscore names are re-exported on purpose
# because callers check mistral_common availability through them).
__all__ = [
    "ModelBase", "TextModel", "MmprojModel", "ModelType", "SentencePieceTokenTypes",
    "get_model_architecture", "LazyTorchTensor", "logger",
    "_mistral_common_installed", "_mistral_import_error_msg",
    "get_model_class", "print_registered_models", "load_all_models",
]


# Text (language-model) architectures: HuggingFace `architectures[0]` string ->
# name of the conversion module whose class handles it. Several architectures
# share one module (e.g. all BERT variants map to "bert"). Keys are matched
# exactly, so a new architecture needs an entry here AND a registered class.
TEXT_MODEL_MAP: dict[str, str] = {
    # Map layout: keys are HuggingFace `architectures[0]` strings, values are module file names in this package; the
    # list is alphabetical by architecture name, so related families are not always adjacent.
    "AfmoeForCausalLM": "afmoe",
    # Several decoder-only architectures reuse the "llama" module because their tensor layout is LLaMA-compatible
    # (Mistral/Mixtral and Llama 4 are routed there too, see below).
    "ApertusForCausalLM": "llama",
    "ArceeForCausalLM": "llama",
    "ArcticForCausalLM": "arctic",
    # Audio-capable multimodal checkpoints appear in BOTH tables: here the module converts the language-model half; the encoder half is picked up from MMPROJ_MODEL_MAP below. The same HF architecture string therefore yields two GGUF files (text + mmproj).
    "AudioFlamingo3ForConditionalGeneration": "qwen",
    "BaiChuanForCausalLM": "baichuan",
    "BaichuanForCausalLM": "baichuan",
    "BailingMoeForCausalLM": "bailingmoe",
    "BailingMoeV2ForCausalLM": "bailingmoe",
    "BambaForCausalLM": "granite",
    # BERT-family encoders (BERT, DistilBERT, Camembert, ...) share one module because they use the same transformer-encoder tensor layout; only the head differs.
    "BertForMaskedLM": "bert",
    "BertForSequenceClassification": "bert",
    "BertModel": "bert",
    "BitnetForCausalLM": "bitnet",
    "BloomForCausalLM": "bloom",
    "BloomModel": "bloom",
    "CamembertModel": "bert",
    "ChameleonForCausalLM": "chameleon",
    "ChameleonForConditionalGeneration": "chameleon",
    "ChatGLMForConditionalGeneration": "chatglm",
    "ChatGLMModel": "chatglm",
    "CodeShellForCausalLM": "codeshell",
    "CogVLMForCausalLM": "cogvlm",
    "Cohere2ForCausalLM": "command_r",
    "CohereForCausalLM": "command_r",
    "DbrxForCausalLM": "dbrx",
    "DeciLMForCausalLM": "deci",
    "DeepseekForCausalLM": "deepseek",
    # DeepSeek V1 has its own class (DeepseekModel); V2 and V3 share DeepseekV2Model, and V3.2 subclasses that, so all of them live in one module.
    "DeepseekV2ForCausalLM": "deepseek",
    "DeepseekV3ForCausalLM": "deepseek",
    "DeepseekV32ForCausalLM": "deepseek",
    "DistilBertForMaskedLM": "bert",
    "DistilBertForSequenceClassification": "bert",
    "DistilBertModel": "bert",
    "Dots1ForCausalLM": "dots1",
    "DotsOCRForCausalLM": "qwen",
    "DreamModel": "dream",
    "Ernie4_5ForCausalLM": "ernie",
    "Ernie4_5_ForCausalLM": "ernie",
    "Ernie4_5_MoeForCausalLM": "ernie",
    "EuroBertModel": "bert",
    "Exaone4ForCausalLM": "exaone",
    "ExaoneForCausalLM": "exaone",
    "ExaoneMoEForCausalLM": "exaone",
    "FalconForCausalLM": "falcon",
    "FalconH1ForCausalLM": "falcon_h1",
    # State-space (Mamba/Mamba2) models are converted by the "mamba" module; they have no attention tensors.
    "FalconMambaForCausalLM": "mamba",
    "GPT2LMHeadModel": "gpt2",
    # StarCoder family (GPT-BigCode) shares the "starcoder" module.
    "GPTBigCodeForCausalLM": "starcoder",
    "GPTNeoXForCausalLM": "gptneox",
    "GPTRefactForCausalLM": "refact",
    "Gemma2ForCausalLM": "gemma",
    "Gemma3ForCausalLM": "gemma",
    "Gemma3ForConditionalGeneration": "gemma",
    "Gemma3TextModel": "gemma",
    "Gemma3nForCausalLM": "gemma",
    "Gemma3nForConditionalGeneration": "gemma",
    "Gemma4ForConditionalGeneration": "gemma",
    "Gemma4ForCausalLM": "gemma",
    "GemmaForCausalLM": "gemma",
    "Glm4ForCausalLM": "glm",
    "Glm4MoeForCausalLM": "glm",
    "Glm4MoeLiteForCausalLM": "glm",
    "Glm4vForConditionalGeneration": "glm",
    "Glm4vMoeForConditionalGeneration": "glm",
    "GlmForCausalLM": "chatglm",
    "GlmMoeDsaForCausalLM": "glm",
    "GlmOcrForConditionalGeneration": "glm",
    "GptOssForCausalLM": "gpt_oss",
    "GraniteForCausalLM": "granite",
    "GraniteMoeForCausalLM": "granite",
    "GraniteMoeHybridForCausalLM": "granite",
    "GraniteMoeSharedForCausalLM": "granite",
    "GraniteSpeechForConditionalGeneration": "granite",
    "Grok1ForCausalLM": "grok",
    "GrokForCausalLM": "grok",
    "GroveMoeForCausalLM": "grovemoe",
    "HunYuanDenseV1ForCausalLM": "hunyuan",
    "HunYuanMoEV1ForCausalLM": "hunyuan",
    "HunYuanVLForConditionalGeneration": "hunyuan",
    "IQuestCoderForCausalLM": "llama",
    "InternLM2ForCausalLM": "internlm",
    "InternLM3ForCausalLM": "internlm",
    "JAISLMHeadModel": "jais",
    "Jais2ForCausalLM": "jais",
    "JambaForCausalLM": "jamba",
    "JanusForConditionalGeneration": "januspro",
    "JinaBertForMaskedLM": "bert",
    "JinaBertModel": "bert",
    "JinaEmbeddingsV5Model": "bert",
    "KORMoForCausalLM": "qwen",
    "KimiK25ForConditionalGeneration": "deepseek",
    "KimiLinearForCausalLM": "kimi_linear",
    "KimiLinearModel": "kimi_linear",
    "KimiVLForConditionalGeneration": "deepseek",
    "LFM2ForCausalLM": "lfm2",
    "LLaDAMoEModel": "llada",
    "LLaDAMoEModelLM": "llada",
    "LLaDAModelLM": "llada",
    "LLaMAForCausalLM": "llama",
    "Lfm25AudioTokenizer": "lfm2",
    "Lfm2ForCausalLM": "lfm2",
    "Lfm2Model": "lfm2",
    "Lfm2MoeForCausalLM": "lfm2",
    "Llama4ForCausalLM": "llama",
    "Llama4ForConditionalGeneration": "llama",
    "LlamaBidirectionalModel": "llama",
    "LlamaForCausalLM": "llama",
    "LlamaModel": "llama",
    "LlavaForConditionalGeneration": "llama",
    "LlavaStableLMEpochForCausalLM": "stablelm",
    "MPTForCausalLM": "mpt",
    "MT5ForConditionalGeneration": "t5",
    "MaincoderForCausalLM": "maincoder",
    "Mamba2ForCausalLM": "mamba",
    "MambaForCausalLM": "mamba",
    "MambaLMHeadModel": "mamba",
    "MiMoV2FlashForCausalLM": "mimo",
    "MiMoV2ForCausalLM": "mimo",
    "MiniCPM3ForCausalLM": "minicpm",
    "MiniCPMForCausalLM": "minicpm",
    "MiniCPMV4_6ForConditionalGeneration": "minicpm",
    "MiniMaxM2ForCausalLM": "minimax",
    "Ministral3ForCausalLM": "mistral3",
    "Mistral3ForConditionalGeneration": "mistral3",
    "MistralForCausalLM": "llama",
    "MixtralForCausalLM": "llama",
    "ModernBertForMaskedLM": "bert",
    "ModernBertForSequenceClassification": "bert",
    "ModernBertModel": "bert",
    "NemotronForCausalLM": "nemotron",
    "NemotronHForCausalLM": "nemotron",
    "NeoBERT": "bert",
    "NeoBERTForSequenceClassification": "bert",
    "NeoBERTLMHead": "bert",
    "NomicBertModel": "bert",
    "OLMoForCausalLM": "olmo",
    "Olmo2ForCausalLM": "olmo",
    "Olmo3ForCausalLM": "olmo",
    "OlmoForCausalLM": "olmo",
    "OlmoeForCausalLM": "olmo",
    "OpenELMForCausalLM": "openelm",
    "OrionForCausalLM": "orion",
    "PLMForCausalLM": "plm",
    "PLaMo2ForCausalLM": "plamo",
    "PLaMo3ForCausalLM": "plamo",
    "PaddleOCRVLForConditionalGeneration": "ernie",
    "PanguEmbeddedForCausalLM": "pangu",
    "Phi3ForCausalLM": "phi",
    "Phi4ForCausalLMV": "phi",
    "PhiForCausalLM": "phi",
    "PhiMoEForCausalLM": "phi",
    "Plamo2ForCausalLM": "plamo",
    "Plamo3ForCausalLM": "plamo",
    "PlamoForCausalLM": "plamo",
    "QWenLMHeadModel": "qwen",
    "Qwen2AudioForConditionalGeneration": "qwen",
    "Qwen2ForCausalLM": "qwen",
    "Qwen2Model": "qwen",
    "Qwen2MoeForCausalLM": "qwen",
    "Qwen2VLForConditionalGeneration": "qwenvl",
    "Qwen2VLModel": "qwenvl",
    "Qwen2_5OmniModel": "qwenvl",
    "Qwen2_5_VLForConditionalGeneration": "qwenvl",
    "Qwen3ASRForConditionalGeneration": "qwen3vl",
    "Qwen3ForCausalLM": "qwen",
    "Qwen3Model": "qwen",
    "Qwen3MoeForCausalLM": "qwen",
    "Qwen3NextForCausalLM": "qwen",
    "Qwen3OmniMoeForConditionalGeneration": "qwen3vl",
    "Qwen3VLForConditionalGeneration": "qwen3vl",
    "Qwen3VLMoeForConditionalGeneration": "qwen3vl",
    "Qwen3_5ForCausalLM": "qwen",
    "Qwen3_5ForConditionalGeneration": "qwen",
    "Qwen3_5MoeForCausalLM": "qwen",
    "Qwen3_5MoeForConditionalGeneration": "qwen",
    "RND1": "qwen",
    "RWForCausalLM": "falcon",
    # RWKV (recurrent) architectures live in the "rwkv" module; they have time-mix/channel-mix tensors instead of attention.
    "RWKV6Qwen2ForCausalLM": "rwkv",
    "RWKV7ForCausalLM": "rwkv",
    "RobertaForSequenceClassification": "bert",
    "RobertaModel": "bert",
    "RuGPT3XLForCausalLM": "gpt2",
    "Rwkv6ForCausalLM": "rwkv",
    "Rwkv7ForCausalLM": "rwkv",
    "RwkvHybridForCausalLM": "rwkv",
    "Sarashina2VisionForCausalLM": "sarashina2",
    "SarvamMoEForCausalLM": "bailingmoe",
    "SeedOssForCausalLM": "olmo",
    "SmallThinkerForCausalLM": "smallthinker",
    "SmolLM3ForCausalLM": "llama",
    "SolarOpenForCausalLM": "glm",
    "StableLMEpochForCausalLM": "stablelm",
    "StableLmForCausalLM": "stablelm",
    "Starcoder2ForCausalLM": "starcoder",
    # Step-3 family models are converted by the "step3" module.
    "Step3p5ForCausalLM": "step3",
    "StepVLForConditionalGeneration": "step3",
    "T5EncoderModel": "t5",
    "T5ForConditionalGeneration": "t5",
    "T5WithLMHeadModel": "t5",
    "TalkieForCausalLM": "talkie",
    "UMT5ForConditionalGeneration": "t5",
    "UMT5Model": "t5",
    # Whisper-style audio encoders (Voxtral, Qwen2-Audio, GLM-ASR, MERaLiON2, Ultravox) are converted by classes in the ultravox module; the text-side UltravoxModel entry only exists to raise and point users at --mmproj.
    "UltravoxModel": "ultravox",
    "VLlama3ForCausalLM": "llama",
    "VoxtralForConditionalGeneration": "llama",
    "WavTokenizerDec": "wavtokenizer",
    "XLMRobertaForSequenceClassification": "bert",
    "XLMRobertaModel": "bert",
    "XverseForCausalLM": "xverse",
    "YoutuForCausalLM": "deepseek",
    "YoutuVLForConditionalGeneration": "deepseek",
    "modeling_grove_moe.GroveMoeForCausalLM": "grovemoe",
    # Some trust_remote_code checkpoints report 'module.Class' as their architecture string, so the module-prefixed spelling needs its own key next to the plain class name.
    "modeling_sarvam_moe.SarvamMoEForCausalLM": "bailingmoe",
}


# Multimodal projector (mmproj) architectures. Same idea as TEXT_MODEL_MAP but
# used when converting the vision/audio encoder to a separate GGUF; the same HF
# architecture may map to a different module than in the text table.
MMPROJ_MODEL_MAP: dict[str, str] = {
    "AudioFlamingo3ForConditionalGeneration": "ultravox",
    "CogVLMForCausalLM": "cogvlm",
    "DeepseekOCR2ForCausalLM": "deepseek",
    "DeepseekOCRForCausalLM": "deepseek",
    "DotsOCRForCausalLM": "dotsocr",
    "Gemma3ForConditionalGeneration": "gemma",
    "Gemma3nForConditionalGeneration": "gemma",
    "Gemma4ForConditionalGeneration": "gemma",
    # The GLM-4V / GLM-OCR vision tower is implemented as a subclass of the Qwen3-VL vision converter (Glm4VVisionModel), hence the shared module.
    "Glm4vForConditionalGeneration": "qwen3vl",
    "Glm4vMoeForConditionalGeneration": "qwen3vl",
    "GlmOcrForConditionalGeneration": "qwen3vl",
    "GlmasrModel": "ultravox",
    "GraniteSpeechForConditionalGeneration": "granite",
    "HunYuanVLForConditionalGeneration": "hunyuan",
    "Idefics3ForConditionalGeneration": "smolvlm",
    "InternVisionModel": "internvl",
    "JanusForConditionalGeneration": "januspro",
    "KimiK25ForConditionalGeneration": "kimivl",
    "KimiVLForConditionalGeneration": "kimivl",
    "Lfm2AudioForConditionalGeneration": "lfm2",
    "Lfm2VlForConditionalGeneration": "lfm2",
    "LightOnOCRForConditionalGeneration": "lighton_ocr",
    "Llama4ForConditionalGeneration": "llama4",
    "LlavaForConditionalGeneration": "llava",
    "MERaLiON2ForConditionalGeneration": "ultravox",
    "MiMoV2ForCausalLM": "mimo",
    "MiniCPMV4_6ForConditionalGeneration": "minicpm",
    # Mistral 3.1 ships a Pixtral-style vision encoder, which the llava module's LlavaVisionModel converts (it handles both the HF and Mistral-native layouts).
    "Mistral3ForConditionalGeneration": "llava",
    "NemotronH_Nano_VL_V2": "nemotron",
    "PaddleOCRVisionModel": "ernie",
    "Phi4ForCausalLMV": "phi",
    "Qwen2AudioForConditionalGeneration": "ultravox",
    # Qwen vision-language models: the language side maps to the qwen modules, the vision tower to qwenvl/qwen3vl.
    "Qwen2VLForConditionalGeneration": "qwenvl",
    "Qwen2VLModel": "qwenvl",
    "Qwen2_5OmniModel": "qwenvl",
    "Qwen2_5_VLForConditionalGeneration": "qwenvl",
    "Qwen3ASRForConditionalGeneration": "qwen3vl",
    "Qwen3OmniMoeForConditionalGeneration": "qwen3vl",
    "Qwen3VLForConditionalGeneration": "qwen3vl",
    "Qwen3VLMoeForConditionalGeneration": "qwen3vl",
    "Qwen3_5ForConditionalGeneration": "qwen3vl",
    "Qwen3_5MoeForConditionalGeneration": "qwen3vl",
    "RADIOModel": "nemotron",
    "Sarashina2VisionForCausalLM": "sarashina2",
    "SmolVLMForConditionalGeneration": "smolvlm",
    "StepVLForConditionalGeneration": "step3",
    "UltravoxModel": "ultravox",
    "VoxtralForConditionalGeneration": "ultravox",
    "YoutuVLForConditionalGeneration": "youtuvl",
}


# Distinct module names, precomputed so load_all_models imports each exactly once.
# Sorted for deterministic import order; a set because many architectures share a module.
_TEXT_MODEL_MODULES = sorted(set(TEXT_MODEL_MAP.values()))
_MMPROJ_MODEL_MODULES = sorted(set(MMPROJ_MODEL_MAP.values()))


# Bookkeeping of modules already imported successfully, so repeated calls to
# load_all_models skip the work (and retry only the ones that failed).
_loaded_text_modules: set[str] = set()
_loaded_mmproj_modules: set[str] = set()


def load_all_models() -> None:
    """Import all model modules to trigger @ModelBase.register() decorators."""
    # Cheap length comparison: only walk the list if something is still unloaded.
    if len(_loaded_text_modules) != len(_TEXT_MODEL_MODULES):
        for module_name in _TEXT_MODEL_MODULES:
            if module_name not in _loaded_text_modules:
                try:
                    # Import relative to whatever name this package was loaded under ("conversion" when its
                    # parent directory is on sys.path, "app.utils.conversion" when imported from the app).
                    # Importing the module executes its @ModelBase.register(...) decorators.
                    importlib.import_module(f"{__package__}.{module_name}")
                    _loaded_text_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")

    if len(_loaded_mmproj_modules) != len(_MMPROJ_MODEL_MODULES):
        for module_name in _MMPROJ_MODEL_MODULES:
            if module_name not in _loaded_mmproj_modules:
                try:
                    importlib.import_module(f"{__package__}.{module_name}")
                    _loaded_mmproj_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")


# Lazy path: import only the single module needed for this architecture instead
# of all of them.
def get_model_class(name: str, mmproj: bool = False) -> Type[ModelBase]:
    """Dynamically import and return a model class by its HuggingFace architecture name.

    Args:
        name: HuggingFace architecture string, e.g. "LlamaForCausalLM".
        mmproj: True to look in MMPROJ_MODEL_MAP (vision/audio encoder GGUF) instead of the text map.

    Returns:
        The registered ModelBase subclass for that architecture and model type.

    Raises:
        NotImplementedError: the architecture has no entry in the selected map.
        ImportError: the mapped module (or one of its optional dependencies) cannot be imported.
        KeyError: the module imported but did not register a class under this name (map and decorator out of sync).
    """
    # Pick the table that matches the requested model kind.
    relevant_map = MMPROJ_MODEL_MAP if mmproj else TEXT_MODEL_MAP
    if name not in relevant_map:
        raise NotImplementedError(f"Architecture {name!r} not supported!")
    module_name = relevant_map[name]
    # Importing the module runs its @ModelBase.register decorators, which fill ModelBase._model_classes; the lookup below only works after this import.
    importlib.import_module(f"{__package__}.{module_name}")
    # Registered classes are stored per ModelType, then by architecture name.
    model_type = ModelType.MMPROJ if mmproj else ModelType.TEXT
    return ModelBase._model_classes[model_type][name]


def print_registered_models() -> None:
    """Log every supported architecture name (text, then mmproj) at error level.

    Used for the "unknown architecture" help output, hence logger.error.
    """
    load_all_models()
    logger.error("TEXT models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
    logger.error("MMPROJ models:")
    for name in sorted(MMPROJ_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
