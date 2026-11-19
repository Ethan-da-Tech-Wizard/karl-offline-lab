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
from __future__ import annotations

# Re-export the shared base classes/helpers so callers can import everything
# from `conversion` without knowing about conversion.base.
from .base import (
    ModelBase, TextModel, MmprojModel, ModelType, SentencePieceTokenTypes,
    logger, _mistral_common_installed, _mistral_import_error_msg,
    get_model_architecture, LazyTorchTensor,
)
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
    "AfmoeForCausalLM": "afmoe",
    "ApertusForCausalLM": "llama",
    "ArceeForCausalLM": "llama",
    "ArcticForCausalLM": "arctic",
    "AudioFlamingo3ForConditionalGeneration": "qwen",
    "BaiChuanForCausalLM": "baichuan",
    "BaichuanForCausalLM": "baichuan",
    "BailingMoeForCausalLM": "bailingmoe",
    "BailingMoeV2ForCausalLM": "bailingmoe",
    "BambaForCausalLM": "granite",
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
    "FalconMambaForCausalLM": "mamba",
    "GPT2LMHeadModel": "gpt2",
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
    "Step3p5ForCausalLM": "step3",
    "StepVLForConditionalGeneration": "step3",
    "T5EncoderModel": "t5",
    "T5ForConditionalGeneration": "t5",
    "T5WithLMHeadModel": "t5",
    "TalkieForCausalLM": "talkie",
    "UMT5ForConditionalGeneration": "t5",
    "UMT5Model": "t5",
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
    "Mistral3ForConditionalGeneration": "llava",
    "NemotronH_Nano_VL_V2": "nemotron",
    "PaddleOCRVisionModel": "ernie",
    "Phi4ForCausalLMV": "phi",
    "Qwen2AudioForConditionalGeneration": "ultravox",
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
_TEXT_MODEL_MODULES = sorted(set(TEXT_MODEL_MAP.values()))
_MMPROJ_MODEL_MODULES = sorted(set(MMPROJ_MODEL_MAP.values()))


# Bookkeeping of modules already imported successfully, so repeated calls to
# load_all_models skip the work (and retry only the ones that failed).
_loaded_text_modules: set[str] = set()
_loaded_mmproj_modules: set[str] = set()


def load_all_models() -> None:
    """Import all model modules to trigger @ModelBase.register() decorators."""
    # Step 1: Validate input parameters, operational preconditions, and state
    # Cheap length comparison: only walk the list if something is still unloaded.
    if len(_loaded_text_modules) != len(_TEXT_MODEL_MODULES):
        for module_name in _TEXT_MODEL_MODULES:
            # Iterate over items, evaluate loop invariant, and update progress
            if module_name not in _loaded_text_modules:
                try:
                    # Execute guarded operations and establish recovery boundaries
                    # Absolute import: assumes the directory containing this package is on sys.path.
                    # Importing the module executes its @ModelBase.register(...) decorators.
                    __import__(f"conversion.{module_name}")
                    _loaded_text_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")

    if len(_loaded_mmproj_modules) != len(_MMPROJ_MODEL_MODULES):
        for module_name in _MMPROJ_MODEL_MODULES:
            # Iterate over items, evaluate loop invariant, and update progress
            if module_name not in _loaded_mmproj_modules:
                try:
                    # Execute guarded operations and establish recovery boundaries
                    __import__(f"conversion.{module_name}")
                    _loaded_mmproj_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")


# Lazy path: import only the single module needed for this architecture instead
# of all of them.
def get_model_class(name: str, mmproj: bool = False) -> Type[ModelBase]:
    """Dynamically import and return a model class by its HuggingFace architecture name."""
    # Step 1: Validate input parameters, operational preconditions, and state
    # Pick the table that matches the requested model kind.
    relevant_map = MMPROJ_MODEL_MAP if mmproj else TEXT_MODEL_MAP
    if name not in relevant_map:
        raise NotImplementedError(f"Architecture {name!r} not supported!")
    module_name = relevant_map[name]
    __import__(f"conversion.{module_name}")
    # Registered classes are stored per ModelType, then by architecture name.
    model_type = ModelType.MMPROJ if mmproj else ModelType.TEXT
    return ModelBase._model_classes[model_type][name]


def print_registered_models() -> None:
    """Log every supported architecture name (text, then mmproj) at error level.

    Used for the "unknown architecture" help output, hence logger.error.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    load_all_models()
    logger.error("TEXT models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        # Iterate over items, evaluate loop invariant, and update progress
        logger.error(f"  - {name}")
    logger.error("MMPROJ models:")
    for name in sorted(MMPROJ_MODEL_MAP.keys()):
        # Iterate over items, evaluate loop invariant, and update progress
        logger.error(f"  - {name}")
