# Backbone
from .backbone import ShuffleNetV2

# Encoder
from .encoder import CNNEncoder

# Decoder
from .decoder import CTCDecoder, AttentionDecoder

# Recognition models
from .recognition import (
    TrOCR,
    MultilingualCTCDecoder,
    LanguageAdapter,
    detect_language,
    detect_languages,
    build_multilingual_charset,
    LANGUAGE_CHARSETS,
)

# Recognizer
from .recognizer import FlashOCR, build_model

# LoRA / QLoRA
from .lora import (
    apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict,
    LORA_VARIANTS, get_variant_description, get_ortho_regularization_loss,
    get_lora_plus_param_groups,
)

__all__ = [
    # Backbone
    "ShuffleNetV2",
    # Encoder
    "CNNEncoder",
    # Decoder
    "CTCDecoder",
    "AttentionDecoder",
    # Recognition models
    "TrOCR",
    "MultilingualCTCDecoder",
    "LanguageAdapter",
    "detect_language",
    "detect_languages",
    "build_multilingual_charset",
    "LANGUAGE_CHARSETS",
    # Recognizer
    "FlashOCR",
    "build_model",
    # LoRA / QLoRA
    "apply_lora",
    "apply_qlora",
    "merge_lora_weights",
    "get_lora_state_dict",
    "LORA_VARIANTS",
    "get_variant_description",
    "get_ortho_regularization_loss",
    "get_lora_plus_param_groups",
]
