from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class CustomGenerationConfig:
    """
    Custom Generation Configuration to replace transformers.GenerationConfig
    """

    max_length: Optional[int] = 2048
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: Union[bool, str] = False
    max_time: Optional[float] = None

    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    penalty_alpha: Optional[float] = None
    use_cache: bool = True

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.0
    encoder_repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    renormalize_logits: bool = False
    constraints: Optional[List[object]] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[List[int]] = None
    remove_invalid_values: bool = False
    exponential_decay_length_penalty: Optional[tuple] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    forced_decoder_ids: Optional[List[List[int]]] = None
    sequence_bias: Optional[dict] = None
    guidance_scale: Optional[float] = None

    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
