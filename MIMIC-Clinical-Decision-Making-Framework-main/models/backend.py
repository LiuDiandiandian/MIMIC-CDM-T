"""
LLM Backend abstraction layer supporting OpenAI-style API.
Each backend handles model loading and inference for specific LLM types.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
import os
from os.path import join


@dataclass
class GenerationConfig:
    """Unified generation configuration across all backends."""
    temperature: float = 0.01
    top_k: int = 1
    top_p: float = 0.95
    max_new_tokens: int = 512
    do_sample: bool = True
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    num_beams: int = 1
    seed: Optional[int] = None


@dataclass
class BackendConfig:
    """Configuration for model backend selection and parameters."""
    backend_type: str  # "openai", "transformers", "exllama", "vllm", "human"
    model_name: str
    openai_api_key: Optional[str] = None
    base_models_path: Optional[str] = None
    exllama_enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_context_length: int = 4096
    tags: Optional[Dict[str, str]] = None  # For prompt formatting
    
    # Model-specific params
    quantization_type: Optional[str] = None  # "gptq", "awq", None
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.probabilities = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        stop_sequences: List[str],
        gen_config: GenerationConfig,
    ) -> str:
        """Generate text from prompt. Returns raw output without stop word removal."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'to'):
                # For PyTorch models
                del self.model
                torch.cuda.empty_cache()


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI API (GPT-3.5, GPT-4, etc.)."""
    
    def load(self) -> None:
        import openai
        import tiktoken
        
        if not self.config.openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI backend")
        
        openai.api_key = self.config.openai_api_key
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)
    
    def generate(
        self,
        prompt: str,
        stop_sequences: List[str],
        gen_config: GenerationConfig,
    ) -> str:
        import openai
        from utils.nlp import extract_sections
        
        messages = extract_sections(prompt, self.config.tags or {})
        
        response = self._completion_with_retry(
            model=self.config.model_name,
            messages=messages,
            stop=stop_sequences[:4] if stop_sequences else None,  # OpenAI API limits to 4 stop sequences
            temperature=gen_config.temperature,
            seed=gen_config.seed,
        )
        return response["choices"][0]["message"]["content"]
    
    @staticmethod
    def _completion_with_retry(**kwargs):
        """Helper method with exponential backoff retry."""
        from tenacity import retry, stop_after_attempt, wait_random_exponential
        import openai
        
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
        def _call(**kw):
            return openai.ChatCompletion.create(**kw)
        
        return _call(**kwargs)


class TransformersBackend(LLMBackend):
    """Backend for HuggingFace Transformers models."""
    
    def load(self) -> None:
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            BitsAndBytesConfig,
        )
        
        cache_dir = self.config.cache_dir or self.config.base_models_path
        
        # Load config and tokenizer
        config = AutoConfig.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            use_fast=False,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Handle pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.cls_token is not None:
                self.tokenizer.pad_token = self.tokenizer.cls_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
        
        # Determine model class
        is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
        model_cls = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
        
        # Setup quantization
        quantization_config = None
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.config.trust_remote_code,
        }
        
        if self.config.load_in_4bit or self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16
        
        # Load model
        self.model = model_cls.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        
        # Optimize model
        self._optimize_model()
        self.tokenizer.truncation_side = "left"
    
    def _optimize_model(self) -> None:
        """Apply optimizations to loaded model."""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        if torch.__version__ >= "2" and not self.config.exllama_enabled:
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass  # Compilation may fail on some setups
    
    def generate(
        self,
        prompt: str,
        stop_sequences: List[str],
        gen_config: GenerationConfig,
    ) -> str:
        from transformers import (
            GenerationConfig as HFGenerationConfig,
            StoppingCriteriaList,
        )
        from models.utils import create_stop_criteria
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_context_length,
            truncation=True,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        
        hf_config = HFGenerationConfig(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            num_beams=gen_config.num_beams,
            do_sample=gen_config.do_sample,
            repetition_penalty=gen_config.repetition_penalty,
            length_penalty=gen_config.length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=gen_config.max_new_tokens,
        )
        
        stop_criteria = create_stop_criteria(
            stop_sequences, self.tokenizer, self.model.device
        )
        
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=hf_config,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        s = generation_output.sequences
        s_no_input = s[:, input_ids.shape[1]:]
        output = self.tokenizer.batch_decode(s_no_input, skip_special_tokens=True)[0]
        
        return output


class ExLlamaBackend(LLMBackend):
    """Backend for ExLlamaV2 quantized models."""
    
    def load(self) -> None:
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from models.exllamav2_generator_base_custom import ExLlamaV2BaseGenerator
        
        torch.cuda._lazy_init()
        
        # Build config path
        model_path = self.config.base_models_path
        if model_path and not os.path.isabs(self.config.model_name):
            model_path = join(model_path, self.config.model_name)
        else:
            model_path = self.config.model_name
        
        # Load ExLlama model
        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.max_seq_len = self.config.max_context_length
        config.scale_pos_emb = 1.0
        config.scale_alpha_value = 1.0
        config.no_flash_attn = not self.config.use_flash_attention
        
        self.model = ExLlamaV2(config)
        self.model.load()
        
        self.tokenizer = ExLlamaV2Tokenizer(config)
        cache = ExLlamaV2Cache(self.model)
        
        self.generator = ExLlamaV2BaseGenerator(self.model, cache, self.tokenizer)
        self.generator.warmup()
    
    def generate(
        self,
        prompt: str,
        stop_sequences: List[str],
        gen_config: GenerationConfig,
    ) -> str:
        from exllamav2.generator import ExLlamaV2Sampler
        from models.utils import create_stop_criteria_exllama
        
        with torch.inference_mode():
            ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
            tokens_prompt = ids.shape[-1]
            
            settings = ExLlamaV2Sampler.Settings()
            if gen_config.temperature > 0.01:
                settings = settings.clone()
                settings.temperature = gen_config.temperature
                seed = None
            else:
                settings = settings.greedy_clone()
                seed = gen_config.seed
            
            stop_criteria = create_stop_criteria_exllama(
                stop_sequences,
                self.tokenizer.eos_token_id,
                self.tokenizer,
            )
            
            output_tokens, self.probabilities = self.generator.generate_simple(
                prompt,
                gen_settings=settings,
                num_tokens=self.config.max_context_length - tokens_prompt,
                seed=seed,
                token_healing=True,
                encode_special_tokens=True,
                decode_special_tokens=False,
                stop_criteria=stop_criteria,
            )
            
            output_tokens = self._remove_input_tokens(output_tokens, ids)
            output = self.tokenizer.decode(output_tokens, decode_special_tokens=False)[0]
        
        return output
    
    @staticmethod
    def _remove_input_tokens(output_tokens, ids):
        """Remove input tokens from output."""
        min_size = min(output_tokens.size(1), ids.size(1))
        truncated_output_tokens = output_tokens[:, :min_size]
        truncated_ids = ids[:, :min_size]
        common_prefix = (
            (truncated_output_tokens == truncated_ids).cumprod(dim=0).sum().item()
        )
        return output_tokens[:, common_prefix:]


class VLLMBackend(LLMBackend):
    """Backend for vLLM engine (high-throughput inference)."""
    
    def load(self) -> None:
        from vllm import LLM
        from transformers import AutoTokenizer
        
        model_path = self.config.model_name
        if self.config.base_models_path:
            local_path = join(self.config.base_models_path, self.config.model_name)
            if os.path.exists(local_path):
                model_path = local_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir or self.config.base_models_path,
            use_fast=False,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
        )
    
    def generate(
        self,
        prompt: str,
        stop_sequences: List[str],
        gen_config: GenerationConfig,
    ) -> str:
        from vllm import SamplingParams
        
        params = {
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "max_tokens": gen_config.max_new_tokens,
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        
        sampling_params = SamplingParams(**params)
        result = self.model.generate([prompt], sampling_params=sampling_params)
        
        return result[0].outputs[0].text


class HumanBackend(LLMBackend):
    """Backend for interactive human input."""
    
    def load(self) -> None:
        pass  # No loading needed
    
    def generate(
        self,
        prompt: str,
        stop_sequences: List[str],
        gen_config: GenerationConfig,
    ) -> str:
        return input(prompt)


class BackendFactory:
    """Factory for creating appropriate backend instances."""
    
    _backends = {
        "openai": OpenAIBackend,
        "transformers": TransformersBackend,
        "exllama": ExLlamaBackend,
        "vllm": VLLMBackend,
        "human": HumanBackend,
    }
    
    @classmethod
    def register(cls, backend_type: str, backend_class: type) -> None:
        """Register a custom backend type."""
        cls._backends[backend_type] = backend_class
    
    @classmethod
    def create(cls, config: BackendConfig) -> LLMBackend:
        """Create a backend instance based on config."""
        backend_type = config.backend_type.lower()
        
        if backend_type not in cls._backends:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available: {', '.join(cls._backends.keys())}"
            )
        
        backend_class = cls._backends[backend_type]
        return backend_class(config)
    
    @classmethod
    def infer_backend_type(cls, model_name: str, exllama: bool = False) -> str:
        """Infer backend type from model name."""
        if model_name == "Human":
            return "human"
        elif model_name.startswith("vllm://"):
            return "vllm"
        elif "GPTQ" in model_name and exllama:
            return "exllama"
        elif "GPTQ" in model_name:
            return "transformers"
        else:
            return "transformers"
