import os
from os.path import join
from typing import Any, List, Mapping, Dict, Optional

import torch
from langchain.llms.base import LLM

from models.backend import BackendFactory, BackendConfig, GenerationConfig
from agents.agent import STOP_WORDS


class CustomLLM(LLM):
    """
    Unified LLM interface supporting multiple backends (OpenAI, HuggingFace, ExLlama, vLLM).
    Uses pluggable backend architecture for extensibility.
    """
    
    model_name: str
    max_context_length: int = 4096
    probabilities: torch.Tensor = None
    openai_api_key: Optional[str] = None
    tags: Dict[str, str] = None
    seed: int = 2023
    self_consistency: bool = False
    
    # Backend configuration
    exllama: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    base_models_path: Optional[str] = None
    backend: Optional[Any] = None
    backend_type: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _llm_name(self) -> str:
        return self.model_name
    
    @property
    def tokenizer(self):
        """Backward compatibility: expose tokenizer from backend."""
        if self.backend and hasattr(self.backend, 'tokenizer'):
            return self.backend.tokenizer
        return None
    
    @property
    def model(self):
        """Backward compatibility: expose model from backend."""
        if self.backend and hasattr(self.backend, 'model'):
            return self.backend.model
        return None
    
    @property
    def generator(self):
        """Backward compatibility: expose generator from backend (ExLlama)."""
        if self.backend and hasattr(self.backend, 'generator'):
            return self.backend.generator
        return None

    def load_model(self, base_models: str) -> None:
        """Load model using appropriate backend."""
        torch.cuda.empty_cache()

        # Determine backend type if not already specified
        if not self.backend_type:
            if self.openai_api_key:
                self.backend_type = "openai"
            else:
                self.backend_type = BackendFactory.infer_backend_type(
                    self.model_name, 
                    exllama=self.exllama
                )
        
        # Build backend config
        backend_config = BackendConfig(
            backend_type=self.backend_type,
            model_name=self.model_name,
            openai_api_key=self.openai_api_key,
            base_models_path=base_models,
            exllama_enabled=self.exllama,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            max_context_length=self.max_context_length,
            tags=self.tags,
        )
        
        # Create and load backend
        self.backend = BackendFactory.create(backend_config)
        
        # Handle special case where backend needs explicit openai key
        if self.backend_type == "openai" and self.openai_api_key:
            import openai
            openai.api_key = self.openai_api_key
        
        self.backend.load()
    
    def _call(
        self,
        prompt: str,
        stop: List[str],
        do_sample=True,
        temperature=0.01,
        top_k=1,
        top_p=0.95,
        num_beams=1,
        repetition_penalty=1.2,
        length_penalty=1.0,
        **kwargs,
    ) -> str:
        """Generate text using configured backend."""
        if not self.backend:
            raise RuntimeError("Backend not initialized. Call load_model() first.")
        
        # Build unified generation config
        gen_config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=self.max_context_length,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_beams=num_beams,
            seed=self.seed if not self.self_consistency else None,
        )
        
        # Generate text
        output = self.backend.generate(prompt, stop, gen_config)
        
        # Copy probabilities if available
        if hasattr(self.backend, 'probabilities'):
            self.probabilities = self.backend.probabilities
        
        # Remove stop words from output (as before)
        for stop_word in STOP_WORDS + stop:
            output = output.replace(stop_word, "")
        
        return output.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "backend_type": self.backend_type or "unknown",
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
        }
