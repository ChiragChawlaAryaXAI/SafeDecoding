"""
MT-Bench adapter for SafeDecoding
Integrates SafeDecoding with FastChat's MT-Bench evaluation
"""
import os
import sys
import torch
from pathlib import Path

# Add SafeDecoding to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.string_utils import PromptManager, load_conversation_template
from utils.opt_utils import load_model_and_tokenizer
from utils.safe_decoding import SafeDecoding
from peft import PeftModel


class SafeDecodingModelAdapter:
    """Adapter to use SafeDecoding with MT-Bench"""
    
    def __init__(
        self,
        model_name="llama2",
        defender="SafeDecoding",
        device="cuda:0",
        alpha=3,
        first_m=2,
        top_k=10,
        num_common_tokens=5,
        max_new_tokens=1024
    ):
        self.model_name = model_name
        self.defender = defender
        self.device = device
        self.alpha = alpha
        self.first_m = first_m
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.max_new_tokens = max_new_tokens
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load model with SafeDecoding"""
        # Model mapping
        if self.model_name == "vicuna":
            model_path = "lmsys/vicuna-7b-v1.5"
            template_name = 'vicuna'
        elif self.model_name == "llama2":
            model_path = "meta-llama/Llama-2-7b-chat-hf"
            template_name = 'llama-2'
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")
        
        # Load conversation template
        self.conv_template = load_conversation_template(template_name)
        
        # Load model and tokenizer
        print(f"Loading {model_path}...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            FP16=True,
            low_cpu_mem_usage=True,
            use_cache=False,
            do_sample=False,
            device=self.device
        )
        
        # Load LoRA adapter
        lora_path = f"../lora_modules/{self.model_name}"
        print(f"Loading LoRA from {lora_path}...")
        self.model = PeftModel.from_pretrained(
            self.model, 
            lora_path, 
            adapter_name="expert"
        )
        
        # Initialize SafeDecoding
        adapter_names = ['base', 'expert']
        self.safe_decoder = SafeDecoding(
            self.model,
            self.tokenizer,
            adapter_names,
            alpha=self.alpha,
            first_m=self.first_m,
            top_k=self.top_k,
            num_common_tokens=self.num_common_tokens,
            verbose=False
        )
        
        print(f"✅ Model loaded with {self.defender} defense")
    
    def generate(self, prompt, temperature=0.7, max_tokens=None):
        """
        Generate response using SafeDecoding
        
        Args:
            prompt: User message
            temperature: Not used (SafeDecoding doesn't support temperature)
            max_tokens: Max new tokens to generate
        
        Returns:
            Generated text
        """
        if max_tokens is None:
            max_tokens = self.max_new_tokens
        
        # Create prompt manager
        input_manager = PromptManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=prompt,
            whitebox_attacker=False
        )
        inputs = input_manager.get_inputs()
        
        # Generate config
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = max_tokens
        gen_config.do_sample = False
        
        # Generate with SafeDecoding
        if self.defender == 'SafeDecoding':
            output, _ = self.safe_decoder.safedecoding_lora(inputs, gen_config=gen_config)
        else:
            # Baseline generation
            output, _ = self.safe_decoder.generate_baseline(inputs, gen_config=gen_config)
        
        return output
    
    def generate_batch(self, prompts, **kwargs):
        """Generate responses for multiple prompts"""
        return [self.generate(p, **kwargs) for p in prompts]