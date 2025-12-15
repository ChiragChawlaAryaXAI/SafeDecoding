import sys  
import time 
from typing import Any, List
import multiprocessing as mp
from functools import wraps
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
import json
import re

class GPT_OpenRouter:
    """GPT class using OpenRouter API"""
    
    def __init__(self, model_name, api=None, temperature=0, seed=0):
        self.model_name = model_name
        self.T = temperature
        self.seed = seed
        
        # OpenRouter model mapping
        self.openrouter_models = {
            'gpt-4': 'openai/gpt-4-turbo',
            'gpt-4o': 'openai/gpt-4o',
            'gpt-4-turbo': 'openai/gpt-4-turbo',
            'gpt-3.5': 'openai/gpt-3.5-turbo',
            'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
            'gpt-3.5-turbo-1106': 'openai/gpt-3.5-turbo',
            
            # Llama models
            'llama-4-scout': 'meta-llama/llama-3.3-70b-instruct',
            'llama-3.3-70b': 'meta-llama/llama-3.3-70b-instruct',
            'llama-3.3-70b-versatile': 'meta-llama/llama-3.3-70b-instruct',
            'llama-3.1-70b': 'meta-llama/llama-3.1-70b-instruct',
            'llama-3.1-70b-versatile': 'meta-llama/llama-3.1-70b-instruct',
            'llama-3.1-8b': 'meta-llama/llama-3.1-8b-instruct',
            'llama-3.1-8b-instant': 'meta-llama/llama-3.1-8b-instruct',
            
            # Claude models
            'claude-3.5-sonnet': 'anthropic/claude-3.5-sonnet',
            'claude-3-opus': 'anthropic/claude-3-opus',
            'claude-3-sonnet': 'anthropic/claude-3-sonnet',
            
            # Gemini models (FREE!)
            'gemini-2.0-flash': 'google/gemini-2.0-flash-exp:free',
            'gemini-pro': 'google/gemini-pro',
            'gemini-flash': 'google/gemini-flash-1.5',
            
            # Reasoning models
            'deepseek-r1': 'deepseek/deepseek-r1',
            'o1-preview': 'openai/o1-preview',
            'o1-mini': 'openai/o1-mini',
            
            # Other popular models
            'qwen-2.5-72b': 'qwen/qwen-2.5-72b-instruct',
            'mistral-large': 'mistralai/mistral-large',
        }
        
        # Map model name to OpenRouter format
        if model_name.lower() in self.openrouter_models:
            self.actual_model = self.openrouter_models[model_name.lower()]
        else:
            # If already in correct format or custom model
            self.actual_model = model_name
        
        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api
        )
        
        # Check if model supports reasoning (o1, o3, deepseek-r1)
        self.supports_reasoning = any(x in self.actual_model.lower() 
                                     for x in ['o1', 'o3', 'deepseek-r1', '/r1'])

    def __call__(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:
        if debug:
            return self.create_completion(
                messages=prompt, 
                n=n, 
                model=self.actual_model, 
                temperature=self.T, 
                **kwargs)
        else:
            return self.call_wrapper(
                messages=prompt, 
                n=n, 
                model=self.actual_model,
                temperature=self.T,
                **kwargs)

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                [wait_fixed(5) for i in range(2)] +
                [wait_fixed(10)]))
    def call_wrapper(self, **kwargs):
        return self.create_completion(**kwargs)
    
    def create_completion(self, **kwargs):
        """Create completion with optional reasoning support"""
        # Add reasoning support if model supports it
        if self.supports_reasoning:
            if 'extra_body' not in kwargs:
                kwargs['extra_body'] = {}
            kwargs['extra_body']['reasoning'] = {'enabled': True}
        
        return self.client.chat.completions.create(**kwargs)
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]