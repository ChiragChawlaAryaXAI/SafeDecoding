from typing import Any
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
import google.generativeai as genai
import boto3
import json

class GPT:
    def __init__(self, model_name, api=None, temperature=0, seed=0):
        self.model_name = model_name
        self.T = temperature
        self.seed = seed
        
        # Map to Groq models (only for backward compatibility)
        if 'gpt-4' in model_name:
            self.actual_model = 'llama-3.3-70b-versatile'
        elif 'gpt-3.5' in model_name:
            self.actual_model = 'llama-3.1-70b-versatile'
        elif 'llama' in model_name.lower():
            self.actual_model = model_name
        elif 'qwen' in model_name.lower():
            self.actual_model = model_name
        else:
            self.actual_model = model_name
        
        # Initialize Groq client
        from groq import Groq
        self.client = Groq(api_key=api)
        

    def __call__(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:
        prompt = [{'role':'user', 'content':prompt}]
        if debug:
            return self.client.chat.completions.create(
                messages=prompt, 
                n=n, 
                model=self.actual_model, 
                temperature=self.T, 
                **kwargs
            )
        else:
            return self.call_wrapper(
                messages=prompt, 
                n=n, 
                model=self.actual_model,
                temperature=self.T,
                **kwargs
            )
    

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                [wait_fixed(5) for i in range(2)] +
                [wait_fixed(10)]))
    def call_wrapper(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]
    

def load_model(model_name, api_idx, **kwargs):
    if "gpt" in model_name and "gpt2" not in model_name:
        return GPT(model_name, **kwargs)
    else:
        raise ValueError(f"model_name invalid")