from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
from typing import Any, List

class GPT_OpenRouter:
    """GPT class using OpenRouter API - Simple Format"""
    
    def __init__(self, model_name, api=None, temperature=0, seed=0):
        self.model_name = model_name
        self.T = temperature
        self.seed = seed
        
        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api
        )

    def __call__(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:
        if debug:
            return self.client.chat.completions.create(
                messages=prompt, 
                n=n, 
                model=self.model_name, 
                temperature=self.T,
                extra_headers={},
                extra_body={},
                **kwargs)
        else:
            return self.call_wrapper(
                messages=prompt, 
                n=n, 
                model=self.model_name,
                temperature=self.T,
                **kwargs)

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                [wait_fixed(5) for i in range(2)] +
                [wait_fixed(10)]))
    def call_wrapper(self, **kwargs):
        return self.client.chat.completions.create(
            extra_headers={},
            extra_body={},
            **kwargs
        )
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]