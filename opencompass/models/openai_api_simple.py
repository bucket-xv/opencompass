import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import warnings

import openai
from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from opencompass.utils.prompt import PromptList
from opencompass.registry import MODELS
from .base_api import BaseAPIModel


PromptType = Union[PromptList, str]

@MODELS.register_module()
class SimpleOpenAI(BaseAPIModel):
    """Model wrapper around 
    """

    def __init__(self,
                 path: str,
                 api_key: str,
                 base_url: str,
                 api_type: str = 'openai',
                 max_concurrency: int = 5,
                 query_per_second: int = 2,
                 max_seq_len: int = 8192,
                 meta_template: Optional[Dict] = None,
                 retry: int = 3,
                 timeout: int = 60,
                 generation_kwargs: Dict = {
                     'temperature': 0.7,
                     'top_p': 0.9,
                     'top_k': 0,
                     'verbose': False,
                 }):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        
        print(f"PATH:{self.path}")
        print(f"API Key:{api_key}")
        print(f"Base URL:{base_url}")
        print(f"Timeout:{timeout}")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.max_tokens = max_seq_len
        self.type = api_type
        self.temperature = generation_kwargs['temperature']
        self.verbose = generation_kwargs['verbose']

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = dict()
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                else:
                    raise ValueError(f"Unknown role {item['role']}")
                msg ['content'] = item['prompt']

                messages.append(msg)
        if self.verbose:
            print(f"Messages:{messages}")

        if self.type == 'openai':
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=float(self.timeout))
        elif self.type == 'azure':
            client = ChatCompletionsClient(endpoint=self.base_url, credential=AzureKeyCredential(self.api_key))
        trial = 0
        while trial < self.retry:
            # self.acquire()
            try:
                if self.type == 'azure':
                    response = client.complete(
                        model=self.path,
                        messages=messages,
                        temperature=self.temperature,
                        stream=True,
                        max_tokens=self.max_tokens
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.path,
                        messages=messages,
                        stream=True,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                reasoning_content = ""  # 定义完整思考过程
                answer_content = ""     # 定义完整回复
                is_answering = False   # 判断是否结束思考过程并开始回复
                for chunk in response:
                    # 如果chunk.choices为空，则打印usage
                    if not chunk.choices:
                        if self.verbose:
                            print("\nUsage:")
                            print(chunk.usage)
                    else:
                        delta = chunk.choices[0].delta
                        # 打印思考过程
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                            if self.verbose:
                                print(delta.reasoning_content, end='', flush=True)
                            if delta.reasoning_content:
                                reasoning_content += delta.reasoning_content
                        else:
                            # 开始回复
                            if delta.content != "" and is_answering == False:
                                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                                is_answering = True
                            # 打印回复过程
                            if self.verbose:
                                print(delta.content, end='', flush=True)
                            if delta.content:
                                answer_content += delta.content
                return answer_content
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec"
                )
                print(e)
                time.sleep(exception_backoff)
                trial += 1
                
            # self.release()


        warnings.warn("Failed to generate response.")
        return "" 
