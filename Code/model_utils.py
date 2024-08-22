import sys
import time
import json
import requests

url_dict = {
    'MiniCPM-dpo': 'http://localhost:8001/generate',
    'MiniCPM-sft': 'http://localhost:8002/generate',
    'Qwen1.5-4B': 'http://localhost:8003/generate',
    'LLAMA3-8B-Meta': 'http://localhost:8004/generate',
    'LLAMA3-8B-HFL': 'http://localhost:8005/generate',
    'LLAMA3.1-8B-Meta': 'http://localhost:8006/v1/chat/completions',
    'InternLM2-20B-4bit': 'http://localhost:23333/v1/chat/completions',
    'InternLM2-20B': 'http://localhost:8000/v1/chat/completions',
    'Qwen2-0.5B-Instruct': 'http://localhost:8010/v1/chat/completions',
    'Qwen2-1.5B-Instruct': 'http://localhost:8011/v1/chat/completions',
    'Qwen2-7B-Instruct': 'http://localhost:8012/v1/chat/completions',
    'Qwen2-72B-Instruct': 'http://xx.xx.xx.xx:12620/v1/chat/completions',
    'ChatGLM3-6B': 'http://localhost:8015/v1/chat/completions',
    'GLM-4-9B-Chat': 'http://localhost:8016/v1/chat/completions',
    'Baichuan2-13B-Chat': 'http://localhost:8017/v1/chat/completions',

    'LLAMA3-70B-Meta': 'http://localhost:8100/v1/chat/completions',
    'airoboros-l2-70B': 'http://localhost:8100/v1/chat/completions',
    'Qwen1.5-110B': 'http://localhost:8100/v1/chat/completions',
    'Qwen2-72B': 'http://localhost:8100/v1/chat/completions',
    'Yi-34B': 'http://localhost:8100/v1/chat/completions',
}


class LLM:
    def __init__(self, url, template, parser):
        self.url = url
        self.template = template
        self.parser = parser

    def __call__(self, prompt):
        json_data = self.create_json(prompt)

        start = time.time()
        response = requests.request('POST', self.url, json=json_data)
        end = time.time()

        cost = end - start
        output = self.parser(response).strip()

        return cost, output

    def create_json(self, prompt):
        return {
            "prompt": self.template.format(message=prompt),
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
            "stop": ["</result>"], "include_stop_str_in_output": True
        }


class MiniCPM(LLM):
    def __init__(self, version='dpo'):
        url = url_dict['MiniCPM-' + version]
        template = '<用户>{message}<AI>'
        parser = lambda x: json.loads(x.text)['text'][0].split('<AI>')[-1]
        super(MiniCPM, self).__init__(url, template, parser)


class Qwen(LLM):
    def __init__(self, version=None):
        url = url_dict['Qwen1.5-4B']
        template = '<|im_start|>system\nYou are a professional content discernment expert<|im_end|>\n' \
                   '<|im_start|>user\n{message}<|im_end|>\n' \
                   '<|im_start|>assistant\n'
        parser = lambda x: json.loads(x.text)['text'][0].split('<|im_start|>assistant\n')[-1]
        super(Qwen, self).__init__(url, template, parser)


class LLAMA3(LLM):
    def __init__(self, version='Meta'):
        url = url_dict['LLAMA3-8B-' + version]
        template = '<|begin_of_text|>' \
                   '<|start_header_id|>system<|end_header_id|>\n\n' \
                   'You are a professional content discernment expert' \
                   '<|eot_id|>' \
                   '<|start_header_id|>user<|end_header_id|>\n\n' \
                   '{message}' \
                   '<|eot_id|>' \
                   '<|start_header_id|>assistant<|end_header_id|>'
        parser = lambda x: json.loads(x.text)['text'][0].split('<|start_header_id|>assistant<|end_header_id|>')[-1]
        super(LLAMA3, self).__init__(url, template, parser)


class LLAMA31(LLM):
    def __init__(self, version=None):
        url = url_dict['LLAMA3.1-8B-Meta']
        template = None
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        super(LLAMA31, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            "model": 'Meta-Llama-3.1-8B-Instruct',
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
            "stop": ["</result>", "\n\n"], "include_stop_str_in_output": True
        }


class InternLM2(LLM):
    def __init__(self, version=None):
        url = url_dict['InternLM2-20B']
        template = None
        # template = '<|im_start|>system\nYou are a professional content discernment expert<|im_end|>\n' \
        #            '<|im_start|>user\n{message}<|im_end|>\n' \
        #            '<|im_start|>assistant\n'
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        # parser = lambda x: json.loads(x.text)['text'][0].split('<|im_start|>assistant\n')[-1]
        super(InternLM2, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            # "model": "internlm2",
            "model": "internlm2-chat-20b",
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
            "stop": ["</result>", "\n\n"], "include_stop_str_in_output": True
        }


class Qwen2(LLM):
    def __init__(self, version=None):
        self.model_name = 'Qwen2-%s-Instruct' % version
        url = url_dict[self.model_name]
        if version == '72B':
            self.model_name = 'Qwen2-72B'
        template = None
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        super(Qwen2, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            "model": self.model_name,
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
            "stop": ["</result>", "\n\n"], "include_stop_str_in_output": True
        }


class ChatGLM3(LLM):
    def __init__(self, version=None):
        self.model_name = 'ChatGLM3-%s' % version
        url = url_dict[self.model_name]
        template = None
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        super(ChatGLM3, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            "model": self.model_name.lower(),
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 128,
            "top_k": 10, "top_p": 0.95,
            "stop": ["</result>", "\n\n"], "include_stop_str_in_output": True
        }


class GLM4(LLM):
    def __init__(self, version=None):
        self.model_name = 'GLM-4-%s-Chat' % version
        url = url_dict[self.model_name]
        template = None
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        super(GLM4, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            "model": self.model_name.lower(),
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
        }


class Baichuan2(LLM):
    def __init__(self, version=None):
        self.model_name = 'Baichuan2-%s-Chat' % version
        url = url_dict[self.model_name]
        template = None
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        super(Baichuan2, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            "model": self.model_name,
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
            "stop_token_ids": [2]
        }


class GGUFModel(LLM):
    def __init__(self, version=None):
        url = url_dict[version]
        template = None
        parser = lambda x: json.loads(x.text)['choices'][0]['message']['content']
        super(GGUFModel, self).__init__(url, template, parser)

    def create_json(self, prompt):
        return {
            "model": '',
            "messages": [
                {'role': 'system', 'content': 'You are a professional content discernment expert'},
                {'role': 'user', 'content': prompt}
            ],
            "temperature": 0.1, "max_tokens": 64,
            "top_k": 10, "top_p": 0.95,
        }


if __name__ == '__main__':
    # getattr(sys.modules['__main__'], 'MiniCPM')()('hello')
    # getattr(sys.modules['__main__'], 'Qwen')()('hello')
    # print(getattr(sys.modules['__main__'], 'LLAMA3')()('1+1=?'))
    # print(getattr(sys.modules['__main__'], 'LLAMA31')()('1+1=?'))
    # print(getattr(sys.modules['__main__'], 'InternLM2')()('hello'))
    # print(getattr(sys.modules['__main__'], 'Qwen2')('7B')(p))
    # print(getattr(sys.modules['__main__'], 'Qwen2')('72B')('hello'))
    # print(getattr(sys.modules['__main__'], 'ChatGLM3')('6B')('hello'))
    # print(getattr(sys.modules['__main__'], 'GLM4')('9B')('hello'))
    # print(getattr(sys.modules['__main__'], 'Baichuan2')('13B')('hello'))
    print(getattr(sys.modules['__main__'], 'GGUFModel')('LLAMA3-70B-Meta')('hello'))
