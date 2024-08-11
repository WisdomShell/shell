<p align="center">
    <h1 align="center"> Shell </h1>
<p>



<p align="center">
  🤗 <a href="https://huggingface.co/WisdomShell" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/WisdomShell" target="_blank">ModelScope</a> • ⭕️ <a href="https://www.wisemodel.cn/organization/WisdomShell" target="_blank">WiseModel</a> • 🌐 <a href="http://se.pku.edu.cn/kcl/" target="_blank">PKU-KCL</a> 
</p>
<div align="center">

[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/WisdomShell/shell/blob/main/License.pdf)

<h4 align="center">
    <p><a href="https://github.com/WisdomShell/shell/blob/main/README.md"><b>中文</b></a>|<a href="https://github.com/WisdomShell/shell/blob/main/README_EN.md">English</a></p>
</h4>
</div>


## Introduction

Shell是[北京大学知识计算实验室](http://se.pku.edu.cn/kcl/)在代码大模型[CodeShell](https://github.com/WisdomShell/codeshell)基础上训练的预训练通用大模型。Shell在保留Codeshell优异的代码能力的同时，具有以下特性：

- **更全面的通用能力**：Shell在Codeshell的基础上继续预训练了1.5 T token的中英文语料，通用能力大幅提升。在语言、知识、推理等评测中，Shell均取得了优异的性能。
- **依旧强大的代码能力**：Shell在继续预训练的过程中，保留了20%高质量代码数据，使得Shell在获得通用能力的同时，依旧保留了CodeShell强大的代码能力。
- **更强大的语义理解能力**：Shell在继承Codeshell优异代码能力的同时，形成了强大的语义理解能力。相比LLaMA2-7B，Shell在RACE-Middle （+102%）、RACE-High（+98%）、OpenbookQA（+42%）等多个语义理解数据集取得更好的性能，达到同等规模开源大模型的领先水平。

本次我们同时发布了Shell-7B的base版本和chat版本，大家可以根据自身需求选择对应的模型。

- **Shell-7B-Base**：具有强大语义理解能力的通用大模型，大家可以基于该模型微调自己的大模型。
- **Shell-7B-Chat**：在Shell-7B-Base微调得到的对话预训练模型，直接下载使用即可获得流畅的对话体验。


## Performance

我们共选取了16个经典数据集对Shell进行了全面评测，评测脚本详见[模型评测](https://github.com/WisdomShell/shell/edit/main/evaluation/README.md)。具体评测结果如下。
| Dataset      | Baichuan2-7B-Base | LLaMA-2-7B | Shell-7B |
| ------------ | ----------------- | ---------- | -------- |
| C-Eval       | 56.3              | 32.5       | 50.13    |
| AGIEval      | 34.6              | 21.8       | 30.69    |
| MMLU         | 54.7              | 46.8       | 49.49    |
| CMMLU        | 57                | 31.8       | 50.4     |
| GAOKAO-Bench | 34.8              | 18.9       | 33       |
| WiC          | 50                | 50         | 50.47    |
| CHID         | 82.7              | 46.5       | 83.17    |
| AFQMC        | 58.4              | 69         | 69       |
| WSC          | 66.3              | 66.3       | 63.46    |
| RACE(Middle) | 50.9              | 40.2       | 82.66    |
| RACE(High)   | 52.5              | 37.5       | 74.24    |
| OpenbookQA   | 32.8              | 57         | 79       |
| GSM8K        | 24.6              | 16.7       | 20.7     |
| HumanEval    | 17.7              | 12.8       | 23.96    |
| MBPP         | 24                | 14.8       | 31.4     |
| BBH          | 41.8              | 38.2       | 38.16    |

## Requirements

- python 3.8 and above
- pytorch 2.0 and above are recommended
- transformers 4.32 and above
- CUDA 11.8 and above are recommended (this is for GPU users, flash-attention users, etc.)

## Quickstart

Shell系列模型已经上传至 <a href="https://huggingface.co/WisdomShell/CodeShell" target="_blank">Hugging Face</a>，开发者可以通过Transformers快速调用Shell-7B和Shell-Chat-7B。

在开始之前，请确保已经正确设置了环境，并安装了必要的代码包，以及满足上一小节的环境要求。你可以通过下列代码快速安装相关依赖。

```
pip install -r requirements.txt
```

接下来你可以通过Transformers使用Shell。

### 加载Shell-7B-Base

您可以通过Transformers加载Shell-7B-Base模型，Shell-7B-Base具备生成流畅自然语言的能力，您可以通过`generate`方法让模型生成相关的文字。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("WisdomShell/Shell-7B-Base")
model = AutoModelForCausalLM.from_pretrained("WisdomShell/Shell-7B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
inputs = tokenizer('你好', return_tensors='pt').to(device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### 加载Shell-7B-Chat

类似的，您可以通过Transformers加载Shell-7B-Chat模型，并通过`chat`方法与其进行对话。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("WisdomShell/Shell-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("WisdomShell/Shell-7B-Chat", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
inputs = tokenizer('你好', return_tensors='pt').to(device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### Shell in c/c++

由于大部分个人电脑没有GPU，Shell提供了C/C++版本的推理支持，开发者可以根据本地环境进行编译与使用，详见[CodeShell C/C++本地化版](https://github.com/WisdomShell/llama_cpp_for_codeshell)。


## Finetune

我们同样提供了模型微调相关代码，大家可以按照示例数据的格式准备自己的数据，进行快速微调，具体请参考[模型微调](https://github.com/WisdomShell/shell/edit/main/finetune/README.md)。

其中，多轮对话微调数据格式如下。

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "您好，我是Shell，请问有什么可以帮助您的吗？"
      }
    ]
  }
]
```

## Demo

我们提供了Web-UI、命令行、API三种形式的Demo。

### Web UI

开发者通过下列命令启动Web服务，服务启动后，可以通过`https://127.0.0.1:8000`进行访问。

```
python demos/web_demo.py
```

### CLI Demo

我们也提供了命令行交互的Demo版本，开发者可以通过下列命令运行。

```
python demos/cli_demo.py
```

### API

CodeShell也提供了基于OpenAI API的部署方法。

```
python demos/openai_api.py
```

启动后即可通过HTTP请求与CodeShell交互。

```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "CodeShell-7B-Chat",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'
```

## License

社区使用Shell模型需要遵循[《Shell模型许可协议》](https://github.com/WisdomShell/shell/blob/main/License.pdf)及[Apache 2.0许可协议](https://www.apache.org/licenses/LICENSE-2.0)。Shell模型允许用于商业用途，但如果您计划将Shell模型或其派生产品用于商业用途，需要您确认主体符合以下条件：

1. 关联方的服务或产品的每日平均活跃用户数（DAU）不能超过100万。
2. 关联方不得是软件服务提供商或云服务提供商。
3. 关联方不存在将获得授予的商业许可，在未经许可的前提下将其再授权给其他第三方的可能性。

在满足上述条件的前提下，您需要通过向shell.opensource@openbankai.com发送电子邮件，提交《Shell模型许可协议》要求的申请材料。经审核通过后，将授予您一个全球的、非排他的、不可转让的、不可再授权的商业版权许可。

