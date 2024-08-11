

本文档将向您介绍Shell模的评测步骤，评测基于[OpenCompass](https://github.com/open-compass/opencompass)



## 准备步骤

首先新建conda环境、克隆OpenCompass代码仓库并安装：

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

准备评估数据：

```bash
# Run in the OpenCompass directory
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip {} \;
```



## 评测

OpenCompass在评估时需要指定评估的模型、评估数据集和评估配置。评估配置包括使用的prompt、评估模式等，如以下命令中的`ceval_ppl_93e5ce`即指定了ceval数据集的评估配置文件，该配置文件保存在`./config/datasets/ceval/ceval_ppl_93e5ce.py`，配置文件的介绍可以查看[OpenCompass官方文档](https://opencompass.org.cn/doc)。需要注意的是，对于相同的模型和数据集，不同的评估配置会得到不同的分数。

将`model_name_or_path`替换为模型路径并执行如下命令即可启动评测：

```bash
python run.py --datasets ceval_ppl_93e5ce mmlu_ppl_ac766d cmmlu_ppl_8b9c76 agieval_mixed GaokaoBench_gen SuperGLUE_WiC_ppl FewCLUE_chid_ppl CLUE_afqmc_ppl SuperGLUE_WSC_ppl race_ppl_5831a0 obqa_ppl_6aac9e hellaswag_ppl_a6e128 bbh_gen_5bf00b gsm8k_gen_1d7fe4 mbpp_gen_1e1056 \
    --hf-path $model_name_or_path \
    --tokenizer-path $model_name_or_path \
    --model-kwargs device_map='auto' trust_remote_code=True \
    --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
    --max-out-len 512 \
    --max-seq-len 8192 \
    --batch-size 8 \
    --no-batch-padding \
    --num-gpus 1 \
    --max-workers-per-gpu 2 \
    --max-partition-size 4000 \
    --max-num-workers 32
```