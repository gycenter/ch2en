# ch2en

一个从零手搓实现的中英翻译 Transformer 项目，使用 PyTorch 训练中文到英文的序列到序列模型。

## 项目特点

- 基于纯 PyTorch 实现 Transformer 编码器-解码器结构
- 支持中文分词和英文分词
- 支持从平行语料构建双语词表
- 支持训练、验证和推理
- 训练过程可通过 SwanLab 进行可视化监控
- 验证阶段支持 sacrebleu 计算 BLEU

## 主要文件

- `train.py`：训练脚本
- `infer.py`：推理脚本
- `transformer_model.py`：Transformer 模型实现
- `simple_tokenizer.py`：中英文 tokenizer 与词表构建
- `translation_dataset.py`：数据集与 batch 组装
- `requirements.txt`：依赖文件

## 环境依赖

建议使用 Python 3.9+，并安装以下依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

训练脚本默认读取：

```text
./translation2019zh.zip
```

你也可以通过 `--dataset_zip` 指定自己的数据路径。

## 训练


```bash
python3 train.py
```

## 推理

单句翻译：

```bash
python3 infer.py --checkpoint_dir ./outputs/scratch-zh-en --checkpoint_name best --text "今天天气很好"
```

批量翻译：

```bash
python3 infer.py --checkpoint_dir ./outputs/scratch-zh-en --checkpoint_name best --input_file input.txt --output_file result.txt
```

## 输出目录

训练完成后，模型和日志通常保存在 `output_dir` 下，例如：

- `config.json`
- `tokenizer.json`
- `checkpoint.pt`
- `trainer_state.json`
- `train_results.json`
- `eval_results.json`
- `sample_predictions.json`

## 说明

这是一个教学和实验性质的项目，重点在于理解 Transformer 翻译模型从数据处理、训练到推理的完整流程。
