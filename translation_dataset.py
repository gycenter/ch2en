"""
把 tokenizer 产生的 token id,组织成 PyTorch 模型训练需要的 batch
train_dataset = TranslationDataset(xxx)
collator = TranslationCollator(0,0)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator,
)
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from simple_tokenizer import ZhEnTokenizer

"""
src_input_ids : Encoder 的输入, 中文
tgt_input_ids : Decoder 的输入, 英文, 前面加 <bos>
labels : 训练目标, 英文, 后面加 <eos>
"""
@dataclass
class TranslationExample:
    src_input_ids: list[int]
    tgt_input_ids: list[int]
    labels: list[int]

"""
继承 torch.utils.data.Dataset,可以被DataLoader 使用
__len__ : 返回数据集长度
__getitem__ : 根据索引返回一条样本
"""
class TranslationDataset(Dataset):
    def __init__(
        self,
        raw_dataset: Any,
        tokenizer: ZhEnTokenizer,
        source_column: str,
        target_column: str,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.target_column = target_column
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.raw_dataset)

    """
    根据索引取出一条原始数据，并转换成模型训练需要的三个 id 列表
    """
    def __getitem__(self, index: int) -> TranslationExample:
        item = self.raw_dataset[index]
        source_text = str(item[self.source_column])
        target_text = str(item[self.target_column])

        src_input_ids = self.tokenizer.encode_src(
            source_text,
            max_length=self.max_source_length,
        )
        tgt_input_ids = self.tokenizer.encode_tgt(
            target_text,
            max_length=self.max_target_length,
            add_bos=True,
            add_eos=False,
        )
        labels = self.tokenizer.encode_tgt(
            target_text,
            max_length=self.max_target_length,
            add_bos=False,
            add_eos=True,
        )

        return TranslationExample(
            src_input_ids=src_input_ids,
            tgt_input_ids=tgt_input_ids,
            labels=labels,
        )

"""
把一组长短不同的样本，补齐成同样长度的 batch tensor
"""
@dataclass
class TranslationCollator:
    src_pad_id: int
    tgt_pad_id: int
    #PyTorch 的 CrossEntropyLoss 支持：ignore_index=-100
    #这表示：label 等于 -100 的位置，不参与 loss 计算
    label_pad_id: int = -100    
    
    """
    让对象可以像函数一样被调用
    input:
        list[TranslationExample] 一组样本
    output:
        dict[str, torch.Tensor]
    """
    def __call__(self, examples: list[TranslationExample]) -> dict[str, torch.Tensor]:
        """
        把 list[int] 转成 tensor
        """
        src_sequences = [
            torch.tensor(example.src_input_ids, dtype=torch.long)
            for example in examples
        ]
        tgt_sequences = [
            torch.tensor(example.tgt_input_ids, dtype=torch.long)
            for example in examples
        ]
        label_sequences = [
            torch.tensor(example.labels, dtype=torch.long)
            for example in examples
        ]

        """
        from torch.nn.utils.rnn import pad_sequence
        把长短不一的一组 tensor 补齐成相同长度
        """
        src_input_ids = pad_sequence(
            src_sequences,
            batch_first=True,
            padding_value=self.src_pad_id,
        )
        tgt_input_ids = pad_sequence(
            tgt_sequences,
            batch_first=True,
            padding_value=self.tgt_pad_id,  #用中文 <pad> 的 id 来补齐
        )
        labels = pad_sequence(
            label_sequences,
            batch_first=True,
            padding_value=self.label_pad_id,
        )

        #补齐之后，模型还需要知道哪些位置是真实 token，哪些位置是 padding
        #True 表示需要被忽略的位置
        src_key_padding_mask = src_input_ids.eq(self.src_pad_id)
        tgt_key_padding_mask = tgt_input_ids.eq(self.tgt_pad_id)


        #返回 batch 字典
        return {
            "src_input_ids": src_input_ids,
            "tgt_input_ids": tgt_input_ids,
            "labels": labels,
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask,
        }

"""
collator = build_translation_collator(tokenizer)
->
collator = TranslationCollator(
    src_pad_id=tokenizer.src_pad_id,
    tgt_pad_id=tokenizer.tgt_pad_id,
)
"""
def build_translation_collator(tokenizer: ZhEnTokenizer) -> TranslationCollator:
    return TranslationCollator(
        src_pad_id=tokenizer.src_pad_id,
        tgt_pad_id=tokenizer.tgt_pad_id,
    )
