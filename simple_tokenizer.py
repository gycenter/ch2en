"""
把中英文字符串变成模型可以处理的整数 id序列
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jieba


PAD_TOKEN = "<pad>" #padding
UNK_TOKEN = "<unk>" #unknown
BOS_TOKEN = "<bos>" #begin of sequence
EOS_TOKEN = "<eos>" #end of sequence
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

"""
正则表达式
[A-Za-z]+(?:'[A-Za-z]+)?  英文单词（带'的单词）
\d+(?:\.\d+)?  数字
[^\w\s]  非单词和空格的符号
"""
_EN_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)

"""
中文分词
input:str
output:list[str]
"""
def tokenize_zh(text: str) -> list[str]:
    return [token.strip() for token in jieba.lcut(text.strip()) if token.strip()]
    #token.strip() 去除首尾空格/空字符串为False
    #jieba.lcut()list cut分词保存为列表形式

"""
英文分词
input:str
output:list[str]
"""
def tokenize_en(text: str) -> list[str]:
    return _EN_TOKEN_PATTERN.findall(text.lower().strip())

"""
token -> id : encode_tokens
id -> token : decode_ids
build
"""
@dataclass  
class Vocab:
    token_to_id: dict[str, int]

    def __post_init__(self) -> None:
        #根据 token_to_id 生成反向字典 id_to_token
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        #逐个检查特殊 token 是否都存在于词表中
        for token in SPECIAL_TOKENS:
            if token not in self.token_to_id:
                raise ValueError(f"Missing required special token: {token}")

    """
    可以直接引用vocab.xxx_id，找到<xxx>的id
    """
    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    """
    len(vocab)
    """
    def __len__(self) -> int:
        return len(self.token_to_id)

    """
    tokens -> ids
    input : Iterable[str]可迭代的字符串集合
    output : list[int]
    """
    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    """
    ids -> tokens
    input : 
        Iterable[int]可迭代的整数集合
        skip_special_tokens: bool = True 是否跳过特殊token
    output : 
        list[str]
    """
    def decode_ids(self, ids: Iterable[int], skip_special_tokens: bool = True) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK_TOKEN)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    """
    把 Vocab 对象转换成普通字典,便于保存成 JSON
    """
    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    """
    从普通字典创建一个 Vocab 对象
    """
    @classmethod    #@classmethod: 类方法，可以直接通过类名调用
    def from_dict(cls, token_to_id: dict[str, int]) -> "Vocab":
        return cls(token_to_id=dict(token_to_id))

    """
    根据已经分好词的文本自动构建词表
    input:
        tokenized_texts: Iterable[Iterable[str]] 可迭代的字符串集合的集合（外层是很多句子，内层是每句话的 token）
        max_vocab_size: int 最大词表大小
        min_freq: int = 1 每个token出现最小频率
    output:
        Vocab
    process:
        1. 检查最大词表大小是否大于特殊token数量
        2. 统计词频
        3. 创建词表
        4. 返回Vocab对象
    """
    @classmethod
    def build(
        cls,
        tokenized_texts: Iterable[Iterable[str]],
        max_vocab_size: int,
        min_freq: int = 1,
    ) -> "Vocab":
        #检查最大词表大小是否大于特殊token数量
        if max_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("max_vocab_size must be at least the number of special tokens.")
        
        #统计词频
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        #创建词表
        token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for token, freq in counter.most_common():   #counter.most_common():返回出现次数最多的 n 个元素及其计数
            if freq < min_freq:
                continue
            if token in token_to_id:
                continue
            if len(token_to_id) >= max_vocab_size:
                break
            token_to_id[token] = len(token_to_id)

        return cls(token_to_id=token_to_id)

"""
句子 -> ids
ids -> 句子
从平行语料构建两个词表
维护两个词表：
    src_vocab: 中文词表
    tgt_vocab: 英文词表
save/load词表
"""
@dataclass
class ZhEnTokenizer:
    src_vocab: Vocab
    tgt_vocab: Vocab

    """
    中/英文词表特殊tokenID
    """
    @property
    def src_pad_id(self) -> int:
        return self.src_vocab.pad_id

    @property
    def tgt_pad_id(self) -> int:
        return self.tgt_vocab.pad_id

    @property
    def tgt_bos_id(self) -> int:
        return self.tgt_vocab.bos_id

    @property
    def tgt_eos_id(self) -> int:
        return self.tgt_vocab.eos_id

    """
    中文句子 -> ids
    input:
        text: str 中文句子
        max_length: int | None = None 最大长度
    output:
        list[int]
    """
    def encode_src(self, text: str, max_length: int | None = None) -> list[int]:
        ids = self.src_vocab.encode_tokens(tokenize_zh(text))
        if max_length is not None:  #如果设置了最大长度就截断，默认值None就不截断
            ids = ids[:max_length]
        return ids

    """
    英文句子 -> ids
    input:
        text: str 英文句子
        max_length: int | None = None 最大长度
        add_bos: bool = False 是否添加bos
        add_eos: bool = False 是否添加eos
    output:
        list[int]
    """
    def encode_tgt(
        self,
        text: str,
        max_length: int | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.tgt_vocab.encode_tokens(tokenize_en(text))

        if max_length is not None:
            special_token_count = int(add_bos) + int(add_eos)
            max_content_length = max(0, max_length - special_token_count)
            ids = ids[:max_content_length]

        if add_bos:
            ids = [self.tgt_vocab.bos_id] + ids
        if add_eos:
            ids = ids + [self.tgt_vocab.eos_id]
        return ids

    """
    中文ids -> tokens -> 文本
    """
    def decode_src(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.src_vocab.decode_ids(ids, skip_special_tokens=skip_special_tokens))

    """
    英文ids -> tokens -> 文本
    """
    def decode_tgt(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = self.tgt_vocab.decode_ids(ids, skip_special_tokens=skip_special_tokens)
        return detokenize_en(tokens)

    """
    保存词表--把 tokenizer 保存到文件夹里的 tokenizer.json 文件中
    input:
        directory: str | Path 保存目录
    output:
        None
    """
    def save(self, directory: str | Path) -> None:
        #把传入的目录转换成 Path 对象。
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        #创建一个字典，准备写入 JSON 文件
        payload = {
            "src_vocab": self.src_vocab.to_dict(),
            "tgt_vocab": self.tgt_vocab.to_dict(),
        }
        (path / "tokenizer.json").write_text(
            #把 payload 字典转换成 JSON 字符串
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    """
    加载词表--从文件夹里的 tokenizer.json 文件中加载词表
    input:
        directory: str | Path 加载目录
    output:
        ZhEnTokenizer
    """
    @classmethod
    def load(cls, directory: str | Path) -> "ZhEnTokenizer":
        path = Path(directory) / "tokenizer.json"
        #把 JSON 字符串转换成 Python 字典
        payload = json.loads(path.read_text(encoding="utf-8"))
        #创建 ZhEnTokenizer 对象，传入 src_vocab 和 tgt_vocab
        return cls(
            src_vocab=Vocab.from_dict(payload["src_vocab"]),
            tgt_vocab=Vocab.from_dict(payload["tgt_vocab"]),
        )

    """
    从平行文本构建词表
    input:
        source_texts: Iterable[str] 源文本集合
        target_texts: Iterable[str] 目标文本集合
        max_src_vocab_size: int = 30000 最大源词表大小
        max_tgt_vocab_size: int = 30000 最大目标词表大小
        min_freq: int = 2 每个token出现最小频率
    """
    @classmethod
    def build_from_parallel_texts(
        cls,
        source_texts: Iterable[str],
        target_texts: Iterable[str],
        max_src_vocab_size: int = 30000,
        max_tgt_vocab_size: int = 30000,
        min_freq: int = 2,
    ) -> "ZhEnTokenizer":
        source_tokenized = [tokenize_zh(text) for text in source_texts]
        target_tokenized = [tokenize_en(text) for text in target_texts]
        return cls(
            src_vocab=Vocab.build(source_tokenized, max_src_vocab_size, min_freq=min_freq),
            tgt_vocab=Vocab.build(target_tokenized, max_tgt_vocab_size, min_freq=min_freq),
        )


def detokenize_en(tokens: Iterable[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    text = text.replace(" 'm", "'m")
    return text.strip()
