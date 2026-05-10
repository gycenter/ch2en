import math
from dataclasses import dataclass

import torch
from torch import nn

"""
保存模型超参数
"""
@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    src_pad_id: int
    tgt_pad_id: int
    d_model: int = 512
    num_heads: int = 8  #多头注意力会把 512 维拆成 8 个 64 维的注意力头并行计算
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048    #每个 Transformer 层里的 FeedForward : d_model -> d_ff -> d_model
    dropout: float = 0.1
    max_position_embeddings: int = 512

"""
给 token embedding 加入位置信息
正余弦位置编码
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        """
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        """
        position = torch.arange(max_len).unsqueeze(1)
        """
        10000^(-2i/d_model) = exp( log(10000^(-2i/d_model)) )
                     = exp( -2i/d_model * log(10000) )
                     = exp( 2i * (-log(10000)/d_model) )
        """
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        #创建全 0 矩阵
        pe = torch.zeros(max_len, d_model)
        #偶数项
        pe[:, 0::2] = torch.sin(position * div_term)
        #奇数项
        pe[:, 1::2] = torch.cos(position * div_term)

        #把 pe 注册到模型里，但它不是可训练参数
        #位置编码是固定公式算出来的，不需要训练，所以用 buffer
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        """
        模型输入:[batch_size, seq_len, d_model]
        位置编码:[1, seq_len, d_model]
        """
        x = x + self.pe[:, :seq_len, :] #[1, seq_len, d_model]
        return self.dropout(x)

"""
创建 decoder 的未来遮挡 mask,防止偷看
"""
def create_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        #创建一个布尔矩阵，上三角为True
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1, #表示主对角线上方一格开始保留
    )

"""
缩放点积注意力
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: #output, attn_weights
        #query.shape = [batch_size, num_heads, seq_len, d_k]
        #key.shape = [batch_size, num_heads, key_len, d_k]
        d_k = query.size(-1)

        #QK^T / sqrt(d_k)
        #为什么除以 sqrt(d_k)？
        #如果维度 d_k 很大，点积值会变得很大，softmax 后容易过于尖锐，导致梯度不稳定。
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        #如果有 mask，就把不能看的位置设置成一个极小值
        #scores.masked_fill(attn_mask, value):凡是 attn_mask 为 True 的位置，都替换成 value
        #torch.finfo(scores.dtype).min:当前浮点类型能表示的最小值，近似负无穷
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        #对最后一维key_len做 softmax    
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #softmax(QK^T / sqrt(d_k)) V
        output = torch.matmul(attn_weights, value)

        return output, attn_weights

"""
多头注意力
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        #检查是否能整除
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        #多头注意力合并后，再经过一个线性层
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #query.shape = [batch_size, num_heads, seq_len, d_k]
        batch_size = query.size(0)

        #[batch_size, num_heads, seq_len, d_k]
        query = self._split_heads(self.q_proj(query), batch_size)
        key = self._split_heads(self.k_proj(key), batch_size)
        value = self._split_heads(self.v_proj(value), batch_size)

        #调整 mask 形状方便广播
        if attn_mask is not None:
            #decoder 的 subsequent mask：[tgt_len, tgt_len] -> [1, 1, tgt_len, tgt_len]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            #[batch_size, query_len, key_len]->[batch_size, 1, query_len, key_len]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
        #点积
        attn_output, attn_weights = self.attention(query, key, value, attn_mask)
        #合并
        attn_output = self._combine_heads(attn_output, batch_size)
        #线性层
        output = self.out_proj(attn_output)
        return output, attn_weights

    #x.shape = [batch_size, seq_len, d_model]->[batch_size, seq_len, num_heads, d_k]s
    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    ##[batch_size, num_heads, seq_len, d_k]->[batch_size, seq_len, d_model]
    def _combine_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

"""
Transformer 中的前馈网络
"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        #512 -> 2048 -> 512
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

"""
一个 Transformer Encoder 层
self-attention
残差连接 + LayerNorm
feed forward
残差连接 + LayerNorm
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

"""
一个 Transformer Decoder 层
masked self-attention
残差连接 + LayerNorm
cross-attention
残差连接 + LayerNorm
feed forward
残差连接 + LayerNorm
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        #用 tgt_mask 遮挡padding 位置和未来位置
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        #query 来自 decoder 当前状态
        #key/value 来自 encoder 输出 memory
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        #memory 是 Encoder 的输出
        cross_attn_output, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

"""
多层 Encoder 堆叠
"""
class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

"""
多层 Decoder 堆叠
"""
class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x

"""
完整的中英翻译模型
中文 embedding
英文 embedding
位置编码
Encoder
Decoder
输出投影层
"""
class TransformerSeq2Seq(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        #[batch_size, src_seq_len]->[batch_size, src_seq_len, d_model]
        self.src_embedding = nn.Embedding(
            config.src_vocab_size,
            config.d_model,
            padding_idx=config.src_pad_id,
        )
        #[batch_size, tgt_seq_len]->[batch_size, tgt_seq_len, d_model]
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size,
            config.d_model,
            padding_idx=config.tgt_pad_id,
        )
        #源语言和目标语言共用同一个位置编码模块
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            max_len=config.max_position_embeddings,
            dropout=config.dropout,
        )
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        #把 decoder 输出的隐藏状态转换成英文词表大小的 logits
        #[batch_size, tgt_seq_len, d_model]->[batch_size, tgt_seq_len, tgt_vocab_size]
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)
        self._reset_parameters()

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_mask, tgt_mask, memory_mask = self.build_attention_masks(
            src_input_ids,
            tgt_input_ids,
            src_key_padding_mask,
            tgt_key_padding_mask,
        )

        memory = self.encode(src_input_ids, src_mask=src_mask)
        decoder_output = self.decode(tgt_input_ids, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.output_projection(decoder_output)

    def encode(
        self,
        src_input_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #embedding 初始化值通常比较小，乘以sqrt(d_model)可以让 embedding 的尺度和位置编码更匹配。
        src_emb = self.src_embedding(src_input_ids) * math.sqrt(self.config.d_model)
        #位置编码
        src_emb = self.positional_encoding(src_emb)
        return self.encoder(src_emb, src_mask=src_mask)

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embedding(tgt_input_ids) * math.sqrt(self.config.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        return self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    """
    构建模型需要的三种 mask
    """
    def build_attention_masks(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        #如果没有传 source padding mask，就自动生成
        if src_key_padding_mask is None:
            src_key_padding_mask = src_input_ids.eq(self.config.src_pad_id)
        #如果没有传 target padding mask，就自动生成
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = tgt_input_ids.eq(self.config.tgt_pad_id)

        src_mask = src_key_padding_mask.unsqueeze(1)

        #[batch_size, 1, tgt_seq_len]
        tgt_padding_mask = tgt_key_padding_mask.unsqueeze(1)

        subsequent_mask = create_subsequent_mask(tgt_input_ids.size(1), tgt_input_ids.device)
        
        #合并 target padding mask 和未来 mask:只要一个位置满足以下任一条件，就被 mask
        #tgt_padding_mask:[batch_size, 1, tgt_seq_len]
        #subsequent_mask.unsqueeze(0):[1, tgt_seq_len, tgt_seq_len]
        #广播后[batch_size, tgt_seq_len, tgt_seq_len]
        tgt_mask = tgt_padding_mask | subsequent_mask.unsqueeze(0)
        
        #decoder 每个目标位置，在看 encoder memory 时，都不能看 source 的 padding 位置
        #[batch_size, tgt_seq_len, src_seq_len]
        memory_mask = src_key_padding_mask.unsqueeze(1).expand(-1, tgt_input_ids.size(1), -1)
        
        return src_mask, tgt_mask, memory_mask

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
