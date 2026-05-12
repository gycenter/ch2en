import argparse
import json
import math
import os
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import sacrebleu

import swanlab


from simple_tokenizer import ZhEnTokenizer
from transformer_model import TransformerConfig, TransformerSeq2Seq, create_subsequent_mask
from translation_dataset import TranslationDataset, build_translation_collator


DEFAULT_DATASET_ZIP = "./translation2019zh.zip"


@dataclass
class DatasetFields:
    source: str
    target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a zh->en Transformer from scratch with standard PyTorch."
    )
    parser.add_argument("--dataset_zip", default=DEFAULT_DATASET_ZIP)
    parser.add_argument("--source_column", default="chinese")
    parser.add_argument("--target_column", default="english")
    parser.add_argument("--output_dir", default="./outputs/scratch-zh-en")
    parser.add_argument("--max_source_length", type=int, default=96)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--min_source_length", type=int, default=2)
    parser.add_argument("--max_source_chars", type=int, default=128)
    parser.add_argument("--min_target_words", type=int, default=2)
    parser.add_argument("--max_target_words", type=int, default=160)
    parser.add_argument("--max_train_samples", type=int, default=200000)
    parser.add_argument("--max_eval_samples", type=int, default=5000)
    parser.add_argument("--validation_split_ratio", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--sample_predictions", type=int, default=5)
    parser.add_argument("--max_src_vocab_size", type=int, default=30000)
    parser.add_argument("--max_tgt_vocab_size", type=int, default=30000)
    parser.add_argument("--min_vocab_freq", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_noam", action="store_true", default=True)
    parser.add_argument("--no_use_noam", action="store_false", dest="use_noam")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--noam_factor", type=float, default=1.0)
    parser.add_argument("--swanlab_project", default="zh-en-transformer")
    parser.add_argument("--swanlab_run_name", default=None)
    return parser.parse_args()



def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_flags(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.max_train_samples = min(args.max_train_samples, 1000)
        args.max_eval_samples = min(args.max_eval_samples, 200)
        args.num_train_epochs = min(args.num_train_epochs, 1.0)
        args.logging_steps = min(args.logging_steps, 10)
        args.d_model = min(args.d_model, 64)
        args.d_ff = min(args.d_ff, 128)
        args.num_encoder_layers = min(args.num_encoder_layers, 1)
        args.num_decoder_layers = min(args.num_decoder_layers, 1)


def print_environment() -> None:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))


def init_swanlab(args: argparse.Namespace) -> Any | None:
    run = swanlab.init(
        project=args.swanlab_project,
        name=args.swanlab_run_name,
        config=vars(args),
    )
    print("SwanLab initialized.")
    return run


def log_swanlab(run: Any | None, metrics: dict[str, float], step: int | None = None) -> None:
    if run is None:
        return
    if step is not None:
        swanlab.log(metrics, step=step)
    else:
        swanlab.log(metrics)


class SimpleDataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | str) -> Any:
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]

    def filter(self, fn: Any, desc: str | None = None) -> "SimpleDataset":
        if desc:
            print(desc)
        return SimpleDataset([row for row in self.rows if fn(row)])

    def map(self, fn: Any, desc: str | None = None) -> "SimpleDataset":
        if desc:
            print(desc)
        return SimpleDataset([fn(row) for row in self.rows])

    def shuffle(self, seed: int) -> "SimpleDataset":
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return SimpleDataset(rows)

    def select(self, indices: range) -> "SimpleDataset":
        return SimpleDataset([self.rows[idx] for idx in indices])

    def train_test_split(self, test_size: float, seed: int) -> dict[str, "SimpleDataset"]:
        rows = self.shuffle(seed).rows
        test_count = max(1, int(len(rows) * test_size))
        return {"train": SimpleDataset(rows[test_count:]), "test": SimpleDataset(rows[:test_count])}


class NoamScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, factor: float) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def rate(self) -> float:
        step = max(1, self.step_num)
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def state_dict(self) -> dict[str, int]:
        return {"step_num": self.step_num}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step_num = int(state_dict.get("step_num", 0))


def infer_translation2019zh_members(zip_path: str) -> tuple[str, str | None]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    train_member = next((name for name in names if "train" in name.lower()), None)
    valid_member = next((name for name in names if any(token in name.lower() for token in ("valid", "validation", "dev", "test"))), None)
    if train_member is None:
        raise ValueError(f"Could not find a train file inside '{zip_path}'.")
    return train_member, valid_member


def load_jsonl_from_zip(zip_path: str, member_name: str, source_column: str, target_column: str) -> SimpleDataset:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw_fp:
            for raw_line in raw_fp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append({source_column: str(item[source_column]), target_column: str(item[target_column])})
    if not rows:
        raise ValueError(f"No usable rows found in '{member_name}' inside '{zip_path}'.")
    return SimpleDataset(rows)


def load_raw_dataset(args: argparse.Namespace) -> dict[str, SimpleDataset]:
    zip_path = Path(args.dataset_zip)
    if not zip_path.exists():
        raise ValueError(f"Dataset zip '{zip_path}' was not found.")
    train_member, valid_member = infer_translation2019zh_members(str(zip_path))
    dataset = {"train": load_jsonl_from_zip(str(zip_path), train_member, args.source_column, args.target_column)}
    if valid_member is not None:
        dataset["validation"] = load_jsonl_from_zip(str(zip_path), valid_member, args.source_column, args.target_column)
    print("Loaded dataset from zip:", zip_path, "train member:", train_member, "validation member:", valid_member)
    return dataset


def detect_splits(dataset: dict[str, SimpleDataset]) -> tuple[SimpleDataset, SimpleDataset | None]:
    train_split = dataset.get("train")
    eval_split = None
    for candidate in ("validation", "valid", "dev", "test"):
        if candidate in dataset:
            eval_split = dataset[candidate]
            break
    if train_split is None:
        raise ValueError("No train split found in dataset.")
    return train_split, eval_split


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def is_valid_pair(example: dict[str, Any], fields: DatasetFields, args: argparse.Namespace) -> bool:
    source = normalize_text(example.get(fields.source, ""))
    target = normalize_text(example.get(fields.target, ""))
    if not source or not target:
        return False
    if len(source) < args.min_source_length or len(source) > args.max_source_chars:
        return False
    target_words = target.split()
    if len(target_words) < args.min_target_words or len(target_words) > args.max_target_words:
        return False
    return True


def clean_example(example: dict[str, Any], fields: DatasetFields) -> dict[str, str]:
    return {fields.source: normalize_text(example.get(fields.source, "")), fields.target: normalize_text(example.get(fields.target, ""))}


def filter_and_clean_dataset(dataset: SimpleDataset, fields: DatasetFields, args: argparse.Namespace, name: str) -> SimpleDataset:
    filtered = dataset.filter(lambda example: is_valid_pair(example, fields, args), desc=f"Filtering {name}")
    cleaned = filtered.map(lambda example: clean_example(example, fields), desc=f"Cleaning {name}")
    print(f"{name} size after filtering:", len(cleaned))
    if len(cleaned) == 0:
        raise ValueError(f"{name} split is empty after filtering.")
    return cleaned


def take_subset(dataset: SimpleDataset, limit: int, seed: int) -> SimpleDataset:
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def ensure_eval_split(train_dataset: SimpleDataset, eval_dataset: SimpleDataset | None, args: argparse.Namespace) -> tuple[SimpleDataset, SimpleDataset]:
    if eval_dataset is not None:
        return train_dataset, eval_dataset
    split = train_dataset.train_test_split(test_size=args.validation_split_ratio, seed=args.seed)
    print("No validation split found. Created one from train with ratio", args.validation_split_ratio)
    return split["train"], split["test"]


def preview_samples(dataset: SimpleDataset, fields: DatasetFields, count: int = 3) -> None:
    sample_count = min(count, len(dataset))
    print(f"Previewing {sample_count} training samples:")
    for idx in range(sample_count):
        item = dataset[idx]
        print(json.dumps({"index": idx, "source": item[fields.source], "target": item[fields.target]}, ensure_ascii=False))


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def train_one_epoch(model: TransformerSeq2Seq, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, scheduler: NoamScheduler | None, grad_clip: float, log_every: int, run: Any | None = None) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    last_lr = optimizer.param_groups[0]["lr"]
    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["src_input_ids"], batch["tgt_input_ids"], batch["src_key_padding_mask"], batch["tgt_key_padding_mask"])
        loss = compute_loss(logits, batch["labels"])
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        if scheduler is not None:
            last_lr = scheduler.step()
        optimizer.step()
        total_loss += float(loss.item())
        if log_every > 0 and step % log_every == 0:
            print(f"step={step} train_loss={loss.item():.4f} lr={last_lr:.6g}")
            log_swanlab(run, {"train/loss": float(loss.item()), "train/lr": last_lr, "train/step": float(step)})
    avg_loss = total_loss / max(1, len(dataloader))
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20.0)), "lr": last_lr}


@torch.no_grad()
def evaluate_loss(model: TransformerSeq2Seq, dataloader: DataLoader, device: torch.device, run: Any | None = None, epoch: int | None = None) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["src_input_ids"], batch["tgt_input_ids"], batch["src_key_padding_mask"], batch["tgt_key_padding_mask"])
        total_loss += float(compute_loss(logits, batch["labels"]).item())
    avg_loss = total_loss / max(1, len(dataloader))
    metrics = {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20.0))}
    if run is not None:
        payload = {"eval/loss": metrics["loss"], "eval/ppl": metrics["ppl"]}
        if epoch is not None:
            payload["eval/epoch"] = float(epoch)
        log_swanlab(run, payload, step=epoch)
    return metrics


@torch.no_grad()
def greedy_decode(model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, source_text: str, device: torch.device, max_source_length: int, max_new_tokens: int) -> list[int]:
    model.eval()
    src_ids = tokenizer.encode_src(source_text, max_length=max_source_length) or [tokenizer.src_vocab.unk_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = src.eq(tokenizer.src_pad_id)
    memory = model.encode(src, src_mask=src_pad_mask.unsqueeze(1))
    generated = torch.tensor([[tokenizer.tgt_bos_id]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        tgt_pad_mask = generated.eq(tokenizer.tgt_pad_id)
        tgt_mask = tgt_pad_mask.unsqueeze(1) | create_subsequent_mask(generated.size(1), device).unsqueeze(0)
        memory_mask = src_pad_mask.unsqueeze(1).expand(-1, generated.size(1), -1)
        output = model.decode(generated, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        next_token = model.output_projection(output[:, -1, :]).argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if int(next_token.item()) == tokenizer.tgt_eos_id:
            break
    return generated.squeeze(0).tolist()


def translate(model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, source_text: str, device: torch.device, max_source_length: int, max_new_tokens: int) -> str:
    return tokenizer.decode_tgt(greedy_decode(model, tokenizer, source_text, device, max_source_length, max_new_tokens), skip_special_tokens=True)


def encode_source_batch(tokenizer: ZhEnTokenizer, texts: list[str], max_source_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    for text in texts:
        ids = tokenizer.encode_src(text, max_length=max_source_length) or [tokenizer.src_vocab.unk_id]
        sequences.append(torch.tensor(ids, dtype=torch.long))
    src_input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=tokenizer.src_pad_id).to(device)
    src_key_padding_mask = src_input_ids.eq(tokenizer.src_pad_id)
    return src_input_ids, src_key_padding_mask


@torch.no_grad()
def greedy_decode_batch(
    model: TransformerSeq2Seq,
    tokenizer: ZhEnTokenizer,
    source_texts: list[str],
    device: torch.device,
    max_source_length: int,
    max_new_tokens: int,
) -> list[list[int]]:
    src_input_ids, src_key_padding_mask = encode_source_batch(tokenizer, source_texts, max_source_length, device)
    memory = model.encode(src_input_ids, src_mask=src_key_padding_mask.unsqueeze(1))
    batch_size = src_input_ids.size(0)
    generated = torch.full((batch_size, 1), tokenizer.tgt_bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        tgt_pad_mask = generated.eq(tokenizer.tgt_pad_id)
        tgt_mask = tgt_pad_mask.unsqueeze(1) | create_subsequent_mask(generated.size(1), device).unsqueeze(0)
        memory_mask = src_key_padding_mask.unsqueeze(1).expand(-1, generated.size(1), -1)
        output = model.decode(generated, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        next_token = model.output_projection(output[:, -1, :]).argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        finished = finished | next_token.squeeze(1).eq(tokenizer.tgt_eos_id)
        if bool(finished.all()):
            break

    return [generated[i].tolist() for i in range(batch_size)]


def sacrebleu_corpus_score(references: list[str], hypotheses: list[str]) -> float:
    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


def evaluate_bleu(
    model: TransformerSeq2Seq,
    tokenizer: ZhEnTokenizer,
    raw_eval_dataset: SimpleDataset,
    fields: DatasetFields,
    device: torch.device,
    max_source_length: int,
    max_new_tokens: int,
    batch_size: int,
    run: Any | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    predictions: list[str] = []
    references: list[str] = []
    for start in range(0, len(raw_eval_dataset), max(1, batch_size)):
        batch_items = [raw_eval_dataset[idx] for idx in range(start, min(start + batch_size, len(raw_eval_dataset)))]
        sources = [item[fields.source] for item in batch_items]
        batch_refs = [item[fields.target] for item in batch_items]
        batch_preds = greedy_decode_batch(model, tokenizer, sources, device, max_source_length, max_new_tokens)
        predictions.extend(tokenizer.decode_tgt(ids, skip_special_tokens=True) for ids in batch_preds)
        references.extend(batch_refs)
    bleu = sacrebleu_corpus_score(references, predictions)
    metrics = {"bleu": bleu}
    if run is not None:
        payload = {"eval/bleu": bleu}
        if epoch is not None:
            payload["eval/epoch"] = float(epoch)
        log_swanlab(run, payload, step=epoch)
    return metrics


def save_sample_predictions(model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, raw_eval_dataset: SimpleDataset, fields: DatasetFields, args: argparse.Namespace, device: torch.device, run: Any | None = None, epoch: int | None = None) -> None:
    rows = []
    for idx in range(min(args.sample_predictions, len(raw_eval_dataset))):
        item = raw_eval_dataset[idx]
        source = item[fields.source]
        rows.append({"source": source, "reference": item[fields.target], "prediction": translate(model, tokenizer, source, device, args.max_source_length, args.max_target_length)})
    sample_path = Path(args.output_dir) / "sample_predictions.json"
    sample_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if run is not None:
        log_swanlab(run, {"eval/sample_count": float(len(rows))}, step=epoch)
    print("Saved sample predictions to", sample_path)


def save_checkpoint(output_dir: str | Path, model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, optimizer: torch.optim.Optimizer, scheduler: NoamScheduler | None, epoch: int, metrics: dict[str, float]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    (path / "config.json").write_text(json.dumps(asdict(model.config), ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict() if scheduler else None, "epoch": epoch, "metrics": metrics}, path / "checkpoint.pt")
    print("Saved checkpoint to", path / "checkpoint.pt")


def load_checkpoint(checkpoint_dir: str, model: TransformerSeq2Seq, optimizer: torch.optim.Optimizer, scheduler: NoamScheduler | None, device: torch.device) -> int:
    checkpoint = torch.load(Path(checkpoint_dir) / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0)) + 1


def main() -> None:
    args = parse_args()
    resolve_runtime_flags(args)
    set_seed(args.seed)
    print_environment()
    os.makedirs(args.output_dir, exist_ok=True)
    run = init_swanlab(args)

    fields = DatasetFields(source=args.source_column, target=args.target_column)
    raw_dataset = load_raw_dataset(args)
    train_raw, eval_raw = detect_splits(raw_dataset)
    train_raw = filter_and_clean_dataset(train_raw, fields, args, "train")
    if eval_raw is not None:
        eval_raw = filter_and_clean_dataset(eval_raw, fields, args, "eval")
    train_raw, eval_raw = ensure_eval_split(train_raw, eval_raw, args)
    train_raw = take_subset(train_raw, args.max_train_samples, args.seed)
    eval_raw = take_subset(eval_raw, args.max_eval_samples, args.seed)
    preview_samples(train_raw, fields)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ZhEnTokenizer.build_from_parallel_texts(train_raw[fields.source], train_raw[fields.target], args.max_src_vocab_size, args.max_tgt_vocab_size, args.min_vocab_freq)
    train_dataset = TranslationDataset(train_raw, tokenizer, fields.source, fields.target, args.max_source_length, args.max_target_length)
    eval_dataset = TranslationDataset(eval_raw, tokenizer, fields.source, fields.target, args.max_source_length, args.max_target_length)
    collator = build_translation_collator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collator)

    config = TransformerConfig(src_vocab_size=len(tokenizer.src_vocab), tgt_vocab_size=len(tokenizer.tgt_vocab), src_pad_id=tokenizer.src_pad_id, tgt_pad_id=tokenizer.tgt_pad_id, d_model=args.d_model, num_heads=args.num_heads, num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers, d_ff=args.d_ff, dropout=args.dropout, max_position_embeddings=max(args.max_source_length, args.max_target_length) + 8)
    model = TransformerSeq2Seq(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    scheduler = NoamScheduler(optimizer, config.d_model, args.warmup_steps, args.noam_factor) if args.use_noam else None
    start_epoch = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, device) if args.resume_from_checkpoint else 1

    history: list[dict[str, float]] = []
    best_eval_loss = float("inf")
    total_epochs = int(args.num_train_epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"Epoch {epoch}/{total_epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scheduler, args.grad_clip, args.logging_steps, run=run)
        eval_metrics = evaluate_loss(model, eval_loader, device, run=run, epoch=epoch) if args.do_eval else {"loss": 0.0, "ppl": 0.0}
        metrics = {"epoch": float(epoch), "train_loss": train_metrics["loss"], "train_ppl": train_metrics["ppl"], "eval_loss": eval_metrics["loss"], "eval_ppl": eval_metrics["ppl"]}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        log_swanlab(run, {"train/epoch_loss": train_metrics["loss"], "train/epoch_ppl": train_metrics["ppl"], "eval/epoch_loss": eval_metrics["loss"], "eval/epoch_ppl": eval_metrics["ppl"]}, step=epoch)
        history.append(metrics)
        save_checkpoint(args.output_dir, model, tokenizer, optimizer, scheduler, epoch, metrics)
        if args.do_eval and metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = metrics["eval_loss"]
            save_checkpoint(Path(args.output_dir) / "best", model, tokenizer, optimizer, scheduler, epoch, metrics)
        (Path(args.output_dir) / "trainer_state.json").write_text(json.dumps({"log_history": history}, ensure_ascii=False, indent=2), encoding="utf-8")

    final_train_metrics = history[-1] if history else {}
    (Path(args.output_dir) / "train_results.json").write_text(json.dumps(final_train_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_metrics: dict[str, float] = final_train_metrics
    if args.do_eval:
        bleu_metrics = evaluate_bleu(
            model,
            tokenizer,
            eval_raw,
            fields,
            device,
            args.max_source_length,
            args.max_target_length,
            args.per_device_eval_batch_size,
            run=run,
            epoch=total_epochs,
        )
        eval_metrics = {**final_train_metrics, **bleu_metrics}
        print(json.dumps(bleu_metrics, ensure_ascii=False, indent=2))
    (Path(args.output_dir) / "eval_results.json").write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_sample_predictions(model, tokenizer, eval_raw, fields, args, device, run=run, epoch=total_epochs)
    (Path(args.output_dir) / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved run config to", Path(args.output_dir) / "run_config.json")
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
