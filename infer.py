import argparse
import json
from pathlib import Path
from typing import Any

import torch

from simple_tokenizer import Vocab, ZhEnTokenizer
from transformer_model import TransformerConfig, TransformerSeq2Seq, create_subsequent_mask

DEFAULT_OUTPUT_DIR = "./outputs/scratch-zh-en"
DEFAULT_CHECKPOINT_NAME = "best"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer zh->en translations from a trained scratch Transformer.")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_name", default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint subdirectory name, like best or .")
    parser.add_argument("--text", default=None, help="Translate a single Chinese sentence.")
    parser.add_argument("--input_file", default=None, help="Read Chinese sentences from a text file, one sentence per line.")
    parser.add_argument("--output_file", default=None, help="Save translations to a file. JSON if suffix is .json, otherwise plain text.")
    parser.add_argument("--source_column", default="chinese")
    parser.add_argument("--target_column", default="english")
    parser.add_argument("--max_source_length", type=int, default=96)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size. 1 means greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_predictions", type=int, default=5)
    return parser.parse_args()


def load_json_or_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_input_sentences(args: argparse.Namespace) -> list[str]:
    if args.text is not None:
        return [args.text.strip()]
    if args.input_file is None:
        raise ValueError("Please provide either --text or --input_file.")
    path = Path(args.input_file)
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")
    return load_text_lines(path)


def load_references(args: argparse.Namespace) -> list[str] | None:
    if args.reference_file is None:
        return None
    path = Path(args.reference_file)
    if not path.exists():
        raise ValueError(f"Reference file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = load_json_or_jsonl_lines(path)
        if args.reference_column is None:
            raise ValueError("When using JSONL references, please provide --reference_column.")
        return [str(row[args.reference_column]).strip() for row in rows]
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("data", [])
        if args.reference_column is None:
            raise ValueError("When using JSON references, please provide --reference_column.")
        return [str(row[args.reference_column]).strip() for row in rows]
    return load_text_lines(path)


def print_environment() -> None:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))


def load_checkpoint_bundle(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    config_path = checkpoint_dir / "config.json"
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json in {checkpoint_dir}")
    if not tokenizer_path.exists():
        raise ValueError(f"Missing tokenizer.json in {checkpoint_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    return config, tokenizer_payload["src_vocab"], tokenizer_payload["tgt_vocab"]


def build_model(checkpoint_dir: Path, device: torch.device) -> tuple[TransformerSeq2Seq, ZhEnTokenizer]:
    config_dict, src_vocab_dict, tgt_vocab_dict = load_checkpoint_bundle(checkpoint_dir)
    tokenizer = ZhEnTokenizer(
        src_vocab=Vocab.from_dict(src_vocab_dict),
        tgt_vocab=Vocab.from_dict(tgt_vocab_dict),
    )
    config = TransformerConfig(**config_dict)
    model = TransformerSeq2Seq(config).to(device)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise ValueError(f"Missing checkpoint.pt in {checkpoint_dir}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def greedy_decode(model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, source_text: str, device: torch.device, max_source_length: int, max_new_tokens: int) -> list[int]:
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
        logits = model.output_projection(output[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if int(next_token.item()) == tokenizer.tgt_eos_id:
            break
    return generated.squeeze(0).tolist()


@torch.no_grad()
def greedy_decode_batch(
    model: TransformerSeq2Seq,
    tokenizer: ZhEnTokenizer,
    source_texts: list[str],
    device: torch.device,
    max_source_length: int,
    max_new_tokens: int,
) -> list[list[int]]:
    sequences = []
    for text in source_texts:
        ids = tokenizer.encode_src(text, max_length=max_source_length) or [tokenizer.src_vocab.unk_id]
        sequences.append(torch.tensor(ids, dtype=torch.long))
    src_input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=tokenizer.src_pad_id).to(device)
    src_key_padding_mask = src_input_ids.eq(tokenizer.src_pad_id)
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


@torch.no_grad()
def beam_search_decode(
    model: TransformerSeq2Seq,
    tokenizer: ZhEnTokenizer,
    source_text: str,
    device: torch.device,
    max_source_length: int,
    max_new_tokens: int,
    beam_size: int,
    temperature: float = 1.0,
) -> list[int]:
    if beam_size <= 1:
        return greedy_decode(model, tokenizer, source_text, device, max_source_length, max_new_tokens)

    src_ids = tokenizer.encode_src(source_text, max_length=max_source_length) or [tokenizer.src_vocab.unk_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = src.eq(tokenizer.src_pad_id)
    memory = model.encode(src, src_mask=src_pad_mask.unsqueeze(1))

    beams: list[tuple[list[int], float, bool]] = [([tokenizer.tgt_bos_id], 0.0, False)]
    for _ in range(max_new_tokens):
        candidates: list[tuple[list[int], float, bool]] = []
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, finished))
                continue
            generated = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_pad_mask = generated.eq(tokenizer.tgt_pad_id)
            tgt_mask = tgt_pad_mask.unsqueeze(1) | create_subsequent_mask(generated.size(1), device).unsqueeze(0)
            memory_mask = src_pad_mask.unsqueeze(1).expand(-1, generated.size(1), -1)
            output = model.decode(generated, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            logits = model.output_projection(output[:, -1, :]) / max(temperature, 1e-6)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)
            for tok_score, tok_id in zip(topk_log_probs.squeeze(0).tolist(), topk_ids.squeeze(0).tolist()):
                new_seq = seq + [int(tok_id)]
                new_score = score + float(tok_score)
                candidates.append((new_seq, new_score, int(tok_id) == tokenizer.tgt_eos_id))
        beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)[:beam_size]
        if all(finished for _, _, finished in beams):
            break

    best_seq, _, _ = max(beams, key=lambda item: item[1] / max(1, len(item[0]) - 1))
    return best_seq


def translate_texts(model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, texts: list[str], device: torch.device, args: argparse.Namespace) -> list[str]:
    outputs: list[str] = []
    for idx, text in enumerate(texts, start=1):
        ids = beam_search_decode(model, tokenizer, text, device, args.max_source_length, args.max_new_tokens, args.beam_size, args.temperature)
        outputs.append(tokenizer.decode_tgt(ids, skip_special_tokens=True))
        print(f"[{idx}/{len(texts)}] {text} -> {outputs[-1]}")
    return outputs


@torch.no_grad()
def translate_texts_batch(model: TransformerSeq2Seq, tokenizer: ZhEnTokenizer, texts: list[str], device: torch.device, args: argparse.Namespace) -> list[str]:
    if args.beam_size > 1:
        return translate_texts(model, tokenizer, texts, device, args)
    outputs: list[str] = []
    batch_size = max(1, args.sample_predictions)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = greedy_decode_batch(model, tokenizer, batch_texts, device, args.max_source_length, args.max_new_tokens)
        batch_outputs = [tokenizer.decode_tgt(ids, skip_special_tokens=True) for ids in batch_ids]
        outputs.extend(batch_outputs)
        for offset, (src, pred) in enumerate(zip(batch_texts, batch_outputs), start=start + 1):
            print(f"[{offset}/{len(texts)}] {src} -> {pred}")
    return outputs


def save_outputs(output_file: str, sources: list[str], predictions: list[str]) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = []
        for idx, (src, pred) in enumerate(zip(sources, predictions)):
            payload.append({"index": idx, "source": src, "prediction": pred})
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        lines = [f"{src}\t{pred}" for src, pred in zip(sources, predictions)]
        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    print_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint_name not in {"", "."}:
        checkpoint_dir = checkpoint_dir / args.checkpoint_name
    model, tokenizer = build_model(checkpoint_dir, device)

    texts = load_input_sentences(args)
    predictions = translate_texts_batch(model, tokenizer, texts, device, args)

    if args.output_file:
        save_outputs(args.output_file, texts, predictions)
        print("Saved predictions to", args.output_file)

    if len(predictions) > 1:
        preview = predictions[: args.sample_predictions]
        print("Preview:")
        for idx, item in enumerate(preview, start=1):
            print(f"{idx}. {item}")
    else:
        print("Prediction:", predictions[0] if predictions else "")


if __name__ == "__main__":
    main()
