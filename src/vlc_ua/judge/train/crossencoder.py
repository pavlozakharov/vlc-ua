"""Train a cross-encoder decision head on gold JSONL. Runs on a Kaggle T4.

Usage (Kaggle kernel or any box with a GPU)::

    python -m vlc_ua.judge.train.crossencoder \
        --gold /kaggle/input/vlc-gold/attribution.jsonl \
        --task /kaggle/input/vlc-gold/attribution.task.json \
        --base BAAI/bge-reranker-v2-m3 --out /kaggle/working/head-attribution \
        --epochs 2 --lr 2e-5 --max-length 1024 --export-onnx

What it does:

1. Turns each gold row into ``len(options)`` pairs (query = instruction +
   option meaning, document = rendered state) with a one-hot target.
2. Trains the cross-encoder with a listwise cross-entropy over the options
   of each row (softmax across the row's pairs), which is the objective that
   matches how the head is used, not pointwise BCE.
3. Holds out a dev split of the *random* rows, fits a temperature per
   question on it, writes ``head.json`` with the temperatures and the
   metrics, and optionally exports ONNX for CPU serving.

Only ``sample=="random"`` rows are used for dev/temperature; enriched rows
train but never calibrate. Rows with ``source=="human"`` are never dropped
by the per-document cap.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ..calibration import Labelled, accuracy, ece, fit_temperature
from ..evalharness import load_gold, load_task
from ..backends.crossencoder import render_query, render_state


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-rows", type=int, default=4, help="gold rows per step (pairs = rows*options)")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--dev-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--export-onnx", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    task = load_task(args.task)
    gold = load_gold(args.gold)
    rnd = random.Random(args.seed)
    rnd.shuffle(gold)
    rand_rows = [g for g in gold if g.get("sample", "random") == "random"]
    n_dev = int(len(rand_rows) * args.dev_share)
    dev_ids = {g["id"] for g in rand_rows[:n_dev]}
    train_rows = [g for g in gold if g["id"] not in dev_ids]
    dev_rows = [g for g in gold if g["id"] in dev_ids]
    print(f"train rows {len(train_rows)}, dev rows {len(dev_rows)} (random only)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def row_pairs(row):
        q = task[row["question"]]
        doc = render_state(row["state"])
        opts = list(q.options)
        return [(render_query(q, o), doc) for o in opts], opts.index(row["gold"])

    model.train()
    for epoch in range(args.epochs):
        rnd.shuffle(train_rows)
        total, steps = 0.0, 0
        for i in range(0, len(train_rows), args.batch_rows):
            batch = train_rows[i:i + args.batch_rows]
            loss = torch.zeros((), device=device)
            for row in batch:
                pairs, gold_idx = row_pairs(row)
                enc = tok([p[0] for p in pairs], [p[1] for p in pairs], truncation=True,
                          max_length=args.max_length, padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits.reshape(-1)
                loss = loss + torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([gold_idx], device=device))
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss); steps += 1
            if steps % 50 == 0:
                print(f"epoch {epoch} step {steps} loss {total / steps:.4f}", flush=True)
        print(f"epoch {epoch} done, mean loss {total / max(steps, 1):.4f}", flush=True)

    # Dev: raw scores per option -> temperature per question, metrics.
    model.eval()
    by_q: dict[str, list[Labelled]] = defaultdict(list)
    with torch.no_grad():
        for row in dev_rows:
            pairs, gold_idx = row_pairs(row)
            enc = tok([p[0] for p in pairs], [p[1] for p in pairs], truncation=True,
                      max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.reshape(-1).tolist()
            opts = list(task[row["question"]].options)
            by_q[row["question"]].append(Labelled(scores=dict(zip(opts, logits)), gold=row["gold"]))
    temps, metrics = {}, {}
    for qn, items in by_q.items():
        t = fit_temperature(items)
        temps[qn] = t
        metrics[qn] = {"n_dev": len(items), "accuracy": accuracy(items, t), "ece": ece(items, t),
                       "ece_uncalibrated": ece(items), "temperature": t}
        print(qn, metrics[qn])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "head.json").write_text(json.dumps({"base": args.base, "task": Path(args.task).name,
                                                "temperatures": temps, "dev_metrics": metrics,
                                                "max_length": args.max_length},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    if args.export_onnx:
        export_onnx(model, tok, out, args.max_length)
    print("saved", out)


def export_onnx(model, tok, out: Path, max_length: int) -> None:
    import torch

    model = model.cpu().eval()
    enc = tok("q", "d", return_tensors="pt", truncation=True, max_length=max_length)
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    torch.onnx.export(model, tuple(enc[k] for k in names), str(out / "model.onnx"),
                      input_names=names, output_names=["logits"],
                      dynamic_axes={k: {0: "batch", 1: "seq"} for k in names} | {"logits": {0: "batch"}},
                      opset_version=17)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    print("onnx exported; quantize with onnxruntime.quantization.quantize_dynamic for CPU int8")


if __name__ == "__main__":
    main()
