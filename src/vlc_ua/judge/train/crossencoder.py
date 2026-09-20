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
import time
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
    ap.add_argument("--max-length", type=int, default=512,
                    help="512 truncates nothing on the attribution gold: the longest "
                         "(query, fragment) pair measured 422 tokens, and padding to 1024 "
                         "is the single most expensive mistake available here")
    ap.add_argument("--dev-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                    help="mixed precision (default on: without it a T4 has no tensor cores "
                         "and the run does not fit in Kaggle's 12 hours)")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade ~30%% speed for memory if the batch does not fit")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="do not train the embedding matrix. On XLM-R large it is 256M of "
                         "the 559M parameters, so freezing it removes ~3.6 GB of gradients "
                         "and AdamW state — the difference between fitting a T4 and not")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop training after this many hours and still calibrate, write "
                         "head.json and export: a kernel killed at the 12h wall leaves nothing")
    ap.add_argument("--limit-rows", type=int, default=0, help="use only the first N gold rows")
    ap.add_argument("--eval-batch-rows", type=int, default=16)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    task = load_task(args.task)
    gold = load_gold(args.gold)
    rnd = random.Random(args.seed)
    rnd.shuffle(gold)
    if args.limit_rows:
        gold = gold[:args.limit_rows]
    rand_rows = [g for g in gold if g.get("sample", "random") == "random"]
    n_dev = int(len(rand_rows) * args.dev_share)
    dev_ids = {g["id"] for g in rand_rows[:n_dev]}
    train_rows = [g for g in gold if g["id"] not in dev_ids]
    dev_rows = [g for g in gold if g["id"] in dev_ids]
    print(f"train rows {len(train_rows)}, dev rows {len(dev_rows)} (random only)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.freeze_embeddings:
        emb = model.get_input_embeddings()
        for prm in emb.parameters():
            prm.requires_grad_(False)
        if args.grad_checkpointing and hasattr(model, "enable_input_require_grads"):
            # With the embedding frozen, the tensor entering the first checkpointed
            # block carries no grad, and checkpointing then recomputes a graph that
            # leads nowhere: the step runs and changes nothing. This hook puts
            # requires_grad back on the embedding OUTPUT without unfreezing weights.
            model.enable_input_require_grads()
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    n_all = sum(prm.numel() for prm in model.parameters())
    n_trn = sum(prm.numel() for prm in trainable)
    print(f"parameters: {n_all/1e6:.1f}M total, {n_trn/1e6:.1f}M trainable", flush=True)
    if not trainable:
        raise SystemExit("nothing left to train: check --freeze-embeddings")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    use_amp = bool(args.fp16) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device {device}, amp {use_amp}, max_length {args.max_length}, "
          f"batch_rows {args.batch_rows}", flush=True)

    def row_pairs(row):
        q = task[row["question"]]
        doc = render_state(row["state"])
        opts = list(q.options)
        return [(render_query(q, o), doc) for o in opts], opts.index(row["gold"])

    def encode_batch(rows):
        """All pairs of the batch in ONE tokenizer call and ONE forward.

        The first version ran a separate forward per row inside the batch and
        kept every graph alive until the shared backward: the memory cost of a
        real batch with none of its speed.
        """
        queries, docs, spans, golds = [], [], [], []
        for row in rows:
            pairs, gold_idx = row_pairs(row)
            spans.append((len(queries), len(pairs)))
            golds.append(gold_idx)
            queries += [p[0] for p in pairs]
            docs += [p[1] for p in pairs]
        enc = tok(queries, docs, truncation=True, max_length=args.max_length,
                  padding=True, return_tensors="pt").to(device)
        return enc, spans, golds

    def listwise_loss(logits, spans, golds):
        loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        for (start, n), g in zip(spans, golds):
            loss = loss + torch.nn.functional.cross_entropy(
                logits[start:start + n].float().unsqueeze(0),
                torch.tensor([g], device=logits.device))
        return loss / max(len(spans), 1)

    model.train()
    t0 = time.time()
    budget = args.max_hours * 3600 if args.max_hours else None
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        rnd.shuffle(train_rows)
        total, steps = 0.0, 0
        for i in range(0, len(train_rows), args.batch_rows):
            batch = train_rows[i:i + args.batch_rows]
            enc, spans, golds = encode_batch(batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.reshape(-1)
            loss = listwise_loss(logits, spans, golds)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt); scaler.update()
            total += float(loss); steps += 1
            if steps % 50 == 0:
                done = time.time() - t0
                per = done / max(steps, 1)
                left = (len(train_rows) // args.batch_rows - steps) * per
                print(f"epoch {epoch} step {steps} loss {total / steps:.4f} "
                      f"{per:.2f}s/step, ~{left / 3600:.1f}h left in this epoch", flush=True)
            if budget and time.time() - t0 > budget:
                print(f"time budget {args.max_hours}h reached at epoch {epoch} step {steps}; "
                      "stopping training and going to calibration", flush=True)
                stop = True
                break
        print(f"epoch {epoch} done, mean loss {total / max(steps, 1):.4f}, "
              f"{(time.time() - t0) / 3600:.2f}h elapsed", flush=True)

    # Dev: raw scores per option -> temperature per question, metrics.
    model.eval()
    by_q: dict[str, list[Labelled]] = defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(dev_rows), args.eval_batch_rows):
            chunk = dev_rows[i:i + args.eval_batch_rows]
            enc, spans, _ = encode_batch(chunk)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.reshape(-1).float().tolist()
            for row, (start, n) in zip(chunk, spans):
                opts = list(task[row["question"]].options)
                by_q[row["question"]].append(
                    Labelled(scores=dict(zip(opts, logits[start:start + n])), gold=row["gold"]))
    temps, metrics = {}, {}
    for qn, items in by_q.items():
        t = fit_temperature(items)
        temps[qn] = t
        metrics[qn] = {"n_dev": len(items), "accuracy": accuracy(items, t), "ece": ece(items, t),
                       "ece_uncalibrated": ece(items), "temperature": t}
        print(qn, metrics[qn])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "head.json").write_text(json.dumps(
        {"base": args.base, "task": Path(args.task).name, "temperatures": temps,
         "dev_metrics": metrics, "max_length": args.max_length,
         "train": {"rows": len(train_rows), "dev_rows": len(dev_rows), "epochs": args.epochs,
                   "lr": args.lr, "batch_rows": args.batch_rows, "fp16": bool(args.fp16),
                   "frozen_embeddings": bool(args.freeze_embeddings),
                   "grad_checkpointing": bool(args.grad_checkpointing),
                   "hours": round((time.time() - t0) / 3600, 2),
                   "stopped_on_budget": bool(stop)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if args.export_onnx:
        try:
            export_onnx(model, tok, out, args.max_length)
        except Exception as exc:   # noqa: BLE001 - the weights are already on disk
            # Never lose a finished training run to the exporter: torch 2.6+ routes
            # torch.onnx.export through dynamo, which needs the separate onnxscript
            # package, and a kernel without it would otherwise end with no head at all.
            print(f"ONNX export failed ({type(exc).__name__}: {exc}); weights and head.json "
                  "are saved, export later with: pip install onnxscript && python -m "
                  "vlc_ua.judge.train.crossencoder --export-onnx-only <dir>", flush=True)
    print("saved", out)


def export_onnx(model, tok, out: Path, max_length: int) -> None:
    import torch

    model = model.cpu().eval()
    enc = tok("q", "d", return_tensors="pt", truncation=True, max_length=max_length)
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    kwargs = dict(input_names=names, output_names=["logits"],
                  dynamic_axes={k: {0: "batch", 1: "seq"} for k in names} | {"logits": {0: "batch"}},
                  opset_version=17)
    try:
        # torch >= 2.6 defaults to the dynamo exporter, which needs onnxscript;
        # the TorchScript path is still there and is enough for this graph.
        torch.onnx.export(model, tuple(enc[k] for k in names), str(out / "model.onnx"),
                          dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, tuple(enc[k] for k in names), str(out / "model.onnx"), **kwargs)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    print("onnx exported; quantize with onnxruntime.quantization.quantize_dynamic for CPU int8")


if __name__ == "__main__":
    main()
