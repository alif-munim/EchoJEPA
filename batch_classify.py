#!/usr/bin/env python3
# ============================================================================
#  batch_classify.py · 2025‑07‑23  (robust, local/remote friendly)
#  • multi‑GPU (torchrun) · FP16
#  • NO silent drops on OOM/IndexMath: recursive split processes ALL samples
#  • strict checkpoint check + per‑rank skip logs
#  • tqdm bars per rank
#  • OUT_DIR env var for non‑SageMaker runs
# ============================================================================

import argparse, csv, gzip, io, itertools, logging, os, re, time, hashlib
from collections import Counter
from math import tanh
from typing import Iterable, Tuple, Optional

import boto3, s3fs
from botocore.config import Config as BotoCfg
from botocore.exceptions import ClientError
from pathlib import Path
import cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────────────────────────── constants ──────────────────────────────
VIEW = [
    "a2c","a3c","a4c","a5c","plax","tee","exclude",
    "psax-av","psax-mv","psax-ap","psax-pm"
]

PNG_ROW_RE = re.compile(
    r"""^s3://[^/]+/
        (?P<key>
            results/echo-study(?:-[12])?/      # echo-study/, echo-study-1/2/
            .+?/unmasked/png/
            .+\.png$
        )""",
    re.IGNORECASE | re.VERBOSE,
)

TF = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

ctr: Counter = Counter()  # global counters
def record_skip(kind: str, msg: str): ...  # rebound in main()

# ───────────────────────────── helpers ────────────────────────────────
def quality_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = gray.mean() / 255.0
    return tanh(0.004 * sharp) * bright

def png_to_mp4(png_key: str) -> str:
    return png_key.replace("/unmasked/png/", "/")[:-4] + ".mp4"

def chunks(it: Iterable, n: int):
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch

def retry_open(fs: s3fs.S3FileSystem, path: str, tries: int = 3):
    delay = 1.0
    for attempt in range(tries):
        try:
            return fs.open(path, "rb")
        except ClientError as e:
            if attempt == tries - 1:
                raise
            code = e.response["Error"].get("Code", "")
            if code in ("500", "503", "InternalError"):
                time.sleep(delay); delay *= 1.5
            else:
                raise

# ───────────────────────────── model ──────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b2(weights=None)
        base.classifier = nn.Identity()
        self.b = base
        f = 1408
        self.vb = nn.Linear(f, 2)
        self.vo = nn.Linear(f, 7)
        self.vs = nn.Linear(f, 4)

    def forward(self, x):
        f = self.b(x)
        pb, po, ps = (F.softmax(h(f), 1) for h in (self.vb, self.vo, self.vs))
        out = x.new_zeros(x.size(0), 11)
        out[:, :7] = pb[:, :1] * po
        out[:, 7:] = pb[:, 1:] * ps
        return out

# ─────────────────────── manifest utilities ──────────────────────────
def open_body(fs: s3fs.S3FileSystem, uri: str):
    bucket, key = uri[5:].split("/", 1)
    rank = os.getenv("LOCAL_RANK", "0")
    uri_hash = hashlib.md5(uri.encode()).hexdigest()[:8]
    local = Path(f"/tmp/manifest_rank{rank}_{uri_hash}.gz")
    if not local.exists():
        logging.info("rank%s downloading manifest → %s", rank, local)
        with fs.open(f"s3://{bucket}/{key}", "rb") as src, local.open("wb") as dst:
            for chunk in iter(lambda: src.read(8 << 20), b""):
                dst.write(chunk)
    fh = local.open("rb")
    return gzip.GzipFile(fileobj=fh) if key.endswith(".gz") else fh

def iter_manifest(fs: s3fs.S3FileSystem, uri: str, world: int, rank: int, limit: Optional[int]):
    seen = 0
    for idx, raw in enumerate(open_body(fs, uri)):
        if idx % world != rank:
            continue
        if limit is not None and seen >= limit:
            break
        line = raw.strip().decode()
        m = PNG_ROW_RE.match(line)
        if m:
            yield m.group("key")
            seen += 1
        else:
            ctr["regex"] += 1
            record_skip("REGEX", line)

def count_samples(fs: s3fs.S3FileSystem, uri: str, world: int, rank: int, limit: Optional[int]) -> int:
    return sum(1 for _ in iter_manifest(fs, uri, world, rank, limit))

# ───────────────────────────── main ──────────────────────────────────
def main(a):
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s │ %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, log_level, logging.INFO),
        force=True,
    )

    # distributed
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    dev_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(dev_id)
    device = torch.device(f"cuda:{dev_id}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "256"))
    boto_cfg = BotoCfg(max_pool_connections=MAX_WORKERS)
    _ = boto3.client("s3", config=boto_cfg)  # kept if needed later
    fs = s3fs.S3FileSystem(
        anon=False,
        default_block_size=8 << 20,
        default_fill_cache=False,
        config_kwargs={"max_pool_connections": MAX_WORKERS},
    )

    out_dir = os.getenv("OUT_DIR", "./sm_output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    skip_path = f"{out_dir}/skip_rank{rank}.txt.gz"
    skip_fh = gzip.open(skip_path, "wt")

    def _record(kind: str, msg: str):
        skip_fh.write(f"{kind}\t{msg}\n")

    globals()["record_skip"] = _record

    # dry run
    if a.dry_run:
        for _ in iter_manifest(fs, a.manifest_s3, world, rank, a.limit):
            pass
        skip_fh.close()
        logging.info("DRY-RUN finished – regex %d", ctr["regex"])
        return

    total_imgs = count_samples(fs, a.manifest_s3, world, rank, a.limit)
    logging.info("rank%d will process ~%s imgs", rank, f"{total_imgs:,d}" if total_imgs else "?")

    bar = tqdm(
        total=total_imgs or None,
        desc=f"rank {rank}",
        unit="img",
        position=rank,
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=1.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]",
    )

    # model
    net = Net().to(device).eval()
    with fs.open(a.model_s3, "rb") as f:
        state = torch.load(io.BytesIO(f.read()), map_location="cpu")
    ck = net.load_state_dict(state, strict=False)
    if ck.missing_keys or ck.unexpected_keys:
        logging.warning("‼️ checkpoint mismatch – %d missing, %d unexpected",
                        len(ck.missing_keys), len(ck.unexpected_keys))
    else:
        logging.info("✅ checkpoint keys match perfectly")
    net.half()

    csv_path = f"{out_dir}/preds_rank{rank}.csv"
    header = ["png_uri", "mp4_uri", "pred_view", "quality", "salience"] + [f"p_{v}" for v in VIEW]

    key_iter = iter_manifest(fs, a.manifest_s3, world, rank, a.limit)
    processed = 0
    t0 = time.time()

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def load_one(k: str) -> Tuple[str, torch.Tensor, float, bool]:
        try:
            with retry_open(fs, f"{a.bucket}/{k}") as f:
                arr = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            q = quality_score(img)
            ten = TF(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).half()
            return k, ten, q, False
        except ClientError as ce:
            ctr["open"] += 1
            record_skip("OPEN", f"{k}\t{ce}")
        except Exception as exc:
            ctr["decode"] += 1
            record_skip("DECODE", f"{k}\t{exc}")
            logging.debug("DECODE‑fail %s — %s", k, exc)
        return k, None, None, True

    def safe_infer(batch: torch.Tensor) -> np.ndarray:
        """Process whole batch; on OOM/IndexMath split and retry, never drop."""
        outs = []
        stack = [batch]
        while stack:
            cur = stack.pop()
            try:
                with torch.cuda.amp.autocast(), torch.no_grad():
                    outs.append(net(cur).cpu())
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                msg = str(exc).lower()
                if "indexmath" not in msg and not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                if cur.size(0) == 1:
                    ctr["oom1"] += 1
                    record_skip("OOM1", "single_sample_after_split")
                    continue
                mid = cur.size(0) // 2
                torch.cuda.empty_cache()
                stack.append(cur[mid:])
                stack.append(cur[:mid])
        return torch.cat(outs, 0).numpy()

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)

        for keys in chunks(key_iter, a.batch_size):
            futs = [pool.submit(load_one, k) for k in keys]
            tens, qs, oks = [], [], []
            for fut in as_completed(futs):
                k, ten, q, bad = fut.result()
                if bad:
                    continue
                tens.append(ten)
                qs.append(q)
                oks.append(k)

            if not oks:
                continue

            probs = safe_infer(torch.stack(tens).to(device, non_blocking=True))
            for k, q, p in zip(oks, qs, probs):
                sal = 0.7 * p.max() + 0.3 * q
                w.writerow([
                    f"s3://{a.bucket}/{k}",
                    f"s3://{a.bucket}/{png_to_mp4(k)}",
                    VIEW[int(p.argmax())],
                    round(q, 6),
                    round(sal, 6),
                    *map(lambda x: round(float(x), 6), p),
                ])

            processed += len(oks)
            bar.update(len(oks))

    bar.close()
    skip_fh.close()
    elapsed = time.time() - t0
    logging.info(
        "✓ finished — %.1f min | %s OK | drops %s",
        elapsed / 60,
        f"{processed:,d}",
        ", ".join(f"{k}:{v}" for k, v in ctr.items()) or "none",
    )

# ────────────────────────── CLI wiring ────────────────────────────────
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--bucket", required=True)
    P.add_argument("--manifest_s3", required=True)
    P.add_argument("--model_s3", required=True)
    P.add_argument("--batch_size", type=int, default=2048)
    P.add_argument("--limit", type=int, default=None, help="debug‑only: cap on #pngs per rank")
    P.add_argument("--dry_run", action="store_true", help="only parse manifest + regex stats")
    main(P.parse_args())
