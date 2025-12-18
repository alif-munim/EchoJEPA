#!/usr/bin/env python3
import argparse, csv, random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

def extract_study_id(path: str) -> Optional[str]:
    key = "h4h_ef_frames/"
    i = path.find(key)
    if i == -1:
        return None
    rest = path[i + len(key):]
    study = rest.split("/", 1)[0]
    return study if study else None

def load_syngo_map(syngo_path: Path) -> Dict[str, str]:
    m = {}
    with syngo_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        need = {"DeidentifiedStudyID", "PATIENT_ID"}
        if not need.issubset(fields):
            raise SystemExit(f"{syngo_path}: missing required columns {need}. Found: {r.fieldnames}")
        for row in r:
            sid = (row.get("DeidentifiedStudyID") or "").strip().strip('"')
            pid = (row.get("PATIENT_ID") or "").strip()
            if sid and pid:
                m[sid] = pid
    return m

def pct(x: int, n: int) -> float:
    return (100.0 * x / n) if n else 0.0

def objective_sse(split_label_counts: Dict[str, Counter],
                  target_label_counts: Dict[str, Dict[str, float]],
                  split_sample_counts: Dict[str, int],
                  target_samples: Dict[str, float]) -> float:
    err = 0.0
    for sp, tgt in target_label_counts.items():
        act = split_label_counts[sp]
        for lab, t in tgt.items():
            d = float(act.get(lab, 0)) - float(t)
            err += d * d
    for sp in ("train","val","test"):
        d = float(split_sample_counts[sp]) - float(target_samples[sp])
        err += 0.01 * d * d
    return err

def meets_minima(split_label_counts: Dict[str, Counter],
                 labels: List[str],
                 min_val: int,
                 min_test: int) -> Tuple[bool, Dict[str, List[Tuple[str,int]]]]:
    deficits = {"val": [], "test": []}
    if min_val > 0:
        for lab in labels:
            c = split_label_counts["val"].get(lab, 0)
            if c < min_val:
                deficits["val"].append((lab, c))
    if min_test > 0:
        for lab in labels:
            c = split_label_counts["test"].get(lab, 0)
            if c < min_test:
                deficits["test"].append((lab, c))
    ok = (not deficits["val"]) and (not deficits["test"])
    return ok, deficits

def feasibility_by_label(all_labels: Counter, min_val: int, min_test: int) -> List[str]:
    msgs = []
    need = min_val + min_test
    if need <= 0:
        return msgs
    for lab, tot in sorted(all_labels.items(), key=lambda x: x[1]):
        if tot < need:
            msgs.append(f"{lab}: total={tot} < min_val+min_test={need} (impossible)")
    return msgs

def try_assign(patients: List[str],
               patient_total_samples: Counter,
               patient_label_counts: Dict[str, Counter],
               all_labels: Counter,
               ratios: Dict[str,float],
               seed: int,
               attempt: int):
    rng = random.Random((seed * 1_000_003) + attempt)

    total_samples = sum(all_labels.values())
    target_samples = {sp: total_samples * ratios[sp] for sp in ("train","val","test")}
    target_label_counts = {
        sp: {lab: all_labels[lab] * ratios[sp] for lab in all_labels}
        for sp in ("train","val","test")
    }

    patients_sorted = sorted(
        patients,
        key=lambda p: (patient_total_samples[p], rng.random()),
        reverse=True
    )

    split_patients = {"train": set(), "val": set(), "test": set()}
    split_label_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    split_sample_counts = {"train": 0, "val": 0, "test": 0}

    splits = ["train","val","test"]

    for pid in patients_sorted:
        plc = patient_label_counts[pid]
        ps = patient_total_samples[pid]

        best_sp, best_obj = None, None
        rng.shuffle(splits)
        for sp in splits:
            split_label_counts[sp].update(plc)
            split_sample_counts[sp] += ps

            obj = objective_sse(split_label_counts, target_label_counts, split_sample_counts, target_samples)

            split_label_counts[sp].subtract(plc); split_label_counts[sp] += Counter()
            split_sample_counts[sp] -= ps

            if best_obj is None or obj < best_obj:
                best_obj, best_sp = obj, sp

        split_patients[best_sp].add(pid)
        split_label_counts[best_sp].update(plc)
        split_sample_counts[best_sp] += ps

    patient_to_split = {}
    for sp in ("train","val","test"):
        for pid in split_patients[sp]:
            patient_to_split[pid] = sp

    obj = objective_sse(split_label_counts, target_label_counts, split_sample_counts, target_samples)
    return patient_to_split, split_patients, split_label_counts, split_sample_counts, obj

def apply_move(pid: str, src: str, dst: str,
               split_patients: Dict[str, Set[str]],
               split_label_counts: Dict[str, Counter],
               split_sample_counts: Dict[str, int],
               patient_label_counts: Dict[str, Counter],
               patient_total_samples: Counter):
    split_patients[src].remove(pid)
    split_patients[dst].add(pid)
    plc = patient_label_counts[pid]
    ps = patient_total_samples[pid]
    split_label_counts[src].subtract(plc); split_label_counts[src] += Counter()
    split_label_counts[dst].update(plc)
    split_sample_counts[src] -= ps
    split_sample_counts[dst] += ps

def repair_minima(split_patients: Dict[str, Set[str]],
                  split_label_counts: Dict[str, Counter],
                  split_sample_counts: Dict[str, int],
                  patient_label_counts: Dict[str, Counter],
                  patient_total_samples: Counter,
                  labels: List[str],
                  min_val: int,
                  min_test: int,
                  ratios: Dict[str,float],
                  seed: int,
                  max_steps: int = 2000) -> Tuple[bool, Dict[str, List[Tuple[str,int]]]]:
    """
    Greedy repair: try to fix val/test deficits by moving a patient from train -> val/test
    who has the deficient label(s). Keeps size drift small by preferring patients whose
    total samples help keep split near target.
    """
    rng = random.Random(seed + 99991)

    total_samples = sum(split_sample_counts.values())
    target_samples = {sp: total_samples * ratios[sp] for sp in ("train","val","test")}

    def deficit_list():
        ok, deficits = meets_minima(split_label_counts, labels, min_val, min_test)
        return ok, deficits

    ok, deficits = deficit_list()
    if ok:
        return True, deficits

    # build label -> candidate patients in train
    train = split_patients["train"]

    for _ in range(max_steps):
        ok, deficits = deficit_list()
        if ok:
            return True, deficits

        # prioritize worst deficit (smallest count relative to min)
        need_items = []
        for lab, c in deficits["val"]:
            need_items.append(("val", lab, min_val - c))
        for lab, c in deficits["test"]:
            need_items.append(("test", lab, min_test - c))
        need_items.sort(key=lambda x: (-x[2], x[0], x[1]))
        if not need_items:
            return True, deficits

        dst, lab, _gap = need_items[0]

        # candidates: patients in train that have this label
        candidates = [pid for pid in train if patient_label_counts[pid].get(lab, 0) > 0]
        if not candidates:
            # cannot fix this deficit via moving from train
            return False, deficits

        # score candidates by: (how much they help) and (how well they keep sample counts near target)
        best_pid = None
        best_score = None
        for pid in candidates:
            add = patient_label_counts[pid][lab]
            # prefer larger add
            # prefer moving if dst is under target_samples and train is over target
            dst_under = target_samples[dst] - split_sample_counts[dst]
            tr_over = split_sample_counts["train"] - target_samples["train"]
            size_bonus = 0.0
            if dst_under > 0: size_bonus += 0.001 * min(dst_under, patient_total_samples[pid])
            if tr_over > 0: size_bonus += 0.001 * min(tr_over, patient_total_samples[pid])

            # penalize huge moves that blow up dst
            overflow = max(0.0, (split_sample_counts[dst] + patient_total_samples[pid]) - target_samples[dst])
            penalty = 0.0005 * overflow

            score = (add * 1.0) + size_bonus - penalty + (rng.random() * 1e-6)
            if best_score is None or score > best_score:
                best_score = score
                best_pid = pid

        # move best_pid train -> dst
        apply_move(best_pid, "train", dst, split_patients, split_label_counts, split_sample_counts,
                   patient_label_counts, patient_total_samples)

    ok, deficits = deficit_list()
    return ok, deficits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--syngo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--min-val", type=int, default=0)
    ap.add_argument("--min-test", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=300)
    ap.add_argument("--repair-steps", type=int, default=2000)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("train+val+test must sum to 1.0")

    ratios = {"train": args.train, "val": args.val, "test": args.test}

    study_to_patient = load_syngo_map(args.syngo)

    rows = []
    missing_study = 0
    missing_patient = 0

    with args.labels.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if "filename" not in fields or "label" not in fields:
            raise SystemExit(f"{args.labels}: must contain filename,label. Found: {r.fieldnames}")

        for row in r:
            fn = (row.get("filename") or "").strip()
            lab = (row.get("label") or "").strip()

            sid = extract_study_id(fn) if fn else None
            if not sid:
                missing_study += 1
                if args.strict:
                    raise SystemExit(f"Could not extract study_id from filename: {fn}")
                pid = ""
            else:
                pid = study_to_patient.get(sid, "")
                if not pid:
                    missing_patient += 1
                    if args.strict:
                        raise SystemExit(f"study_id {sid} not found in {args.syngo}")
            rows.append((row, sid or "", pid, lab))

    patient_label_counts = defaultdict(Counter)
    patient_total_samples = Counter()
    all_labels = Counter()
    valid_patients = set()

    for _orig, _sid, pid, lab in rows:
        if not pid:
            continue
        valid_patients.add(pid)
        patient_label_counts[pid][lab] += 1
        patient_total_samples[pid] += 1
        all_labels[lab] += 1

    patients = list(valid_patients)
    labels_sorted = sorted(all_labels.keys(), key=lambda x: (-all_labels[x], x))

    feas_msgs = feasibility_by_label(all_labels, args.min_val, args.min_test)

    best_ok = None
    best_any = None

    for attempt in range(args.max_tries):
        st = try_assign(patients, patient_total_samples, patient_label_counts, all_labels, ratios, args.seed, attempt)
        patient_to_split, split_patients, split_label_counts, split_sample_counts, obj = st
        ok, deficits = meets_minima(split_label_counts, labels_sorted, args.min_val, args.min_test)

        # try repair if not ok
        if not ok and (args.min_val > 0 or args.min_test > 0):
            ok2, deficits2 = repair_minima(
                split_patients, split_label_counts, split_sample_counts,
                patient_label_counts, patient_total_samples,
                labels_sorted, args.min_val, args.min_test,
                ratios, seed=(args.seed * 1_000_003 + attempt),
                max_steps=args.repair_steps
            )
            ok, deficits = ok2, deficits2
            # rebuild patient_to_split after repair
            patient_to_split = {}
            for sp in ("train","val","test"):
                for pid in split_patients[sp]:
                    patient_to_split[pid] = sp

        if best_any is None or obj < best_any[4]:
            best_any = (patient_to_split, split_patients, split_label_counts, split_sample_counts, obj, attempt, deficits, ok)

        if ok:
            if best_ok is None or obj < best_ok[4]:
                best_ok = (patient_to_split, split_patients, split_label_counts, split_sample_counts, obj, attempt, deficits, ok)

    if best_ok is not None:
        patient_to_split, split_patients, split_label_counts, split_sample_counts, obj, attempt, deficits, ok = best_ok
        satisfied_minima = True
    else:
        patient_to_split, split_patients, split_label_counts, split_sample_counts, obj, attempt, deficits, ok = best_any
        satisfied_minima = False

    # sanity overlaps
    train_p, val_p, test_p = split_patients["train"], split_patients["val"], split_patients["test"]
    overlap_tv = len(train_p & val_p)
    overlap_tt = len(train_p & test_p)
    overlap_vt = len(val_p & test_p)

    # write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    orig_fields = list(rows[0][0].keys())
    out_fields = orig_fields + ["study_id","patient_id","split"]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for orig, sid, pid, _lab in rows:
            sp = patient_to_split.get(pid, "") if pid else ""
            out = dict(orig)
            out["study_id"] = sid
            out["patient_id"] = pid
            out["split"] = sp
            w.writerow(out)

    # patient lists
    for sp in ("train","val","test"):
        pfile = args.out.with_name(f"patients_{sp}.txt")
        with pfile.open("w", encoding="utf-8") as f:
            for pid in sorted(split_patients[sp]):
                f.write(pid + "\n")

    # dist csv
    dist_csv = args.out.with_name("class_distribution_by_split.csv")
    with dist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","total","train","val","test"])
        for lab in labels_sorted:
            w.writerow([
                lab, all_labels[lab],
                split_label_counts["train"].get(lab,0),
                split_label_counts["val"].get(lab,0),
                split_label_counts["test"].get(lab,0)
            ])

    # summary
    summary = args.out.with_name("split_summary.txt")
    total_rows = len(rows)
    total_valid_rows = sum(all_labels.values())
    total_patients = len(patients)

    with summary.open("w", encoding="utf-8") as f:
        f.write("=== Patient-disjoint split summary ===\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"chosen_attempt={attempt}\n")
        f.write(f"ratios train/val/test={args.train}/{args.val}/{args.test}\n")
        f.write(f"min_per_class val={args.min_val}, test={args.min_test}\n")
        f.write(f"satisfied_minima={satisfied_minima}\n")
        if feas_msgs:
            f.write("\n--- Feasibility warnings (totals too small for requested minima) ---\n")
            for m in feas_msgs:
                f.write(m + "\n")

        f.write("\n--- Data mapping ---\n")
        f.write(f"total_rows_in_labels={total_rows}\n")
        f.write(f"rows_with_patient_id={total_valid_rows}\n")
        f.write(f"rows_missing_study_id={missing_study}\n")
        f.write(f"rows_missing_patient_id={missing_patient}\n")

        f.write("\n--- Patient counts (disjointness sanity) ---\n")
        f.write(f"unique_patients_total={total_patients}\n")
        f.write(f"train_patients={len(train_p)}, val_patients={len(val_p)}, test_patients={len(test_p)}\n")
        f.write(f"overlap(train,val)={overlap_tv}\n")
        f.write(f"overlap(train,test)={overlap_tt}\n")
        f.write(f"overlap(val,test)={overlap_vt}\n")

        f.write("\n--- Sample counts ---\n")
        f.write(f"train_samples={split_sample_counts['train']} ({pct(split_sample_counts['train'], total_valid_rows):.2f}%)\n")
        f.write(f"val_samples={split_sample_counts['val']} ({pct(split_sample_counts['val'], total_valid_rows):.2f}%)\n")
        f.write(f"test_samples={split_sample_counts['test']} ({pct(split_sample_counts['test'], total_valid_rows):.2f}%)\n")

        f.write("\n--- Class distribution by split (counts) ---\n")
        f.write("label, total, train, val, test\n")
        for lab in labels_sorted:
            f.write(f"{lab},{all_labels[lab]},{split_label_counts['train'].get(lab,0)},{split_label_counts['val'].get(lab,0)},{split_label_counts['test'].get(lab,0)}\n")

        ok2, deficits2 = meets_minima(split_label_counts, labels_sorted, args.min_val, args.min_test)
        f.write("\n--- Minima check details ---\n")
        if ok2:
            f.write("All minima satisfied.\n")
        else:
            f.write("Minima NOT satisfied.\n")
            if args.min_val > 0 and deficits2["val"]:
                f.write("VAL deficits (label,count):\n")
                for lab, c in deficits2["val"]:
                    f.write(f"  {lab}: {c}\n")
            if args.min_test > 0 and deficits2["test"]:
                f.write("TEST deficits (label,count):\n")
                for lab, c in deficits2["test"]:
                    f.write(f"  {lab}: {c}\n")

    print(f"Wrote: {args.out}")
    print(f"Wrote: {summary}")
    print(f"Wrote: {dist_csv}")
    print(f"Patients: train={len(train_p)}, val={len(val_p)}, test={len(test_p)} (total={total_patients})")
    print(f"Samples:  train={split_sample_counts['train']}, val={split_sample_counts['val']}, test={split_sample_counts['test']} (rows_with_patient_id={total_valid_rows})")
    if feas_msgs:
        print("WARNING: some requested minima are impossible for some labels (see split_summary.txt).")
    if not satisfied_minima:
        print("WARNING: minima still not satisfied (see split_summary.txt).")

if __name__ == "__main__":
    main()