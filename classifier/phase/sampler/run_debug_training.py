"""CPU sanity harness for the multiview launcher.

This is NOT a production launcher. It stubs decord with the pydicom
reader (same trick as ``check_anchor_loading.py``) so
``VideoGroupDataset`` can decode from local .dcm files, then runs
``app.vjepa_multiview.train.main`` with a debug config. The harness
installs a monkey-patch that, after each ``builder.refresh_epoch``,
restricts the pair DataFrame to pairs whose DICOMs are on local disk
and rewrites the ``view_0``/``view_1`` columns to absolute local paths.

Use for:
  * verifying the full training loop end-to-end on this CPU dev box
  * smoke-testing the launcher after a loss / schema change

Do NOT use for:
  * anything that requires real GPU throughput numbers
  * any run where label fidelity matters (we subsample heavily)
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
VJEPA_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))


def _install_decord_stub():
    """Replicate the stub from check_anchor_loading.py."""
    from check_anchor_loading import _DicomVR
    mod = types.ModuleType("decord")

    class _VideoReader:
        def __new__(cls, uri, *args, **kwargs):
            return _DicomVR(Path(uri))

    mod.VideoReader = _VideoReader
    mod.cpu = lambda *a, **kw: None
    sys.modules["decord"] = mod


def _patch_builder_to_local_dicoms(dicom_dir: Path):
    """Monkey-patch PhaseMatchedEpochBuilder.refresh_epoch to additionally
    filter the pair DataFrame to only local DICOMs and rewrite paths.
    """
    from phase_matched_pair_dataset import PhaseMatchedEpochBuilder
    from phase_matched_pair_dataset import _records_to_anchor_table, _records_to_pair_dataframe
    from phase_matched_sampler import PhaseMatchedStudySampler

    avail = {p.stem for p in dicom_dir.glob("*.dcm")}
    print(f"[debug harness] on-disk DICOMs available: {len(avail)}")

    original = PhaseMatchedEpochBuilder.refresh_epoch

    def _patched(self, epoch):
        # Run the normal refresh; records are stored in self.sampler.
        self.sampler.set_epoch(epoch)
        records = self.sampler.build_records()
        # Filter + rebuild to on-disk studies. If the random draw has zero
        # overlap with local DICOMs, synthesize pairs from on-disk multi-
        # clip studies via _draw_pair.
        filtered = []
        for r in records:
            if (r.clip_a.dicom_id in avail) and (r.clip_b.dicom_id in avail):
                filtered.append(r)
        if not filtered:
            sdf = self.sampler._df
            on_disk_src = sdf[sdf.dicom_id.astype(str).isin(avail)]
            multi = on_disk_src.groupby("study_id").size()
            multi = multi[multi >= 2].index.tolist()
            rng = np.random.default_rng(epoch)
            for sid in multi:
                sub = on_disk_src[on_disk_src.study_id == sid]
                row_idxs = sub.index.tolist()
                i, j = rng.choice(len(row_idxs), size=2, replace=False)
                self.sampler.study_to_rows[str(sid)] = [int(row_idxs[int(i)]), int(row_idxs[int(j)])]
                rec = self.sampler._draw_pair(str(sid), rng)
                if rec is not None:
                    filtered.append(rec)
        print(f"[debug harness] epoch {epoch}: {len(filtered)} on-disk pair records")
        # Debug harness uses local .dcm paths (rewritten below); pass
        # video_uri_mode='dicom' so the builder doesn't assert on .mp4.
        pair_df = _records_to_pair_dataframe(
            filtered, self.sampler._df, video_uri_mode="dicom"
        )
        pair_df["view_0"] = pair_df.clip_a_dicom_id.map(lambda d: str(dicom_dir / f"{d}.dcm"))
        pair_df["view_1"] = pair_df.clip_b_dicom_id.map(lambda d: str(dicom_dir / f"{d}.dcm"))
        anchors = _records_to_anchor_table(filtered)
        self.dataset.set_pair_dataframe(pair_df, anchors_by_index=anchors)
        self._last_pair_df = pair_df
        self._last_anchors = anchors
        return len(pair_df)

    PhaseMatchedEpochBuilder.refresh_epoch = _patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dicom-dir", type=Path,
                    default=HERE.parents[1] / "dicoms")
    args = ap.parse_args()

    _install_decord_stub()
    _patch_builder_to_local_dicoms(args.dicom_dir)

    cfg = yaml.safe_load(args.config.read_text())
    from app.scaffold import main as scaffold_main
    app_name = cfg.get("app", "vjepa_multiview")
    scaffold_main(app=app_name, args=cfg)


if __name__ == "__main__":
    main()
