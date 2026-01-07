#!/usr/bin/env bash
# =============================================================================
# dicom_batch_convert.sh  — SLURM-friendly DICOM → MP4/PNG converter
# - SINGLE_STUDY mode (env SINGLE_STUDY) or manifest-driven worker pool
# =============================================================================
set -euo pipefail
export SHELL=/bin/bash

# ---------- IDs / tags ----------
NODE_ID="${1:-0}"
HOSTTAG="$(hostname -s 2>/dev/null || hostname || echo node$NODE_ID)"
START_TS="$(date '+%Y-%m-%d %H:%M:%S')"

# ---------- Paths (overridable via env or your .slurm) ----------
BASE_INPUT_DIR="${BASE_INPUT_DIR:-/gpfs/data/whitney-lab/echo-FM/March_2022_complete/DICOM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/data/whitney-lab/echo-FM/March_2022_complete/CONVERTED}"
SHARED_DIR="${SHARED_DIR:-$OUTPUT_ROOT}"
MANIFEST="${MANIFEST:-$SHARED_DIR/manifest.txt}"
CURSOR="${CURSOR:-$SHARED_DIR/cursor.txt}"
mkdir -p "$OUTPUT_ROOT"

# ---------- Perf tuning ----------
PAR_STUDIES="${PAR_STUDIES:-1}"                                # manifest mode only
JOBS_PER_STUDY="${JOBS_PER_STUDY:-${SLURM_CPUS_PER_TASK:-28}}" # GNU parallel slots
PYTHON_THREADS="${PYTHON_THREADS:-2}"
OPENJPEG_THREADS="${OPENJPEG_THREADS:-2}"

# ---------- Color space handling ----------
FORCE_GRAY="${FORCE_GRAY:-1}"   # default ON: write grayscale (3 identical channels)
export FORCE_GRAY

FF_ENC="${FF_ENC:-libx264}"
QFLAGS="${QFLAGS:--preset fast -crf 22 -threads 1}"
GDCMCONV="${GDCMCONV:-$(command -v gdcmconv || true)}"         # optional

# Export variables needed by parallel-invoked functions
export OUTPUT_ROOT BASE_INPUT_DIR FF_ENC HOSTTAG
export QFLAGS  # Must export this for create_white_sample in parallel

# ---------- FFmpeg discovery ----------
FFMPEG_BIN="${FFMPEG_BIN:-$(command -v ffmpeg || true)}"
if [[ -z "$FFMPEG_BIN" ]]; then
  FFMPEG_BIN="$(python3 - <<'PY'
try:
    import imageio_ffmpeg as i
    print(i.get_ffmpeg_exe(), end="")
except Exception:
    pass
PY
)"
fi
if [[ -z "$FFMPEG_BIN" ]]; then
  echo "ERROR: FFmpeg not found. Install 'imageio-ffmpeg' in your venv or set FFMPEG_BIN." >&2
  exit 1
fi
# Prefer x264; otherwise fall back to mpeg4 (works everywhere)
if ! "$FFMPEG_BIN" -hide_banner -codecs 2>/dev/null | grep -q 'libx264'; then
  echo "[WARN] libx264 not found; using mpeg4." >&2
  FF_ENC="mpeg4"
  QFLAGS="-q:v 4 -threads 1"
fi

# Export FFmpeg binary for parallel-invoked functions
export FFMPEG_BIN

# ---------- Logging ----------
NODE_LOG_DIR="${OUTPUT_ROOT}/logs/${HOSTTAG}"
mkdir -p "$NODE_LOG_DIR"
ERRORS="$NODE_LOG_DIR/errors.log"
SKIPPED="$NODE_LOG_DIR/skip.log"
PROCESSING_LOG="$NODE_LOG_DIR/processing.log"
: >"$ERRORS" ; : >"$SKIPPED" ; : >"$PROCESSING_LOG"
echo "[$HOSTTAG] Started at $START_TS (NODE_ID=$NODE_ID)" | tee -a "$PROCESSING_LOG"

# Export logging paths for parallel-invoked functions
export ERRORS SKIPPED PROCESSING_LOG NODE_LOG_DIR

# ---------- Temp space ----------
# Use GPFS for temp files (with aggressive sync to handle metadata caching)
# Local storage (/tmp, /scratch) may have permissions/quota issues on some clusters
TEMP_DIR="${OUTPUT_ROOT}/tmp_${HOSTTAG}_$$"
mkdir -p "$TEMP_DIR"
cleanup(){ rm -rf "$TEMP_DIR"; }
trap cleanup EXIT INT TERM

echo "[$HOSTTAG] Using temp directory: $TEMP_DIR" | tee -a "$PROCESSING_LOG"

# ---------- Utils ----------
hash_path() {
  local s="$1" h=""
  if command -v md5sum >/dev/null 2>&1; then
    h="$(printf '%s' "$s" | md5sum | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    h="$(printf '%s' "$s" | shasum | awk '{print $1}')"
  else
    h="$(printf '%s' "$s" | cksum | awk '{print $1}')"
  fi
  printf '%s' "$h"
}

# ---------- Mask filter ----------
# Rect-lite approach: balanced coverage without nibbling diagnostic content
# - Top strip: 50/480 (~10.4%) to catch gray bar
# - Top-left: 220/640 at 20% height
# - Top-right cap: 230/640×24% (narrower+shallower to avoid fan)
# - Tight dot patch: 430/640 to 490/640, 78/480 to 138/480 (9%W × 12.5%H, nudged up/right)
MASK_FILTER="drawbox=x=0:y=0:w=iw:h=ih*(50/480):color=black:t=fill,
      drawbox=x=0:y=ih-(ih*(90/480)):w=iw:h=ih*(120/480):color=black:t=fill,
      drawbox=x=0:y=0:w=iw*(60/640):h=ih:color=black:t=fill,
      drawbox=x=0:y=0:w=iw*(90/640):h=ih*0.55:color=black:t=fill,
      drawbox=x=0:y=0:w=iw*(77/640):h=ih*0.62:color=black:t=fill,
      drawbox=x=0:y=0:w=iw*(130/640):h=ih*0.30:color=black:t=fill,
      drawbox=x=0:y=0:w=iw*(150/640):h=ih*0.26:color=black:t=fill,
      drawbox=x=0:y=0:w=iw*(220/640):h=ih*0.20:color=black:t=fill,
      drawbox=x=0:y=ih*0.72:w=iw*(105/640):h=ih*0.30:color=black:t=fill,
      drawbox=x=iw-(iw*(230/640)):y=0:w=iw*(230/640):h=ih*0.24:color=black:t=fill,
      drawbox=x=iw-(iw*(145/640)):y=0:w=iw*(120/640):h=ih*0.49:color=black:t=fill,
      drawbox=x=iw-(iw*(130/640)):y=0:w=iw*(120/640):h=ih*0.51:color=black:t=fill,
      drawbox=x=iw-(iw*(115/640)):y=0:w=iw*(120/640):h=ih*0.53:color=black:t=fill,
      drawbox=x=iw-(iw*(90/640)):y=0:w=iw*(120/640):h=ih:color=black:t=fill,
      drawbox=x=iw-(iw*(105/640)):y=ih*0.68:w=iw*(105/640):h=ih*0.30:color=black:t=fill,
      drawbox=x=iw-(iw*(115/640)):y=ih*0.72:w=iw*(105/640):h=ih*0.30:color=black:t=fill,
      drawbox=x=iw*(430/640):y=ih*(78/480):w=iw*(60/640):h=ih*(60/480):color=black:t=fill"

# Export MASK_FILTER for parallel-invoked functions
export MASK_FILTER

# ---------- Manifest popper (atomic) ----------
popnext() {
  exec 9>>"$CURSOR"
  flock -x 9
  local next
  next=$(grep -vxF -f "$CURSOR" "$MANIFEST" | head -n 1 || true)
  [[ -n "$next" ]] && echo "$next" >> "$CURSOR"
  echo "$next"
  flock -u 9
}

# ---------- Helpers (Python-backed) ----------
# replace your check_dicom_type() with this
check_dicom_type() {
  local dcm="$1"
  local cache="$TEMP_DIR/$(hash_path "$dcm").json"
  [[ -s "$cache" ]] && { cat "$cache"; return 0; }
  python3 - "$dcm" "$cache" <<'PY' 2>>/dev/null
import sys, json, pydicom
dcm, cache = sys.argv[1], sys.argv[2]
try:
    ds = pydicom.dcmread(dcm, force=True, stop_before_pixels=True)
    rows = int(getattr(ds,"Rows",0) or 0)
    cols = int(getattr(ds,"Columns",0) or 0)
    has_px = "PixelData" in ds
    is_image = (rows>0 and cols>0) or has_px
    nframes = int(getattr(ds,"NumberOfFrames",1) or 1)
    info = {"file": dcm, "is_image": bool(is_image), "is_multi_frame": nframes>1,
            "num_frames": nframes, "rows": rows, "cols": cols,
            "modality": getattr(ds,"Modality",""),
            "sop_uid": str(getattr(ds,"SOPClassUID",""))}
    with open(cache,"w") as f: json.dump(info,f)
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({"file": dcm, "error": str(e),
                      "is_image": False, "is_multi_frame": False, "num_frames": 0}))
    sys.exit(0)
PY
}


probe_ok() {
  local dcm="$1"
  local cache="$TEMP_DIR/$(hash_path "$dcm").probe"
  [[ -s "$cache" ]] && return 0
  if OPENJPEG_NUM_THREADS="$OPENJPEG_THREADS" python3 - "$dcm" <<'PY' >/dev/null 2>&1; then
import os, sys, pydicom
os.environ["OPENJPEG_NUM_THREADS"] = os.environ.get("OPENJPEG_NUM_THREADS","2")
pydicom.dcmread(sys.argv[1], force=True).pixel_array
PY
    : > "$cache"; return 0
  else
    return 1
  fi
}

check_frame_count() {
  local dcm="$1"
  local j; j="$(check_dicom_type "$dcm")"
  [[ "$(echo "$j" | python3 -c 'import sys,json;print(json.load(sys.stdin)["is_multi_frame"])')" == "True" ]]
}

get_dimensions() {
  local dcm="$1"
  local cache="$TEMP_DIR/$(hash_path "$dcm").dims"
  [[ -s "$cache" ]] && { cat "$cache"; return 0; }

  local dims
  dims=$(
python3 - "$dcm" <<'PY' 2>&1
import sys, pydicom
fn = sys.argv[1]

def rows_cols(ds):
    r = int(getattr(ds, "Rows", 0) or 0)
    c = int(getattr(ds, "Columns", 0) or 0)
    return (r, c) if (r > 0 and c > 0) else (0, 0)

try:
    # 1) Fast: header-only
    ds = pydicom.dcmread(fn, force=True, stop_before_pixels=True)
    r, c = rows_cols(ds)

    # 2) Fallback: read header with PixelData deferred (no big allocation)
    if not (r and c):
        ds = pydicom.dcmread(fn, force=True, stop_before_pixels=False, defer_size="256 KB")
        r, c = rows_cols(ds)

    # 3) Last-resort: derive from first decoded frame (only if needed)
    if not (r and c):
        try:
            arr = ds.pixel_array
            # shape handling for 2D / 3D / 4D arrays
            if arr.ndim == 2:
                r, c = arr.shape
            else:
                r, c = arr.shape[-2], arr.shape[-1]
        except Exception as e:
            print(f"pixel_array fallback failed: {e}", file=sys.stderr)

    if not (r and c):
        raise RuntimeError("No Rows/Columns")

    print(f"{c}x{r}")
except Exception as e:
    print(f"get_dimensions error for {fn}: {e}", file=sys.stderr)
    sys.exit(2)
PY
  ) || { echo "$dims" >> "$ERRORS"; return 1; }

  echo "$dims" > "$cache"
  printf '%s\n' "$dims"
}


# ---------- Create white sample(s) ----------
create_white_sample() {
  local outdir="$1"
  
  # Safety check: ensure outdir is not empty
  if [[ -z "$outdir" || ! -d "$outdir" ]]; then
    echo "[$HOSTTAG] ERROR: create_white_sample called with invalid dir: '$outdir'" >> "$ERRORS"
    return 1
  fi
  
  local sample="$outdir/mask_visualization.mp4"
  [[ -s "$sample" ]] && return 0
  
  "$FFMPEG_BIN" -loglevel error -y -f lavfi -i "color=white:size=336x336:rate=30:duration=3" \
    -vf "$MASK_FILTER,format=yuv420p" -c:v "$FF_ENC" $QFLAGS "$sample" 2>>"$ERRORS" || {
      echo "white sample failed in $outdir" >> "$ERRORS"; return 1; }
  
  # mirrored unmasked sample
  if [[ "$outdir" != */unmasked/* ]]; then
    local study_dir; study_dir="$(dirname "$outdir")"
    local rel="${outdir#$study_dir/}"
    local um="$study_dir/unmasked/$rel"
    mkdir -p "$um"
    local us="$um/mask_visualization.mp4"
    [[ -s "$us" ]] || "$FFMPEG_BIN" -loglevel error -y -f lavfi -i "color=white:size=336x336:rate=30:duration=3" \
      -vf "format=yuv420p" -c:v "$FF_ENC" $QFLAGS "$us" 2>>"$ERRORS" || {
        echo "unmasked white sample failed in $um" >> "$ERRORS"; return 1; }
  fi
}

# ---------- Convert one DICOM ----------
convert_one() {
  local dcm="$1"           # full path to .dcm file
  local study="$2"         # study UID (dir under $BASE_INPUT_DIR)

  # --- per-file temp workspace
  local pid="$$"
  local tmp="$TEMP_DIR/tmp_${pid}_$RANDOM"
  mkdir -p "$tmp"
  local raw="$tmp/$(basename "$dcm").raw"

  # --- derive output paths
  local study_base; study_base="$(basename "$study")"
  local rel="${dcm#"$BASE_INPUT_DIR/$study"/}"
  local outdir="$OUTPUT_ROOT/$study_base/$(dirname "$rel")"
  local outfile="$outdir/$(basename "${dcm%.dcm}").mp4"
  mkdir -p "$outdir"

  local um_dir="$OUTPUT_ROOT/$study_base/unmasked/$(dirname "$rel")"
  mkdir -p "$um_dir"
  local um_file="$um_dir/$(basename "${dcm%.dcm}").mp4"

  local png_dir="$OUTPUT_ROOT/$study_base/png"
  mkdir -p "$png_dir"
  local png_file="$png_dir/$(basename "${dcm%.dcm}").png"

  local um_png_dir="$OUTPUT_ROOT/$study_base/unmasked/png"
  mkdir -p "$um_png_dir"
  local um_png_file="$um_png_dir/$(basename "${dcm%.dcm}").png"

  # --- fast skip if everything already exists
  if [[ -s "$outfile" && -s "$um_file" && -s "$png_file" && -s "$um_png_file" ]]; then
    rm -rf "$tmp"
    return 0
  fi

  # --- determine dimensions (robust helper); bail if not an image header
  local dims
  if ! dims=$(get_dimensions "$dcm"); then
    echo "$dcm (no Rows/Columns - skipped)" >> "$SKIPPED"
    rm -rf "$tmp"
    return 0
  fi

  # --- check number of frames (we only process multi-frame loops)
  local nframes
  nframes=$(python3 - "$dcm" <<'PY'
import sys, pydicom
ds = pydicom.dcmread(sys.argv[1], force=True, stop_before_pixels=True)
print(int(getattr(ds, "NumberOfFrames", 1) or 1))
PY
  ) || nframes=1
  if [[ -z "$nframes" || "$nframes" -le 1 ]]; then
    echo "$dcm (single-frame - skipped)" >> "$SKIPPED"
    rm -rf "$tmp"
    return 0
  fi

  # --- verify we can decode pixels; try gdcmconv rescue for troublesome files
  if ! probe_ok "$dcm"; then
    if [[ -x "$GDCMCONV" ]]; then
      local conv="$tmp/converted.dcm"
      if "$GDCMCONV" --raw "$dcm" "$conv" >/dev/null 2>&1 && probe_ok "$conv"; then
        dcm="$conv"
      else
        echo "$dcm (unreadable pixel data)" >> "$SKIPPED"
        rm -rf "$tmp"
        return 1
      fi
    else
      echo "$dcm (unreadable pixel data, no gdcmconv)" >> "$SKIPPED"
      rm -rf "$tmp"
      return 1
    fi
  fi

  # --- extract frames → raw RGB (robust, logs shape & bytes)
  # Capture both stdout and stderr from Python
  local python_exit_code=0
  local python_output_file="$tmp/python_output.log"
  
  OPENJPEG_NUM_THREADS="${OPENJPEG_THREADS}" OMP_NUM_THREADS="${PYTHON_THREADS}" \
       NUMEXPR_NUM_THREADS="${PYTHON_THREADS}" MKL_NUM_THREADS="${PYTHON_THREADS}" \
       python3 - "$dcm" "$raw" >"$python_output_file" 2>&1 <<'PY' || python_exit_code=$?
import os, sys, numpy as np, pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

dcm_path, raw_path = sys.argv[1], sys.argv[2]
os.environ.setdefault("OPENJPEG_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","2")

# DEBUG: Log the exact path we're writing to
sys.stderr.write(f"[PYTHON] Target raw path: {raw_path}\n")
sys.stderr.write(f"[PYTHON] Parent dir: {os.path.dirname(raw_path)}\n")
sys.stderr.write(f"[PYTHON] Parent dir exists: {os.path.isdir(os.path.dirname(raw_path))}\n")
sys.stderr.flush()  # Force immediate output

def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    imin, imax = img.min(), img.max()
    if imax <= imin:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img.astype(np.float32) - imin) / (imax - imin) * 255.0).astype(np.uint8)

def to_gray3(fr, pi):
    """Convert frame to grayscale with 3 identical channels"""
    if fr.ndim == 2:
        g = fr
    elif fr.shape[-1] == 3:
        if pi.startswith('YBR'):
            # If YBR, first channel is luminance (Y)
            g = fr[..., 0]
        else:
            # ITU-R BT.709 luma coefficients for RGB
            g = 0.2126 * fr[..., 0] + 0.7152 * fr[..., 1] + 0.0722 * fr[..., 2]
    else:
        g = fr[..., 0]
    g8 = to_uint8(g)
    return np.stack([g8, g8, g8], axis=-1)

try:
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array
    pi = getattr(ds, 'PhotometricInterpretation', '') or ''
    print(f"decoded shape={getattr(arr,'shape',None)} dtype={arr.dtype} PI={pi} NF={getattr(ds,'NumberOfFrames',1)}", file=sys.stderr)

    # Heuristic guard: only convert YBR→RGB if it still *looks* like YCbCr (Cb/Cr near 128 on average)
    if pi.startswith('YBR') and arr.ndim >= 3 and arr.shape[-1] == 3:
        try:
            # Check if chroma channels are near neutral (128) - indicates true YBR
            m0, m1, m2 = [float(arr[..., k].mean()) for k in range(3)]
            if abs(m1 - 128) < 40 and abs(m2 - 128) < 40:  # chroma near neutral
                arr = convert_color_space(arr, pi, 'RGB')
                print("colorspace: YBR→RGB (applied)", file=sys.stderr)
            else:
                print(f"colorspace: looks like RGB already (means: {m0:.1f}, {m1:.1f}, {m2:.1f}); skip convert", file=sys.stderr)
        except Exception as e:
            print(f"colorspace convert skipped: {e}", file=sys.stderr)

    frames = []
    nf = int(getattr(ds, 'NumberOfFrames', 1) or 1)

    # Normalize to [F, H, W, 3]
    if arr.ndim == 2:                          # single grayscale
        frames = [np.stack([arr, arr, arr], axis=-1)]
    elif arr.ndim == 3:
        if arr.shape[-1] == 3:                  # single RGB
            frames = [arr]
        elif arr.shape[0] == nf and nf > 1:     # [F, H, W]
            frames = [np.stack([f, f, f], axis=-1) for f in arr]
        else:                                   # treat as single grayscale
            frames = [np.stack([arr, arr, arr], axis=-1)]
    elif arr.ndim == 4:
        if arr.shape[-1] == 3:                  # [F, H, W, 3]
            frames = [f for f in arr]
        else:                                   # make last dim=3
            frames = [np.stack([f[...,0], f[...,0], f[...,0]], axis=-1) for f in arr]
    else:
        raise RuntimeError(f"Unsupported array ndim={arr.ndim}")

    if len(frames) == 0:
        raise RuntimeError("decoded 0 frames")
    
    # Apply grayscale conversion if FORCE_GRAY is enabled
    if os.getenv("FORCE_GRAY", "1") == "1":
        print("colorspace: converting to grayscale (FORCE_GRAY=1)", file=sys.stderr)
        frames = [to_gray3(f, pi) for f in frames]
    else:
        print("colorspace: preserving color (FORCE_GRAY=0)", file=sys.stderr)
        frames = [to_uint8(f) if f.dtype != np.uint8 else f for f in frames]

    # write raw WITH EXPLICIT FLUSH AND FSYNC
    total = 0
    with open(raw_path, "wb") as f:
        for fr in frames:
            fr8 = to_uint8(fr)
            b = np.ascontiguousarray(fr8).tobytes()
            f.write(b)
            total += len(b)
        
        # CRITICAL: Force flush to OS and sync to disk
        f.flush()
        os.fsync(f.fileno())
    
    # After closing file, sync parent directory to ensure directory entry is visible
    import time
    try:
        dir_fd = os.open(os.path.dirname(raw_path), os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)
    except Exception:
        pass  # Best effort - some filesystems don't support directory fsync
    
    # CRITICAL: Verify file with retry for GPFS visibility
    max_retries = 50
    for attempt in range(max_retries):
        try:
            sz = os.stat(raw_path).st_size
            if sz > 0:
                break
        except FileNotFoundError:
            if attempt < max_retries - 1:
                time.sleep(0.05)  # 50ms between retries
                continue
            else:
                sys.stderr.write(f"ERROR: File not visible after {max_retries} retries\n")
                sys.exit(1)
    
    if sz == 0:
        sys.stderr.write(f"ERROR: File is empty (0 bytes)\n")
        sys.exit(1)
    
    if sz != total:
        sys.stderr.write(f"WARNING: Size mismatch: wrote {total} bytes, file is {sz} bytes\n")
    
    print(f"SUCCESS: wrote {total} bytes, verified {sz} bytes on disk to {raw_path}", file=sys.stderr)
    sys.exit(0)

except Exception as e:
    sys.stderr.write(f"extract error: {e}\n")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PY
  
  # Check Python exit code and capture output
  if [[ $python_exit_code -ne 0 ]]; then
    echo "[$HOSTTAG] Python extraction failed with exit code $python_exit_code for $dcm" >> "$ERRORS"
    echo "[$HOSTTAG] Python output:" >> "$ERRORS"
    cat "$python_output_file" >> "$ERRORS" 2>&1
    echo "$dcm (pixel extraction failed, exit code $python_exit_code)" >> "$SKIPPED"
    rm -rf "$tmp"
    return 1
  fi
  
  # Log Python output for debugging
  if [[ -s "$python_output_file" ]]; then
    cat "$python_output_file" >> "$ERRORS"
  fi
  
  # File visibility wait with retry logic for GPFS
  # GPFS metadata caching can delay file visibility for several seconds
  local max_wait=100  # Maximum 10 seconds (very conservative for GPFS)
  local wait_count=0
  while [[ ! -e "$raw" && $wait_count -lt $max_wait ]]; do
    sleep 0.1
    ((wait_count++))
    
    # Every second, try to force directory sync
    if [[ $((wait_count % 10)) -eq 0 ]]; then
      sync 2>/dev/null || true
    fi
  done

  # Final check: verify raw file exists and has content
  if [[ ! -e "$raw" ]]; then
    echo "[$HOSTTAG] CRITICAL: Python exit code 0 but raw file never appeared after $((max_wait * 100))ms: $raw" >> "$ERRORS"
    echo "[$HOSTTAG] Tmp dir contents:" >> "$ERRORS"
    ls -laR "$tmp" >> "$ERRORS" 2>&1
    echo "[$HOSTTAG] DICOM file: $dcm" >> "$ERRORS"
    echo "[$HOSTTAG] TEMP_DIR base: $TEMP_DIR (expected in: $tmp)" >> "$ERRORS"
    echo "[$HOSTTAG] Python output was:" >> "$ERRORS"
    [[ -s "$python_output_file" ]] && cat "$python_output_file" >> "$ERRORS" 2>&1
    echo "$dcm (no raw after Python exit 0, waited $((max_wait * 100))ms)" >> "$SKIPPED"
    rm -rf "$tmp"
    return 1
  fi

  # --- ensure the raw exists AND has content before FFmpeg
  raw_size=$(stat -c%s "$raw" 2>/dev/null || echo "0")
  if [[ ! -s "$raw" || "$raw_size" -eq 0 ]]; then
    echo "[$HOSTTAG] CRITICAL: Raw file empty or missing for $dcm" >> "$ERRORS"
    echo "[$HOSTTAG] File: $raw, Size: $raw_size bytes" >> "$ERRORS"
    if [[ -e "$raw" ]]; then
      echo "[$HOSTTAG] File exists but is empty (size=0)" >> "$ERRORS"
    else
      echo "[$HOSTTAG] File does not exist at all" >> "$ERRORS"
    fi
    echo "$dcm (no raw produced, size=$raw_size)" >> "$ERRORS"
    echo "$dcm (no raw produced)" >> "$SKIPPED"
    rm -rf "$tmp"
    return 1
  fi

  # Log success for debugging
  echo "[$HOSTTAG] $(date +%T) Raw file ready: $raw (${raw_size} bytes) for $(basename "$dcm")" >> "$NODE_LOG_DIR/extraction_success.log"

  # --- FFmpeg: masked/unmasked MP4 + PNG thumbs
  if ! "$FFMPEG_BIN" -loglevel error -y \
        -f rawvideo -pixel_format rgb24 -video_size "$dims" -r 30 -i "$raw" \
        -filter_complex "[0:v]split=2[base1][base2]; \
                         [base1]$MASK_FILTER,split=2[v1][v2]; \
                         [v1]scale=336:336,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p[vout1]; \
                         [v2]scale=224:224[vout2]; \
                         [base2]split=2[v3][v4]; \
                         [v3]scale=336:336,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p[vout3]; \
                         [v4]scale=224:224[vout4]" \
        -map "[vout1]" -c:v "$FF_ENC" $QFLAGS "$outfile" \
        -map "[vout2]" -frames:v 1 -f image2 "$png_file" \
        -map "[vout3]" -c:v "$FF_ENC" $QFLAGS "$um_file" \
        -map "[vout4]" -frames:v 1 -f image2 "$um_png_file" 2>>"$ERRORS"
  then
    echo "$dcm (FFmpeg failure)" >> "$SKIPPED"
    rm -f "$outfile" "$um_file" "$png_file" "$um_png_file"
    rm -rf "$tmp"
    return 1
  fi

  rm -rf "$tmp"
  return 0
}




# ---------- Process one study ----------
process_study() {
  local study="$1"
  local in_dir="$BASE_INPUT_DIR/$study"
  local done_flag="$OUTPUT_ROOT/$study/.done"
  local um_dir="$OUTPUT_ROOT/$study/unmasked"

  if [[ -f "$done_flag" && -z "${FORCE_REDO:-}" ]]; then
    echo "[$HOSTTAG] Study $study already processed" >> "$PROCESSING_LOG"
    return 0
  fi

  mkdir -p "$OUTPUT_ROOT/$study" "$um_dir"
  echo "[$HOSTTAG] $(date +%T) -> Processing $study" | tee -a "$PROCESSING_LOG"

  local list="$TEMP_DIR/${study}_dicoms.list"
  # Try to find files with .dcm or .dicom extensions first
  find "$in_dir" -type f \( -iname "*.dcm" -o -iname "*.dicom" \) -size +0c > "$list" 2>/dev/null || true
  local n; n=$(wc -l < "$list" || echo 0)
  
  # If no .dcm/.dicom files found, try all files (assume no-extension DICOMs)
  if [[ "$n" -eq 0 ]]; then
    find "$in_dir" -type f -size +0c > "$list" 2>/dev/null || true
    n=$(wc -l < "$list" || echo 0)
    if [[ "$n" -gt 0 ]]; then
      echo "[$HOSTTAG] Found $n files (no extension) in $study" >> "$PROCESSING_LOG"
    fi
  else
    echo "[$HOSTTAG] Found $n DICOM files (.dcm/.dicom) in $study" >> "$PROCESSING_LOG"
  fi
  
  if [[ "$n" -eq 0 ]]; then
    echo "[$HOSTTAG] No files found; leaving unflagged" | tee -a "$PROCESSING_LOG"
    return 0
  fi

  # conversion (force bash so exported functions are visible)
  export SHELL=/bin/bash
  export -f convert_one create_white_sample

  parallel \
    --jobs "${SLURM_CPUS_PER_TASK:-$JOBS_PER_STUDY}" \
    --joblog "$NODE_LOG_DIR/parallel_${study}.log" \
    --delay 0.2 \
    convert_one {} "$study" :::: "$list"

  # white sample at study level
  create_white_sample "$OUTPUT_ROOT/$study"

  # white samples in study subdirs (exclude png/unmasked)
  if [[ -d "$OUTPUT_ROOT/$study" ]]; then
    find "$OUTPUT_ROOT/$study" -mindepth 1 -maxdepth 1 -type d ! -name png ! -name unmasked 2>/dev/null \
      | grep -v '^$' \
      | parallel -j 8 create_white_sample {}
  fi

  # under unmasked as well
  if [[ -d "$OUTPUT_ROOT/$study/unmasked" ]]; then
    find "$OUTPUT_ROOT/$study/unmasked" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
      | grep -v '^$' \
      | parallel -j 8 create_white_sample {}
  fi

  # summary & flag (robust numeric handling)
  mp4=$(find "$OUTPUT_ROOT/$study" -type f -name "*.mp4" ! -name "mask_visualization.mp4" ! -path "*/unmasked/*" 2>/dev/null | wc -l | awk '{print $1}')
  mp4_unm=$(find "$OUTPUT_ROOT/$study/unmasked" -type f -name "*.mp4" ! -name "mask_visualization.mp4" ! -path "*/png/*" 2>/dev/null | wc -l | awk '{print $1}')
  png=$(find "$OUTPUT_ROOT/$study/png" -type f -name "*.png" 2>/dev/null | wc -l | awk '{print $1}')
  png_unm=$(find "$OUTPUT_ROOT/$study/unmasked/png" -type f -name "*.png" 2>/dev/null | wc -l | awk '{print $1}')

  echo "[$HOSTTAG] Completed $study at $(date)" >> "$PROCESSING_LOG"
  echo "[$HOSTTAG] MP4(masked): ${mp4:-0} | MP4(unmasked): ${mp4_unm:-0} | PNG(masked): ${png:-0} | PNG(unmasked): ${png_unm:-0}" >> "$PROCESSING_LOG"

  total=$(( ${mp4:-0} + ${mp4_unm:-0} + ${png:-0} + ${png_unm:-0} ))
  if (( total > 0 )); then
    touch "$done_flag"
  else
    echo "[$HOSTTAG] No outputs produced; NOT flagging as done" | tee -a "$PROCESSING_LOG"
  fi

}

# ---------- Worker loop ----------
worker() {
  while study="$(popnext)"; do
    [[ -z "$study" ]] && break
    process_study "$study"
  done
}

# === Export functions (must be AFTER all defs above) ===
export SHELL=/bin/bash
export -f hash_path popnext check_dicom_type probe_ok check_frame_count get_dimensions \
          create_white_sample convert_one process_study worker
export BASE_INPUT_DIR OUTPUT_ROOT SHARED_DIR MANIFEST CURSOR HOSTTAG NODE_ID \
       NODE_LOG_DIR ERRORS SKIPPED PROCESSING_LOG TEMP_DIR MASK_FILTER \
       FFMPEG_BIN FF_ENC QFLAGS PYTHON_THREADS OPENJPEG_THREADS GDCMCONV \
       SLURM_CPUS_PER_TASK JOBS_PER_STUDY FORCE_REDO

# ---------- SINGLE-STUDY FAST PATH ----------
if [[ -n "${SINGLE_STUDY:-}" ]]; then
  echo "[$HOSTTAG] SINGLE_STUDY → $SINGLE_STUDY" | tee -a "$PROCESSING_LOG"
  process_study "$SINGLE_STUDY"
  echo "[$HOSTTAG] SINGLE_STUDY finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$PROCESSING_LOG"
  exit 0
fi

# ---------- Manifest mode ----------
if [[ ! -s "$MANIFEST" ]]; then
  echo "ERROR: manifest not found or empty: $MANIFEST" >&2
  exit 1
fi
touch "$CURSOR"

echo "[$HOSTTAG] Starting workers: $PAR_STUDIES" | tee -a "$PROCESSING_LOG"
seq "$PAR_STUDIES" | xargs -n1 -P"$PAR_STUDIES" bash -c 'worker'

# ---------- Final summary ----------
echo "[$HOSTTAG] All done at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$PROCESSING_LOG"
TOTAL_PROCESSED=$(ls -1 "$OUTPUT_ROOT"/*/.done 2>/dev/null | wc -l || echo 0)
TOTAL_MP4=$(find "$OUTPUT_ROOT" -type f -name "*.mp4" ! -name "mask_visualization.mp4" ! -path "*/unmasked/*" 2>/dev/null | wc -l || echo 0)
TOTAL_UNMASKED=$(find "$OUTPUT_ROOT" -path "*/unmasked/*" -name "*.mp4" ! -name "mask_visualization.mp4" ! -path "*/png/*" 2>/dev/null | wc -l || echo 0)
TOTAL_PNG=$(find "$OUTPUT_ROOT" -path "*/png/*.png" 2>/dev/null | wc -l || echo 0)
TOTAL_UNMASKED_PNG=$(find "$OUTPUT_ROOT" -path "*/unmasked/png/*.png" 2>/dev/null | wc -l || echo 0)
{
  echo "[$HOSTTAG] SUMMARY: studies: $TOTAL_PROCESSED"
  echo "[$HOSTTAG] SUMMARY: MP4(masked): $TOTAL_MP4 | MP4(unmasked): $TOTAL_UNMASKED"
  echo "[$HOSTTAG] SUMMARY: PNG(masked):  $TOTAL_PNG  | PNG(unmasked):  $TOTAL_UNMASKED_PNG"
} | tee -a "$PROCESSING_LOG"
