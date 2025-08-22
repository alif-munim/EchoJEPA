# watcher.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./watcher.sh <DIR_TO_MONITOR> <KEEP_COUNT> [INTERVAL_SECONDS] [IGNORE_PREFIX]
DIR="${1:-}"
KEEP="${2:-}"
INTERVAL="${3:-60}"
IGNORE_PREFIX="${4:-keep}"   # files starting with this prefix are never deleted

if [[ -z "$DIR" || -z "$KEEP" ]]; then
  echo "Usage: $0 <DIR_TO_MONITOR> <KEEP_COUNT> [INTERVAL_SECONDS] [IGNORE_PREFIX]" >&2; exit 1
fi
[[ -d "$DIR" ]] || { echo "Directory not found: $DIR" >&2; exit 1; }
[[ "$KEEP" =~ ^[0-9]+$ && "$KEEP" -ge 1 ]] || { echo "KEEP must be integer >=1" >&2; exit 1; }

ts() { date '+%Y-%m-%d %H:%M:%S'; }

prune() {
  cd "$DIR"
  # Newest-first list of *.pt (robust when none exist)
  mapfile -t files < <( { ls -1t -- *.pt 2>/dev/null || true; } )

  # Filter out items we should never delete
  filtered=()
  for f in "${files[@]}"; do
    # Always protect latest.pt; also protect IGNORE_PREFIX*
    if [[ "$f" == "latest.pt" ]]; then continue; fi
    if [[ -n "$IGNORE_PREFIX" && "$f" == "$IGNORE_PREFIX"* ]]; then continue; fi
    filtered+=("$f")
  done

  echo "$(ts) — scan: found ${#files[@]} *.pt | protected: latest.pt + prefix='$IGNORE_PREFIX' | candidates=${#filtered[@]} | KEEP=$KEEP"

  if ((${#filtered[@]} > KEEP)); then
    to_delete=("${filtered[@]:KEEP}")
    echo "$(ts) — deleting: ${to_delete[*]}"
    printf '%s\0' "${to_delete[@]}" | xargs -0 -r rm -f --
  else
    echo "$(ts) — nothing to delete."
  fi
}

echo "$(ts) — watcher started: DIR='$DIR' KEEP=$KEEP INTERVAL=${INTERVAL}s IGNORE_PREFIX='${IGNORE_PREFIX}'"
trap 'echo "$(ts) — stop signal received; exiting."; exit 0' INT TERM
prune
while sleep "$INTERVAL"; do prune; done
