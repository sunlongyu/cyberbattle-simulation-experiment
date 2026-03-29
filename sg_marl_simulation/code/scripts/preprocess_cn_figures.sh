#!/usr/bin/env bash
set -euo pipefail

MODE="dry-run"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/_archive/preprocess_cn_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/preprocess_cn_figures.sh [--dry-run|--archive|--clean]

Modes:
  --dry-run  只检查并打印候选清理项（默认）
  --archive  将候选产物移动到 _archive/ 下（安全）
  --clean    直接删除候选产物（谨慎）
EOF
}

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run|--archive|--clean)
      MODE="${1#--}"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
fi

cd "$ROOT_DIR"

declare -a TARGETS=(
  "experiments/**/epoch_results"
  "experiments/**/figures"
  "experiments/**/training_curves.png"
  "experiments/**/*.png"
  "experiments/**/*.pdf"
  "ray_results"
  "results"
  "logs"
  "tensorboard"
  "mappo_checkpoints"
  "mappo_signaling_checkpoints"
)

echo "[INFO] Repo root: $ROOT_DIR"
echo "[INFO] Mode: $MODE"

CANDIDATES=()
while IFS= read -r line; do
  CANDIDATES+=("$line")
done < <(
  for pattern in "${TARGETS[@]}"; do
    find . -path "./.git" -prune -o -path "./${pattern}" -print
  done | sort -u
)

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[INFO] No generated artifacts found."
  exit 0
fi

echo "[INFO] Candidate artifacts (${#CANDIDATES[@]}):"
for path in "${CANDIDATES[@]}"; do
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null || true
  fi
done

if [[ "$MODE" == "dry-run" ]]; then
  echo "[INFO] Dry run completed. No files changed."
  exit 0
fi

if [[ "$MODE" == "archive" ]]; then
  mkdir -p "$ARCHIVE_DIR"
  for path in "${CANDIDATES[@]}"; do
    [[ -e "$path" ]] || continue
    rel="${path#./}"
    dest="$ARCHIVE_DIR/$rel"
    mkdir -p "$(dirname "$dest")"
    mv "$path" "$dest"
    echo "[ARCHIVE] $rel -> ${dest#$ROOT_DIR/}"
  done
  echo "[INFO] Archived to ${ARCHIVE_DIR#$ROOT_DIR/}"
  exit 0
fi

for path in "${CANDIDATES[@]}"; do
  [[ -e "$path" ]] || continue
  rm -rf "$path"
  echo "[CLEAN] removed ${path#./}"
done

echo "[INFO] Clean completed."
