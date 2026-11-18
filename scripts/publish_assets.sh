#!/usr/bin/env bash
set -euo pipefail

# Publishes asset files from <dist_dir> (default: dist/assets) into two permanent,
# bottom-pinned GitHub Releases on strataui/earth:
#   - assets        : latest unversioned flat files + manifest.json
#   - assets-legacy : superseded files archived with -v1, -v2, ... suffixes
#
# Usage:
#   ./scripts/publish_assets.sh [dist_dir] [--dry-run]

REPO="${STRATA_EARTH_GITHUB_REPO:-strataui/earth}"
DIST_DIR="${1:-dist/assets}"
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

if [[ ! -d "$DIST_DIR" ]]; then
  echo "error: asset directory not found: $DIST_DIR" >&2
  exit 1
fi

command -v gh >/dev/null 2>&1 || { echo "error: gh CLI not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 not found" >&2; exit 1; }

ensure_release() {
  local tag="$1"
  local title="$2"
  local notes="$3"
  if gh release view "$tag" --repo "$REPO" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would create permanent release tag '$tag' on $REPO"
  else
    gh release create "$tag" --repo "$REPO" --title "$title" --notes "$notes"
  fi
}

ensure_release "assets" "Assets (Latest)" "Permanent rolling release holding latest unversioned Strata Earth asset packs and manifest.json."
ensure_release "assets-legacy" "Assets (Legacy Archive)" "Permanent archive of superseded Strata Earth asset packs with -v1, -v2 version suffixes."

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Download previous manifest.json if present so updated files can be archived cleanly into assets-legacy.
PREV_MANIFEST="$TMP_DIR/prev_manifest.json"
if gh release download "assets" --repo "$REPO" --pattern "manifest.json" --dir "$TMP_DIR" >/dev/null 2>&1; then
  mv "$TMP_DIR/manifest.json" "$PREV_MANIFEST"
else
  echo '{"schema_version":1,"assets":{}}' > "$PREV_MANIFEST"
fi

# Plan version bumps and legacy archiving via Python.
PLAN_FILE="$TMP_DIR/plan.txt"
python3 - "$PREV_MANIFEST" "$DIST_DIR" "$PLAN_FILE" <<'PY'
import hashlib, json, os, sys

prev_path, dist_dir, plan_path = sys.argv[1], sys.argv[2], sys.argv[3]
prev = json.load(open(prev_path)).get("assets", {})
cur_manifest_path = os.path.join(dist_dir, "manifest.json")
cur = json.load(open(cur_manifest_path)).get("assets", {}) if os.path.exists(cur_manifest_path) else {}

lines = []
for fname in sorted(os.listdir(dist_dir)):
    if fname == "manifest.json":
        continue
    fpath = os.path.join(dist_dir, fname)
    if not os.path.isfile(fpath):
        continue
    new_sha = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
    new_size = os.path.getsize(fpath)
    old = prev.get(fname)
    if old and old.get("sha256") and old["sha256"] != new_sha:
        old_ver = int(old.get("version", 1))
        stem, ext = os.path.splitext(fname)
        legacy_name = f"{stem}-v{old_ver}{ext}"
        new_ver = old_ver + 1
        lines.append(f"ARCHIVE\t{fname}\t{legacy_name}")
    elif old:
        new_ver = int(old.get("version", 1))
        legacy_name = old.get("legacy_filename")
        if old.get("sha256") == new_sha:
            lines.append(f"SKIP\t{fname}\t")
            continue
    else:
        new_ver = 1
        legacy_name = None

    entry = cur.get(fname, {"filename": fname})
    entry["version"] = new_ver
    entry["sha256"] = new_sha
    entry["bytes"] = new_size
    if legacy_name:
        entry["legacy_filename"] = legacy_name
    cur[fname] = entry
    lines.append(f"UPLOAD\t{fname}\t{legacy_name or ''}")

with open(cur_manifest_path, "w") as f:
    json.dump({"schema_version": 1, "assets": cur}, f, indent=2, sort_keys=True)
    f.write("\n")

with open(plan_path, "w") as f:
    for l in lines:
        f.write(l + "\n")
PY

while IFS=$'\t' read -r action fname legacy_name; do
  case "$action" in
    ARCHIVE)
      echo "[publish] archiving previous $fname -> assets-legacy/$legacy_name"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] gh release download assets --pattern $fname && gh release upload assets-legacy $legacy_name"
      else
        gh release download "assets" --repo "$REPO" --pattern "$fname" --dir "$TMP_DIR"
        mv "$TMP_DIR/$fname" "$TMP_DIR/$legacy_name"
        gh release upload "assets-legacy" "$TMP_DIR/$legacy_name" --repo "$REPO" --clobber
      fi
      ;;
    UPLOAD)
      echo "[publish] uploading $fname -> assets/"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] gh release upload assets $DIST_DIR/$fname --clobber"
      else
        gh release upload "assets" "$DIST_DIR/$fname" --repo "$REPO" --clobber
      fi
      ;;
    SKIP)
      echo "[publish] unchanged $fname (skipping)"
      ;;
  esac
done < "$PLAN_FILE"

echo "[publish] uploading manifest.json -> assets/"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] gh release upload assets $DIST_DIR/manifest.json --clobber"
else
  gh release upload "assets" "$DIST_DIR/manifest.json" --repo "$REPO" --clobber
fi
