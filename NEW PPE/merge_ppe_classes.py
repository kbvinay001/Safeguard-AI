"""
merge_ppe_classes.py — PPE Dataset Class Merger
================================================
Converts the raw Roboflow PPE dataset from 17 noisy classes to a clean
set of 11 meaningful classes by remapping class IDs in every label file
in-place.  Also writes a new data_clean.yaml that points to the updated
labels so YOLO training uses the merged class set.

Run this script ONCE before training the PPE model:
    python "NEW PPE/merge_ppe_classes.py"

Original 17 classes → Merged 11 classes:
  SafetyShoe + boot       → shoe (0)
  faceMask                → mask (1)
  glove + gloves          → glove (2)
  goggle                  → goggle (3)
  hardhat + helmet        → helmet (4)
  no vest + no_vest       → no_vest (5)
  no_faceMask             → no_mask (6)
  no_gloves               → no_glove (7)
  no_helmet               → no_helmet (8)
  person                  → person (9)
  vest                    → vest (10)
  head (ambiguous)        → DROPPED
  object (noise class)    → DROPPED
"""

import io
import sys
import shutil
from pathlib import Path

# Force UTF-8 output so Windows console does not fail on special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Class remapping table
# ---------------------------------------------------------------------------
# Keys are the original class indices (0–16) from the Roboflow export.
# Values are the new indices (0–10), or None to drop that class entirely.
#
# Original class list:
#   0=SafetyShoe  1=boot       2=faceMask    3=glove      4=gloves
#   5=goggle      6=hardhat    7=head        8=helmet     9=no vest
#  10=no_faceMask 11=no_gloves 12=no_helmet 13=no_vest   14=object
#  15=person      16=vest
# ---------------------------------------------------------------------------
MERGE_MAP = {
    0:  0,     # SafetyShoe → shoe
    1:  0,     # boot       → shoe  (merged)
    2:  1,     # faceMask   → mask
    3:  2,     # glove      → glove
    4:  2,     # gloves     → glove (merged)
    5:  3,     # goggle     → goggle
    6:  4,     # hardhat    → helmet
    7:  None,  # head       → DROPPED (too ambiguous, hurts training mAP)
    8:  4,     # helmet     → helmet (merged with hardhat)
    9:  5,     # no vest    → no_vest
    10: 6,     # no_faceMask → no_mask
    11: 7,     # no_gloves  → no_glove
    12: 8,     # no_helmet  → no_helmet
    13: 5,     # no_vest    → no_vest (merged with "no vest")
    14: None,  # object     → DROPPED (generic noise class)
    15: 9,     # person     → person
    16: 10,    # vest       → vest
}

# New class names in index order (0–10)
MERGED_CLASS_NAMES = [
    "shoe",       # 0
    "mask",       # 1
    "glove",      # 2
    "goggle",     # 3
    "helmet",     # 4
    "no_vest",    # 5
    "no_mask",    # 6
    "no_glove",   # 7
    "no_helmet",  # 8
    "person",     # 9
    "vest",       # 10
]

# Dataset root — all split folders are relative to this
DATASET_DIR = Path(__file__).parent
DATASET_SPLITS = ["train", "valid", "test"]


def remap_label_file(label_file: Path):
    """
    Remap class IDs in a single YOLO label (.txt) file.

    Each line in a YOLO label file has the format:
        <class_id> <cx> <cy> <width> <height>

    This function replaces the class_id with the new merged ID,
    or removes the line entirely if the class is being dropped.

    Args:
        label_file (Path): Path to the .txt label file to update

    Returns:
        tuple: (kept_count int, dropped_count int)
    """
    try:
        raw_text = label_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0, 0

    new_lines = []
    kept    = 0
    dropped = 0

    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        try:
            original_class_id = int(parts[0])
        except (ValueError, IndexError):
            continue   # Skip malformed lines

        new_class_id = MERGE_MAP.get(original_class_id)
        if new_class_id is None:
            # This class is in the drop list — remove the annotation entirely
            dropped += 1
            continue

        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))
        kept += 1

    # Write the updated label back to the same file
    updated_content = "\n".join(new_lines) + "\n" if new_lines else ""
    label_file.write_text(updated_content, encoding="utf-8")
    return kept, dropped


def main():
    separator = "=" * 60
    print(separator)
    print("  PPE Class Merger — 17 classes → 11 clean classes")
    print(separator)

    # Step 1: Back up the original label files before we modify them.
    # The backup is skipped if it already exists (idempotent re-runs).
    backup_dir = DATASET_DIR / "labels_backup_original"
    if not backup_dir.exists():
        print("\n[1/3] Backing up original label files...")
        for split in DATASET_SPLITS:
            source_labels = DATASET_DIR / split / "labels"
            if source_labels.exists():
                shutil.copytree(source_labels, backup_dir / split / "labels")
        print(f"      Backup saved to: {backup_dir}")
    else:
        print("\n[1/3] Backup already exists — skipping.")

    # Step 2: Remap all label files in every split
    print("\n[2/3] Remapping label files...")
    total_files   = 0
    total_kept    = 0
    total_dropped = 0

    for split in DATASET_SPLITS:
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.exists():
            print(f"      {split}/labels — folder not found, skipping")
            continue

        label_files  = list(labels_dir.glob("*.txt"))
        split_kept   = 0
        split_dropped = 0

        for label_file in label_files:
            kept, dropped = remap_label_file(label_file)
            split_kept    += kept
            split_dropped += dropped

        total_files   += len(label_files)
        total_kept    += split_kept
        total_dropped += split_dropped

        print(
            f"      {split:6s} : {len(label_files):4d} files  |"
            f"  kept {split_kept:6d}  |  dropped {split_dropped:5d}"
        )

    print(
        f"\n      Total  : {total_files} files  |"
        f"  kept {total_kept}  |  dropped {total_dropped}"
    )

    # Step 3: Write the clean dataset YAML that points to the remapped labels
    print("\n[3/3] Writing data_clean.yaml...")
    yaml_lines = [
        "# PPE Dataset — Merged class labels (17 original → 11 clean classes)",
        "train: ../train/images",
        "val:   ../valid/images",
        "test:  ../test/images",
        "",
        f"nc: {len(MERGED_CLASS_NAMES)}",
        f"names: {MERGED_CLASS_NAMES}",
    ]
    yaml_path = DATASET_DIR / "data_clean.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"      Saved: {yaml_path}")

    print("\n" + separator)
    print("  Class merge complete!")
    print(f"  New class count : {len(MERGED_CLASS_NAMES)}")
    print(f"  New class names : {MERGED_CLASS_NAMES}")
    print()
    print('  Next step: run  python "NEW PPE/train_optimised.py"')
    print(separator)


if __name__ == "__main__":
    main()
