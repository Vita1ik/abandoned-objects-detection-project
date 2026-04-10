from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


CLASS_NAMES = ["person", "backpack", "handbag", "suitcase"]
SPLITS = ("train", "val", "test")
LUGGAGE_CLASS_IDS = {1, 2, 3}
PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class SampleInfo:
    image_path: Path
    label_path: Path
    classes: frozenset[int]
    box_counts: Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Balance a merged YOLO dataset by keeping all luggage-positive samples and "
            "downsampling person-only samples."
        )
    )
    parser.add_argument("--input-root", required=True, help="Path to the merged YOLO dataset root.")
    parser.add_argument("--output-root", required=True, help="Path where the balanced dataset will be created.")
    parser.add_argument(
        "--person-only-ratio",
        type=float,
        default=1.0,
        help=(
            "How many person-only images to keep relative to luggage-positive images in each split. "
            "For example 1.0 keeps at most the same number of person-only images as luggage-positive images."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for person-only sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rng = random.Random(args.seed)

    if args.person_only_ratio < 0:
        raise ValueError("--person-only-ratio must be non-negative.")

    for split in SPLITS:
        balance_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            person_only_ratio=args.person_only_ratio,
            rng=rng,
        )

    copy_data_yaml(input_root, output_root)
    print(f"\nDone. Balanced YOLO dataset saved to: {output_root}")


def balance_split(
    split: str,
    input_root: Path,
    output_root: Path,
    person_only_ratio: float,
    rng: random.Random,
) -> None:
    image_dir = input_root / "images" / split
    label_dir = input_root / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        return

    samples = collect_samples(image_dir, label_dir)
    if not samples:
        return

    luggage_positive: list[SampleInfo] = []
    person_only: list[SampleInfo] = []
    backgrounds: list[SampleInfo] = []
    other: list[SampleInfo] = []

    for sample in samples:
        if sample.classes & LUGGAGE_CLASS_IDS:
            luggage_positive.append(sample)
        elif sample.classes == {PERSON_CLASS_ID}:
            person_only.append(sample)
        elif not sample.classes:
            backgrounds.append(sample)
        else:
            other.append(sample)

    person_only_limit = min(len(person_only), round(len(luggage_positive) * person_only_ratio))
    selected_person_only = rng.sample(person_only, person_only_limit) if person_only_limit else []
    selected = luggage_positive + selected_person_only + backgrounds + other
    selected.sort(key=lambda sample: sample.image_path.name)

    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    class_image_counts = Counter()
    class_box_counts = Counter()

    for sample in selected:
        shutil.copy2(sample.image_path, output_images / sample.image_path.name)
        shutil.copy2(sample.label_path, output_labels / sample.label_path.name)

        stats["images"] += 1
        if not sample.classes:
            stats["backgrounds"] += 1

        for class_id, count in sample.box_counts.items():
            class_box_counts[class_id] += count
        for class_id in sample.classes:
            class_image_counts[class_id] += 1

    print(f"\n[{split}]")
    print(
        f"selected images={stats['images']} luggage_positive={len(luggage_positive)} "
        f"person_only_kept={len(selected_person_only)} person_only_total={len(person_only)} "
        f"backgrounds={len(backgrounds)}"
    )
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: images={class_image_counts[class_id]} "
            f"boxes={class_box_counts[class_id]}"
        )


def collect_samples(image_dir: Path, label_dir: Path) -> list[SampleInfo]:
    samples: list[SampleInfo] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        label_path = label_dir / f"{image_path.stem}.txt"
        box_counts = parse_label_counts(label_path)
        samples.append(
            SampleInfo(
                image_path=image_path,
                label_path=label_path,
                classes=frozenset(box_counts),
                box_counts=box_counts,
            )
        )
    return samples


def parse_label_counts(label_path: Path) -> Counter:
    counts: Counter = Counter()
    if not label_path.exists():
        return counts

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if not parts:
            continue
        class_id = int(float(parts[0]))
        counts[class_id] += 1
    return counts


def copy_data_yaml(input_root: Path, output_root: Path) -> None:
    data = yaml.safe_load((input_root / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(output_root)
    (output_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
