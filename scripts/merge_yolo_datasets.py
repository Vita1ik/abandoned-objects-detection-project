from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


DEFAULT_CLASS_ORDER = ["person", "backpack", "handbag", "suitcase"]
OUTPUT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO-format datasets into one dataset with a unified class order."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Path to a YOLO dataset root containing data.yaml. Repeat for each dataset to merge.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Path where the merged YOLO dataset will be created.",
    )
    parser.add_argument(
        "--class-order",
        nargs="+",
        default=DEFAULT_CLASS_ORDER,
        help="Canonical output class order. Defaults to person backpack handbag suitcase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [Path(path).expanduser().resolve() for path in args.dataset]
    output_root = Path(args.output_root).expanduser().resolve()
    class_order = list(args.class_order)

    for split in OUTPUT_SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_stats: dict[str, Counter] = {split: Counter() for split in OUTPUT_SPLITS}

    for dataset_root in datasets:
        dataset_stats = merge_dataset(dataset_root, output_root, class_order)
        print(f"\nMerged dataset: {dataset_root}")
        for split in OUTPUT_SPLITS:
            if not dataset_stats[split]:
                continue
            total_stats[split].update(dataset_stats[split])
            print(
                f"  [{split}] images={dataset_stats[split]['images']} "
                f"backgrounds={dataset_stats[split]['backgrounds']} boxes={dataset_stats[split]['boxes']}"
            )

    write_data_yaml(output_root, class_order)

    print(f"\nDone. Merged YOLO dataset saved to: {output_root}")
    for split in OUTPUT_SPLITS:
        if not total_stats[split]:
            continue
        print(
            f"  [{split}] images={total_stats[split]['images']} "
            f"backgrounds={total_stats[split]['backgrounds']} boxes={total_stats[split]['boxes']}"
        )


def merge_dataset(dataset_root: Path, output_root: Path, class_order: list[str]) -> dict[str, Counter]:
    data_yaml = load_data_yaml(dataset_root / "data.yaml")
    source_names = parse_names(data_yaml.get("names"))
    class_id_map = build_class_id_map(source_names, class_order)
    dataset_prefix = sanitize_prefix(dataset_root.name)
    stats: dict[str, Counter] = {split: Counter() for split in OUTPUT_SPLITS}

    for source_key, output_split in (("train", "train"), ("val", "val"), ("test", "test")):
        if source_key not in data_yaml:
            continue

        split_ref = data_yaml[source_key]
        split_path = resolve_split_path(dataset_root, split_ref)
        if not split_path.exists():
            continue

        if split_path.is_file():
            merge_split_from_list(
                dataset_root=dataset_root,
                dataset_prefix=dataset_prefix,
                split_list_path=split_path,
                output_root=output_root,
                output_split=output_split,
                class_id_map=class_id_map,
                stats=stats[output_split],
            )
        else:
            label_dir = infer_label_dir(split_path)
            merge_split(
                dataset_prefix=dataset_prefix,
                image_dir=split_path,
                label_dir=label_dir,
                output_root=output_root,
                output_split=output_split,
                class_id_map=class_id_map,
                stats=stats[output_split],
            )

    return stats


def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_names(raw_names) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(name) for _, name in sorted(raw_names.items(), key=lambda item: int(item[0]))]
    raise ValueError("Dataset data.yaml must define class names as a list or dict.")


def build_class_id_map(source_names: list[str], class_order: list[str]) -> dict[int, int]:
    target_lookup = {name: index for index, name in enumerate(class_order)}
    class_id_map: dict[int, int] = {}

    for source_index, class_name in enumerate(source_names):
        if class_name not in target_lookup:
            continue
        class_id_map[source_index] = target_lookup[class_name]

    return class_id_map


def resolve_split_path(dataset_root: Path, split_path: str) -> Path:
    split = Path(split_path)
    candidates: list[Path] = []

    if split.is_absolute():
        candidates.append(split)
    else:
        candidates.append((dataset_root / split).resolve())
        stripped_parts = [part for part in split.parts if part not in ("..", ".")]
        if stripped_parts:
            candidates.append((dataset_root / Path(*stripped_parts)).resolve())
        if len(split.parts) >= 2:
            candidates.append((dataset_root / split.parts[-2] / split.parts[-1]).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def infer_label_dir(image_dir: Path) -> Path:
    if image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name
    return image_dir.parent / "labels"


def merge_split(
    dataset_prefix: str,
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    output_split: str,
    class_id_map: dict[int, int],
    stats: Counter,
) -> None:
    output_images = output_root / "images" / output_split
    output_labels = output_root / "labels" / output_split

    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        output_name = f"{dataset_prefix}__{image_path.name}"
        label_path = label_dir / f"{image_path.stem}.txt"
        converted_lines = convert_label_file(label_path, class_id_map)

        shutil.copy2(image_path, output_images / output_name)
        (output_labels / f"{Path(output_name).stem}.txt").write_text(
            "\n".join(converted_lines) + ("\n" if converted_lines else ""),
            encoding="utf-8",
        )

        stats["images"] += 1
        stats["boxes"] += len(converted_lines)
        if not converted_lines:
            stats["backgrounds"] += 1


def merge_split_from_list(
    dataset_root: Path,
    dataset_prefix: str,
    split_list_path: Path,
    output_root: Path,
    output_split: str,
    class_id_map: dict[int, int],
    stats: Counter,
) -> None:
    output_images = output_root / "images" / output_split
    output_labels = output_root / "labels" / output_split
    split_name = split_list_path.stem

    image_paths = [
        resolve_listed_image_path(dataset_root, line)
        for line in split_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for image_path in image_paths:
        if not image_path.exists():
            continue

        output_name = f"{dataset_prefix}__{image_path.name}"
        label_path = resolve_listed_label_path(dataset_root, split_name, image_path)
        converted_lines = convert_label_file(label_path, class_id_map)

        shutil.copy2(image_path, output_images / output_name)
        (output_labels / f"{Path(output_name).stem}.txt").write_text(
            "\n".join(converted_lines) + ("\n" if converted_lines else ""),
            encoding="utf-8",
        )

        stats["images"] += 1
        stats["boxes"] += len(converted_lines)
        if not converted_lines:
            stats["backgrounds"] += 1


def resolve_listed_image_path(dataset_root: Path, listed_path: str) -> Path:
    candidate = Path(listed_path)
    if candidate.is_absolute():
        return candidate
    return (dataset_root / candidate).resolve()


def resolve_listed_label_path(dataset_root: Path, split_name: str, image_path: Path) -> Path:
    stem = image_path.stem
    candidates = [
        dataset_root / "labels" / split_name / f"{stem}.txt",
        dataset_root / "labels" / split_name.capitalize() / f"{stem}.txt",
        dataset_root / "labels" / split_name.lower() / f"{stem}.txt",
        dataset_root / "labels" / split_name.upper() / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def convert_label_file(label_path: Path, class_id_map: dict[int, int]) -> list[str]:
    if not label_path.exists():
        return []

    converted_lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) < 5:
            continue

        source_class_id = int(float(parts[0]))
        target_class_id = class_id_map.get(source_class_id)
        if target_class_id is None:
            continue

        converted_lines.append(" ".join([str(target_class_id), *parts[1:]]))

    return converted_lines


def write_data_yaml(output_root: Path, class_order: Iterable[str]) -> None:
    lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for index, name in enumerate(class_order):
        lines.append(f"  {index}: {name}")
    lines.append("")

    (output_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def sanitize_prefix(name: str) -> str:
    safe = []
    for char in name:
        safe.append(char if char.isalnum() else "_")
    return "".join(safe).strip("_") or "dataset"


if __name__ == "__main__":
    main()
