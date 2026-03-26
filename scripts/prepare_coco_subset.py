from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_ID_MAP = {
    1: 0,   # person
    27: 1,  # backpack
    31: 2,  # handbag
    33: 3,  # suitcase
}

CLASS_NAMES = {
    0: "person",
    1: "backpack",
    2: "handbag",
    3: "suitcase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a COCO subset for person/backpack/handbag/suitcase in YOLO format."
    )
    parser.add_argument(
        "--coco-root",
        required=True,
        help="Path to the extracted COCO 2017 root containing train2017, val2017, and annotations/.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Path where the YOLO-format subset will be created.",
    )
    parser.add_argument(
        "--include-val-as-test",
        action="store_true",
        help="Also mirror the validation split into images/test and labels/test for convenience.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_root = Path(args.coco_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    convert_split(
        split="train",
        images_dir=coco_root / "train2017",
        annotations_path=coco_root / "annotations" / "instances_train2017.json",
        output_root=output_root,
    )
    convert_split(
        split="val",
        images_dir=coco_root / "val2017",
        annotations_path=coco_root / "annotations" / "instances_val2017.json",
        output_root=output_root,
    )

    if args.include_val_as_test:
        mirror_split_as_test(output_root)

    write_data_yaml(output_root, include_test=args.include_val_as_test)
    print(f"Done. YOLO subset saved to: {output_root}")


def convert_split(split: str, images_dir: Path, annotations_path: Path, output_root: Path) -> None:
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    image_id_to_info = {image["id"]: image for image in data["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in data["annotations"]:
        category_id = annotation["category_id"]
        if category_id not in CLASS_ID_MAP:
            continue
        if annotation.get("iscrowd", 0) == 1:
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    kept_images = 0
    kept_boxes = 0

    for image_id, annotations in annotations_by_image.items():
        image_info = image_id_to_info.get(image_id)
        if image_info is None:
            continue

        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]
        source_image = images_dir / file_name

        if not source_image.exists():
            continue

        yolo_lines = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            if w <= 1 or h <= 1:
                continue

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            width_norm = w / width
            height_norm = h / height

            yolo_class_id = CLASS_ID_MAP[annotation["category_id"]]
            yolo_lines.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            )

        if not yolo_lines:
            continue

        destination_image = output_images / file_name
        destination_label = output_labels / f"{Path(file_name).stem}.txt"

        shutil.copy2(source_image, destination_image)
        destination_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        kept_images += 1
        kept_boxes += len(yolo_lines)

    print(f"{split}: kept {kept_images} images and {kept_boxes} boxes")


def mirror_split_as_test(output_root: Path) -> None:
    val_images = output_root / "images" / "val"
    val_labels = output_root / "labels" / "val"
    test_images = output_root / "images" / "test"
    test_labels = output_root / "labels" / "test"

    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)

    for image_path in val_images.glob("*"):
        if image_path.is_file():
            shutil.copy2(image_path, test_images / image_path.name)

    for label_path in val_labels.glob("*.txt"):
        shutil.copy2(label_path, test_labels / label_path.name)


def write_data_yaml(output_root: Path, include_test: bool) -> None:
    lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")

    lines.extend(
        [
            "",
            "names:",
            "  0: person",
            "  1: backpack",
            "  2: handbag",
            "  3: suitcase",
            "",
        ]
    )

    (output_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
