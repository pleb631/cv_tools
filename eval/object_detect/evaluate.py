import tqdm
import argparse
from pathlib import Path
from itertools import chain

from uits import *
from schemes import *
import os


formats = [".jpg", ".png", ".jpeg",".bmp"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default=r"D:\project\datasets",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=r"D:\project\imgalz\weights\yolov8s.onnx",
    )

    parser.add_argument("--mean", nargs="+", type=int, default=[0, 0, 0])
    parser.add_argument("--std", nargs="+", type=int, default=[1, 1, 1])
    parser.add_argument("--classes", nargs="+", type=int, default=list(range(80)))
    parser.add_argument("--obj_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--padding", type=bool, default=True)
    parser.add_argument(
        "--save_path",
        type=Path,
        default="./temp",
    )

    parser.add_argument("--use_yolov8", type=bool, default=True)
    parser.add_argument("--aug_test", type=bool, default=False)
    # mode
    parser.add_argument("--save_json", type=str, choices=["coco","yolov5",None],default='coco')
    parser.add_argument("--save_img", type=bool, default=False)
    parser.add_argument("--val", type=bool, default=True)
    parser.add_argument("--val1", type=bool, default=True)
    parser.add_argument("--only_save_badcase", type=bool, default=False)

    return parser.parse_args()


def main(args):
    print(f"load model from {args.model_path}")
    if args.save_path == "":
        args.save_path = os.path.join(args.model_path.parent / "output")

    evaluator = OnnxScheme(args)

    image_paths = sorted(chain(*[args.data_path.rglob("*" + f) for f in formats]))
    print(f"The number of all processed files: {len(image_paths)}")
    for image_path in tqdm.tqdm(image_paths):
        if not image_path.is_file():
            print(image_path)
            continue
        evaluator.run(image_path)
    print("-" * 30)
    evaluator.show_metric()


if __name__ == "__main__":
    main(args())
