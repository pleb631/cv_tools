import tqdm
import argparse
from pathlib import Path
from itertools import chain

from uits import *
from schemes import *
import os


formats = [".jpg", ".png", ".jpeg"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num2cls = {0: "persons", 1: "cars"}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default=r"datasets/val",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="model.onnx",
    )
    parser.add_argument("--scheme", type=str, default="onnx")
    parser.add_argument(
        "--image_size", nargs="+", type=int, default=(640, 640), help="(w,h)"
    )
    parser.add_argument("--mean", nargs="+", type=int, default=[0, 0, 0])
    parser.add_argument("--std", nargs="+", type=int, default=[1, 1, 1])
    parser.add_argument("--classes", nargs="+", type=dict, default=num2cls)
    parser.add_argument("--obj_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.3)
    parser.add_argument("--padding", type=bool, default=True)
    parser.add_argument(
        "--save_path",
        type=Path,
        default="./temp",
    )
    parser.add_argument("--filter_area", type=int, default=0)
    parser.add_argument("--npy_path", type=str, default="")
    parser.add_argument("--use_yolov8", type=bool, default=True)

    # mode
    parser.add_argument("--save_box", type=bool, default=True)
    parser.add_argument("--save_img", type=bool, default=False)
    parser.add_argument("--val", type=bool, default=True)
    parser.add_argument("--only_save_badcase", type=bool, default=True)

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
