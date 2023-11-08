import tqdm
import argparse
from pathlib import Path
from itertools import chain
import os

from detection import OnnxScheme
        

formats = ['.jpg','.png','.jpeg']
#os.environ['CUDA_VISIBLE_DEVICES']='9'


       
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=r"/mnt/data4/dataset/face_landmark/val"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="/home/dml/project/mmcv1.7/mmlab/work_dirs/shufflentv2fpn/latest.onnx",
    )
    parser.add_argument("--scheme", type=str, default="onnx")
    parser.add_argument(
        "--image_size", nargs="+", type=int, default=(112, 112), help="(w,h)"
    )
    parser.add_argument("--mean", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--std", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--padding", type=bool, default=False)
    parser.add_argument(
        "--save_path",
        type=Path,
        default="/home/dml/project/KPDValTools/monitor-landmark_detect-v2_1_1_rknn",
    )

    parser.add_argument("--methods", nargs="+", type=str, default=["pck"])

    parser.add_argument(
        "--pck_th", type=float, default=0.06, help="通常使用0.05,0.1,0.2,0.3,0.4"
    )
    parser.add_argument(
        "--kpt_indexs",
        nargs="+",
        type=int,
        default=1,
        help="选择归一化所需要的两个索引,如果是单索引,则归一化参数默认是图片长宽",
    )

    parser.add_argument(
        "--npy_root",
        type=str,
        default="/mnt/data3/alg_workspace/common/rknn_quant/face_landmark/out/",
    )

    parser.add_argument("--heat_maps",action="store_true",default=False)
    parser.add_argument("--badcase_th", type=float, default=0.7)

    # mode
    parser.add_argument("--save_kpts", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=True)
    parser.add_argument("--save_badcase", action="store_true", default=True)
    parser.add_argument("--save_pred", action="store_true", default=True)
    parser.add_argument("--validated_dataset", action="store_true", default=False)
    parser.add_argument("--use_npy", action="store_true", default=False)

    return parser.parse_args()


def main(args):
    if args.save_path == "":
        args.save_path = os.path.join(args.model_path.parent / "output")

    print(f"load model from {args.model_path}")
    evaluator = OnnxScheme(args)

    image_paths = sorted(chain(*[args.data_path.rglob("*" + f) for f in formats]))
    print(f"The number of all processed files: {len(image_paths)}")
    for image_path in tqdm.tqdm(image_paths):
        if not image_path.is_file():
            print(image_path)
            continue
        evaluator.run(str(image_path))
    print("-" * 30)
    evaluator.show_metric()


if __name__ == "__main__":
    main(args())
