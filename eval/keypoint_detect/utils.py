import numpy as np
import re
import cv2
import os
from pathlib import Path
import json


def creatdict(imgpath, kpts, wh):
    kpts = kpts.reshape(-1).tolist()
    params = {"image_path": imgpath, "results": kpts, "wh": wh}
    return params


def load_kpts_gt(image_path):
    if len(re.findall("images", image_path)) > 1:
        anno_path_part = list(Path(image_path).parts)
        for index in range(len(anno_path_part) - 1, -1, -1):
            if anno_path_part[index] == "images":
                anno_path_part[index] = "annotations"
                break
        anno_path = os.path.join(*anno_path_part)
    else:
        anno_path = image_path.replace("images", "annotations")

    anno_path = os.path.splitext(anno_path)[0] + ".json"
    assert os.path.exists(anno_path), image_path

    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not (isinstance(data, dict) or isinstance(data, list)):
        raise "content in annotations file is not list or dict"

    if isinstance(data, list):
        #raise "需要修改"
        return np.array(data).reshape(-1, 2),1#####需要修改

    if isinstance(data, dict):
        data1 = data.copy()
        if "annotation" in data1:
            data1 = data1["annotation"]["license"][0]
        landmark = data1["landmark"]
        quality=data1["quality"]
        if not len(landmark):
            print(data)

        return np.array(landmark).reshape(-1, 2),quality


def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage

def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1

def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt

def printf(content, logger=None):
    print(content, file=logger)
    print(content)


import functools
round4 = functools.partial(round,ndigits=4)
def compute_pck_acc(result, logger, args):
    preds = np.concatenate([instance["pred"] for instance in result], 0)
    gts = np.concatenate([instance["gt"] for instance in result], 0)
    target_weight = np.concatenate([instance["mask"] for instance in result], 0)[...,0]

    kpt_indexs = args.kpt_indexs
    if (isinstance(kpt_indexs,list) or isinstance(kpt_indexs,tuple)) and len(kpt_indexs)==2:
        normalize = np.linalg.norm(gts[:,kpt_indexs[0],:]-gts[:,kpt_indexs[1],:],axis=1).reshape(-1,1)
        #normalize = np.tile(normalize,(1,2))
        
    else:
        normalize=np.ones((preds.shape[0], 1), dtype=np.float32)
    
    
    acc, avg_acc, cnt = keypoint_pck_accuracy(
            preds,
            gts,
            target_weight,
            thr=args.pck_th,
            normalize=normalize)
    #printf(f"{acc=}\n{L2dist=}\n", logger)
    printf(f"pck_th={args.pck_th:.2f}", logger)
    printf(f"{avg_acc=:.4f}", logger)
    printf(f"each point acc:", logger)
    printf(f'{list(map(round4,acc.tolist()))}', logger)

    return 0


def ones2origin(pkts, img_shape):
    pkts = pkts * img_shape
    return pkts


def compute_dist_acc(result, logger, args):
    preds = np.concatenate([instance["pred"] for instance in result], 0)
    gts = np.concatenate([instance["gt"] for instance in result], 0)
    hws = np.concatenate(
        [np.array(instance["wh"]).reshape(1, 1, 2) for instance in result], 0
    )
    preds = preds * hws
    gts = gts * hws
    thresh = [4, 5, 6, 7, 8]
    dist = np.sqrt(np.sum(np.square(preds - gts), 2))
    dist = np.transpose(dist, (1, 0))
    for th in thresh:
        printf(f"[INFO]{th=}", logger)
        TPMask = dist < th
        printf(f"all={np.sum(TPMask)/TPMask.size:.2f}", logger)
        for kp in TPMask:
            pre = np.sum(kp) / kp.size
            printf(f"{pre=:.2f}", logger)

    return 0


methodFactory = {
    "pck": compute_pck_acc,
    "dist": compute_dist_acc,
}
