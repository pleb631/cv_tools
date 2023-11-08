# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import numpy as np

from .hook import HOOKS, Hook


def save_json(json_path, info, indent=4, mode="w", with_return_char=False, **kwargs):
    json_str = json.dumps(info, indent=indent)
    if with_return_char:
        json_str += "\n"

    with open(json_path, mode, encoding="UTF-8") as json_file:
        json_file.write(json_str)

    json_file.close()


def save_result(info, save_keys, save_path, **kwargs):
    results = []
    for path, value in info.items():
        single_result = {"image_file": path}
        for key in save_keys:
            v = value.get(key, None)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            single_result[key] = v
        results.append(single_result)
    save_json(save_path, results, **kwargs)


@HOOKS.register_module()
class run_model(Hook):
    def __init__(
        self, collect_keys, save_key=[], use_key=None, work_dir=None, **kwargs
    ) -> None:
        self.collect_keys = collect_keys
        self.save_key = save_key
        self.kwargs = kwargs
        self.use_key = use_key
        self.save_folder = work_dir

    def before_run(self, runner):
        work_dir = runner.work_dir
        if not self.save_folder:
            self.save_folder = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def before_iter(self, runner):
        if runner.image_info.__contains__(self.use_key):
            pred_box = runner.image_info[self.use_key]
        else:
            pred_box = runner.model.detect(runner.image_info)

        
        runner.image_info.update({"pred": pred_box})
        runner.result = runner.image_info

    def after_iter(self, runner):
        runner.total_results[runner.result["image_file"]].update(
            {"pred_box": runner.result["pred"]}
        )

        result = {}
        for key in self.collect_keys:
            result[key] = runner.result.get(key, None)
        runner.total_results[result["image_file"]].update(result)

    def after_run(self, runner):
        if len(self.save_key):
            file_path = os.path.join(self.save_folder, "info.json")
            save_result(runner.total_results, self.save_key, file_path, **self.kwargs)
