# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import pickle
import pprint


from .hook import HOOKS, Hook
@HOOKS.register_module()
class eval(Hook):
    def __init__(self, work_dir=None, th=0.5,**kwargs) -> None:
        self.save_folder = work_dir
        self.stats = []
        self.kwargs = kwargs
        self.th = int(th*2000)

    def before_run(self, runner):
        work_dir = runner.work_dir
        if not self.save_folder:
            self.save_folder = work_dir
        # self.save_folder = Path(self.save_folder)
        os.makedirs(work_dir,exist_ok=True)

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        result = runner.result
        preds = result["pred"]
        try:
            gt = result["gt"]
        except:
            runner.run_mode="pass"
            return -1
        correct_bboxes = np.zeros((len(preds), 10))
        matches = np.empty((0,2))
        stat =(correct_bboxes, np.empty(0), np.empty(0), np.empty(0))
        
        if len(gt) > 0 and len(preds) == 0:
            stat = (correct_bboxes, np.empty(0), np.empty(0), gt[:, 5])
        elif len(preds) > 0 and len(gt) > 0:
            correct_bboxes,matches = runner.process_batch(preds, gt)
            stat = ((correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 5]))
        elif len(preds) > 0 and len(gt) == 0:
            stat = (correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 5])
            
        self.stats.append(stat)

        
        runner.result.update(
            {"stats": stat, "matches": matches}  # array([[label, detect, iou]])
        )
        
        runner.total_results[result["image_file"]].update(
            {"stats": stat, "matches": matches} 
        )

    def after_run(self, runner):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            runner.metrics.process(*stats)

        result = runner.metrics.result
        px, ps, rs = result[0:3]

        print(f"[info]:最优阈值:{px[runner.metrics.index]}")
        pprint.pprint(runner.metrics.results_dict)
        print("-" * 30)
        print(f"[info]阈值: {px[self.th]:.3f}")
        print(f"prec:{np.round(ps[:,self.th],3)}\nrec:{np.round(rs[:,self.th],3)}")
        out_file = os.path.join(self.save_folder, "plot_data.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(result, f)
