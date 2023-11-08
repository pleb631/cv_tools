from collections import defaultdict
import tqdm
import mmcv
import os
import shutil


from .DetMetrics import DetMetrics,process_batch
from .priority import get_priority
from .hooks import HOOKS, Hook


class runner(object):
    def __init__(self, model,dataset,classes,work_dir,**kwargs):
        super(runner, self).__init__()
        
        self.model = model
        self.cls2num = classes
        self.dataset = dataset
        self.num2cls = {int(classes[i]): i for i in classes.keys()}

        self.classes = list(classes.values())
        self.total_results = defaultdict(dict)
        self.kwargs = kwargs
        self.stats = []
        self.metrics = DetMetrics(names=self.num2cls,plot=True,save_dir=work_dir,**kwargs)
        self._hooks=[]
        self.process_batch = process_batch
        self.work_dir = work_dir
        if kwargs.__contains__("overwrite") and kwargs["overwrite"] and os.path.exists(work_dir):
            shutil.rmtree(work_dir)   
        os.makedirs(work_dir)


    def run(self):
        img_num = len(self.dataset)
        self.call_hook('before_run')
        for i in tqdm.tqdm(range(img_num)):
            image_info = self.dataset[i]
            self.image_info = image_info
            self.call_hook('before_iter')
            self.call_hook('after_iter')
        self.call_hook('after_run')
               
    def register_hook(self,
                      hook: Hook,
                      priority = 'NORMAL') -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
            
            
    def register_hook_from_cfg(self, hook_cfg) -> None:
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)
            
    def call_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        self.run_mode = "continue"
        for hook in self._hooks:
            if self.run_mode=="pass":
                break
            getattr(hook, fn_name)(self)
        