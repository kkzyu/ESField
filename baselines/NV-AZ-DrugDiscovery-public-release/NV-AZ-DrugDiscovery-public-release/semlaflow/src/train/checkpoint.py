from abc import ABC
from pathlib import Path
from time import time
import importlib
import os


from natsort import natsorted
import torch


class Criterion(ABC):
    def reset(self):
        pass

    def save(self, *args, **kwargs):
        pass


class TimeCriterion(Criterion):
    def __init__(self, save_every=1800):
        self.time = 0
        self.save_every = save_every

    def reset(self):
        self.time = time()

    def save(self, *args, **kwargs):
        return (time() - self.time) > self.save_every


class CheckpointManager:
    def __init__(
        self, model, optimizer, save_path, seed, batch=0, epoch=0, device="cpu"
    ):
        self.model = model
        if not hasattr(model, "get_state_dict") or not hasattr(
            model, "load_from_state_dict"
        ):
            raise ValueError(
                "model needs to have implemented `get_state_dict` and `load_from_state_dict`"
            )
        self.optimizer = optimizer
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = seed
        self.epoch = epoch
        self.batch = batch

    def get_state_dict(self):
        params = {
            "seed": self.seed,
            "epoch": self.epoch,
            "batch": self.batch,
            "device": self.device,
        }
        return params

    def save(self, epoch, batch):
        model_state = self.model.get_state_dict()
        opt_state = self.optimizer.state_dict()
        opt_state["class_name"] = self.optimizer.__class__.__name__
        opt_state["class_module"] = self.optimizer.__class__.__module__
        t = int(time())
        torch.save(model_state, self.save_path / f"model_{t:d}_{epoch:d}_{batch:d}.pt")
        torch.save(opt_state, self.save_path / f"opt_{t:d}_{epoch:d}_{batch:d}.pt")

        self.epoch = epoch
        self.batch = batch
        checkpoint_state = self.get_state_dict()

        torch.save(
            checkpoint_state, self.save_path / f"chpt_{t:d}_{epoch:d}_{batch:d}.pt"
        )

    @staticmethod
    def load_model(model_state, device):
        model_class_name = model_state["class_name"]
        model_class_module = model_state["class_module"]
        try:
            if model_class_module == "__main__":
                model_class = globals()[model_class_name]
            else:
                model_class = getattr(
                    importlib.import_module(model_class_module), model_class_name
                )
        except BaseException:
            raise ValueError(f"Model class `{model_class_name}` not found")
        model = model_class.load_from_state_dict(model_state, device)
        return model

    @staticmethod
    def load_optimizer(opt_state, model):
        opt_class_name = opt_state["class_name"]
        opt_class_module = opt_state["class_module"]
        try:
            if opt_class_module == "__main__":
                optimizer_class = globals()[opt_class_name]
            else:
                optimizer_class = getattr(
                    importlib.import_module(opt_class_module), opt_class_name
                )
        except BaseException:
            raise ValueError(f"Optimizer class `{opt_class_name}` not found")
        opt = optimizer_class(model.parameters())
        # Some optimizer-state keys can be absent when loading a conditional
        # checkpoint into an unconditional model (or vice versa); we fall
        # through and keep the freshly-initialized optimizer in that case.
        try:
            opt.load_state_dict(opt_state)
        except ValueError:
            pass
        return opt

    @classmethod
    def restore_last_checkpoint(cls, save_path):
        save_path = Path(save_path)
        checkpoints = natsorted([f for f in save_path.glob("chpt_*.pt")])[::-1]
        checkpoint_loaded = False
        for checkpoint in checkpoints:
            # try:
            if True:
                chpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                device = chpt["device"]
                model_chpt = os.path.basename(str(checkpoint)).replace(
                    "chpt_", "model_"
                )
                opt_chpt = os.path.basename(str(checkpoint)).replace("chpt_", "opt_")

                model_state = torch.load(
                    save_path / model_chpt, map_location="cpu", weights_only=False
                )
                opt_state = torch.load(
                    save_path / opt_chpt, map_location=device, weights_only=False
                )

                model = cls.load_model(model_state, device)
                opt = cls.load_optimizer(opt_state, model)
                checkpoint_manager = cls(model, opt, save_path, **chpt)

                checkpoint_loaded = True
                break
            # except BaseException:
            #    pass
        if not checkpoint_loaded:
            return None, None, None
        else:
            print(checkpoint)
        return model, opt, checkpoint_manager
