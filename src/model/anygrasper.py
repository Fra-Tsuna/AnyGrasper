import torch
from pathlib import Path
from omegaconf import DictConfig
import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup


class AnyGrasper():
    def __init__(self, anygrasp_cfg: DictConfig,
                    apply_object_mask: bool = True,
                    dense_grasp: bool = False,
                    collision_detection: bool = True):
        
        self.cfg = anygrasp_cfg
        self.anygrasp = AnyGrasp(self.cfg)
        self.anygrasp.load_net()
        
        self.apply_object_mask = apply_object_mask
        self.dense_grasp = dense_grasp
        self.collision_detection = collision_detection
        
    def __call__(self, points, colors, lims):
        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=lims,
                                            apply_object_mask=self.apply_object_mask,
                                            dense_grasp=self.dense_grasp,
                                            collision_detection=self.collision_detection)
        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:20]
        return gg_pick, cloud