import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm.auto import tqdm


from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

from src.model.anygrasper import AnyGrasper
from src.dataset.RGBD import RGBD
from src.dataset.BaseRGBDDataset import BaseRGBDDataset
from src.utils.viz import capture, BEST_VIEW


def main(cfg: DictConfig):

    grasper: AnyGrasper = instantiate(cfg.anygrasper)

    dataloader: DataLoader = instantiate(cfg.dataloader)

    dataset: BaseRGBDDataset = dataloader.dataset

    iterator = iter(dataloader)
    pbar = tqdm(
        range(len(dataset)),
        desc="Processing",
        total=len(dataset),
        dynamic_ncols=True,
        leave=True,
    )
    for data in iterator:
        #TODO fix the collate_fn when point_cloud is true
        rgb = data['rgb'].squeeze(0).numpy() / 255.0
        depth = data['depth'].squeeze(0).numpy()
        intrinsics = data['intrinsics'].squeeze(0).numpy()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        lims = [
            cfg.workspace_limits.xmin, cfg.workspace_limits.xmax,
            cfg.workspace_limits.ymin, cfg.workspace_limits.ymax,
            cfg.workspace_limits.zmin, cfg.workspace_limits.zmax
        ]
        xmap, ymap = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        points_z = depth
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
        colors = rgb[mask].astype(np.float32)
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]],dtype=np.float32)
        points = points @ R.T
        
        gg, cloud = grasper(points, colors, lims=lims)
        
        if gg is None:
            pbar.update(1)
            continue
        if cfg.debug:
            trans_mat = BEST_VIEW
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
                
            o3d.visualization.draw_geometries_with_key_callbacks(
                [cloud, *grippers],
                {ord('C'): capture}
            )
            # o3d.visualization.draw_geometries([grippers[0], cloud])
        pbar.update(1)
