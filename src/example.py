"""
Example script to run AnyGrasp on a single image.
Taken from official repo.
github.com/graspnet/anygrasp_sdk/tree/main/grasp_detection
"""

import hydra
from omegaconf import DictConfig
import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

def main(cfg: DictConfig):

    anygrasp = AnyGrasp(cfg.anygrasp)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(cfg.data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(cfg.data_dir, 'depth.png')))

    # camera intrinsics from config
    fx, fy = cfg.camera.fx, cfg.camera.fy
    cx, cy = cfg.camera.cx, cfg.camera.cy
    scale = cfg.camera.scale

    lims = [
        cfg.workspace_limits.xmin, cfg.workspace_limits.xmax,
        cfg.workspace_limits.ymin, cfg.workspace_limits.ymax,
        cfg.workspace_limits.zmin, cfg.workspace_limits.zmax
    ]

    # point cloud computation
    xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    print(points.min(axis=0), points.max(axis=0))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfg.anygrasp.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])
