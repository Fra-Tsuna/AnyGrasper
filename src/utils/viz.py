import numpy as np

flip = np.array([
                [ 1,  0,  0, 0],
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 0,  0,  0, 1]
            ], dtype=np.float32)
theta = np.deg2rad(-45.0)
rot_x = np.array([
    [1,           0,            0, 0],
    [0,  np.cos(theta), -np.sin(theta), 0],
    [0,  np.sin(theta),  np.cos(theta), 0],
    [0,           0,            0, 1]
], dtype=np.float32)
trans_mat = rot_x @ flip

BEST_VIEW = trans_mat

def capture(vis):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    print("\n=== Camera Parameters ===")
    # full 4x4 extrinsic matrix
    print("Extrinsic:")
    print(params.extrinsic)
    # compute camera center and orientation axes in world coords
    R_mat = params.extrinsic[:3, :3]
    t_vec = params.extrinsic[:3, 3]
    cam_center = -R_mat.T @ t_vec
    front = R_mat.T @ np.array([0, 0, 1], dtype=np.float32)
    up = R_mat.T @ np.array([0, 1, 0], dtype=np.float32)
    print(f"Center: {cam_center}")
    print(f"Front:  {front}")
    print(f"Up:     {up}")
    print("========================\n")
    return False