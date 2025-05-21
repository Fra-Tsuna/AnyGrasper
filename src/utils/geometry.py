import numpy as np
import open3d as o3d
import math
from typing import Tuple

def invert_se3(se3_matrix: np.ndarray) -> np.ndarray:
    """
    From https://github.com/sachaMorin/rgbd_dataset/blob/main/rgbd_dataset/utils.py
    """
    rotation_inv = se3_matrix[:3, :3].T
    translation_inv = -rotation_inv @ se3_matrix[:3, 3]

    inverted_se3_matrix = np.eye(4)
    inverted_se3_matrix[:3, :3] = rotation_inv
    inverted_se3_matrix[:3, 3] = translation_inv

    return inverted_se3_matrix


def rgbd_to_pcd(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    depth_trunc: float = 8.0,
    depth_scale: float = 1.0,
) -> o3d.geometry.PointCloud:
    """
    From https://github.com/sachaMorin/rgbd_dataset/blob/main/rgbd_dataset/rgbd_to_pcd.py
    """

    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
        depth_trunc=depth_trunc,
        depth_scale=depth_scale,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd_image,
        intrinsic=intrinsics_o3d,
        extrinsic=invert_se3(camera_pose),  # World to cam
    )

    return pcd

def rt_to_pose(R: np.ndarray, t: np.ndarray):
    """
    Convert a rotation matrix and translation vector into a 7-vector pose.

    Parameters
    ----------
    R : np.ndarray, shape (3,3)
        A valid rotation matrix.
    t : np.ndarray, shape (3,)
        Translation vector [x, y, z].

    Returns
    -------
    pose : np.ndarray, shape (7,)
        [x, y, z, qx, qy, qz, qw], where the quaternion is normalized.
    """
    # Ensure inputs are the right shape
    assert R.shape == (3, 3)
    assert t.shape == (3,)

    m00, m01, m02 = R[0, :]
    m10, m11, m12 = R[1, :]
    m20, m21, m22 = R[2, :]
    trace = m00 + m11 + m22

    if trace > 0.0:
        S = np.sqrt(trace + 1.0) * 2.0  # S = 4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    else:
        # find the largest diagonal element
        if (m00 > m11) and (m00 > m22):
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0  # S=4*qx
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0  # S=4*qy
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0  # S=4*qz
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S

    # Normalize quaternion
    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q)

    # Position
    x, y, z = t

    return np.array([x, y, z, q[0], q[1], q[2], q[3]])


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a 3×3 rotation matrix to Tait–Bryan angles (roll, pitch, yaw).

    Args:
        R: a 3×3 numpy array.

    Returns:
        (roll, pitch, yaw) in radians.
    """
    assert R.shape == (3, 3), "R must be 3×3"

    # sy = sqrt(R00² + R10²)
    sy = math.hypot(R[0, 0], R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2( R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch ~ ±90°
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0

    return roll, pitch, yaw