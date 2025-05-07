import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from scipy.spatial.transform import Rotation
import trimesh
import imageio.v2 as imageio
import PIL.Image

import torch
def geotrf(Trf, pts, ncol=None, norm=False):
    # function from dust3r.utils.geometry.geotrf
    # if you have a dust3r repo, use "from dust3r.utils.geometry import geotrf"
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


# def convert_pose_for_viz(pose):
#     # pose[:3, :3] = R.T  
#     # pose[:3, 3] = -R.T @ t
#     pose_viz = np.eye(4)
#     pose_viz[:3, :3] = pose[:3, :3].T
#     pose_viz[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
#     return pose_viz

from kaggle_utils.metric import score_transf, read_csv, tth_from_csv

CAM_COLORS = [(255.0, 0.0, 0.0), (0.0, 0.0, 255.0)]  # GT=red, Pred=blue

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def to_pose(R, t):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = -R.T @ t
    return pose

def RT2pose(R, t):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

def get_csv_name(csv_path):
    """Extract csv filename without extension"""
    return Path(csv_path).stem

def export_scene(glb_path, poses_gt, imgs_gt, poses_pred, imgs_pred, scene=None):
    scene = trimesh.Scene()
    camera_scale = 0.2

    # 直接使用提供的 poses_pred，不再估计变换
    poses_pred_aligned = poses_pred

    # Add cameras to scene
    for pose, img_path in zip(poses_gt, imgs_gt):
        if pose is not None:
            img = imageio.imread(img_path)
            add_scene_cam(scene, pose, CAM_COLORS[0], img, screen_width=camera_scale)

    for pose, img_path in zip(poses_pred_aligned, imgs_pred):
        if pose is not None:
            img = imageio.imread(img_path)
            add_scene_cam(scene, pose, CAM_COLORS[1], img, screen_width=camera_scale)

    # Apply view transform to make GT cameras look forward
    if len(poses_gt) > 0 and poses_gt[0] is not None:
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(poses_gt[0] @ OPENGL @ rot))

    scene.export(glb_path, file_type="glb", include_normals=True)
    print(f"✅ Scene exported to: {glb_path}")
    

def add_scene_cam(scene, pose_c2w, color_rgb, image, focal=800, imsize=None, screen_width=0.2):
    edge_color = color_rgb
    marker = None
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        if focal.shape == ():
            focal = focal.item()
        else:
            focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = edge_color
        scene.add_geometry(marker)

def get_gt_scene_name(gt_data, dataset, img_name_prefix):
    metadata = gt_data[dataset]
    for scene_name, scene_data in metadata.items():
        if len(scene_data) == 0:
            continue
        for img_name in scene_data.keys():
            if img_name.startswith(img_name_prefix):
                return scene_name
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--output_dir", default="./output/")
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img_prefix", type=str, required=True)
    parser.add_argument("--thresholds_csv", type=str, default=None)
    args = parser.parse_args()

    from kaggle_utils.metric import mAA_on_cameras
    skip_top_thresholds = 2
    to_dec = 3

    gt_data = read_csv(args.gt_csv)
    user_data = read_csv(args.pred_csv)
    thresholds_data, _ = tth_from_csv(args.thresholds_csv)

    # Run score_transf
    _, _, dataset_transf = score_transf(
        gt_csv=args.gt_csv,
        user_csv=args.pred_csv,
        thresholds_csv=args.thresholds_csv,
        verbose=True
    )
    
    gt_scene_name = get_gt_scene_name(gt_data, args.dataset, args.img_prefix)
    user_scene_name = get_gt_scene_name(user_data, args.dataset, args.img_prefix)

    # 查找当前变换
    transf = dataset_transf.get(args.dataset, {}).get(gt_scene_name, None)
    if transf is None:
        raise ValueError(f"No transformation found for dataset={args.dataset}, scene={gt_scene_name}")

    R, t = transf
    pose_transform = RT2pose(R, t)

    # 图像路径及相机姿态
    imgs_gt, poses_gt = [], []
    imgs_pred, poses_pred_aligned = [], []
    poses_pred_raw = []

    # GT poses
    for img_name, cam in gt_data[args.dataset][gt_scene_name].items():
        img_path = os.path.join(args.img_root, 'train' if args.is_train else 'test', args.dataset, img_name)
        imgs_gt.append(img_path)

        R_w2c = np.array(cam["R"])
        t_w2c = np.array(cam["t"])
        
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        pose = np.eye(4)
        pose[:3, :3] = R_c2w
        pose[:3, 3] = t_c2w
        poses_gt.append(pose)

    # Predicted poses
    for img_name, cam in user_data[args.dataset][user_scene_name].items():
        img_path = os.path.join(args.img_root, 'train' if args.is_train else 'test', args.dataset, img_name)
        imgs_pred.append(img_path)

        R_w2c = np.array(cam["R"])
        t_w2c = np.array(cam["t"])
        
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        pose = np.eye(4)
        pose[:3, :3] = R_c2w
        pose[:3, 3] = t_c2w
        poses_pred_raw.append(pose)

        pose_aligned = pose_transform @ pose
        poses_pred_aligned.append(pose_aligned)
        
    pred_name = get_csv_name(args.pred_csv)
    gt_name = get_csv_name(args.gt_csv)
    output_subdir = f"{pred_name}-{gt_name}-{args.dataset}-{args.img_prefix}"
    output_path = os.path.join(args.output_dir, output_subdir)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    else:
        print(f"Output directory {output_path} already exists. Please remove it or choose a different name.")
    
    output_path_raw = os.path.join(output_path, f"{args.dataset}_{args.img_prefix}_raw.glb")
    output_path_aligned = os.path.join(output_path, f"{args.dataset}_{args.img_prefix}_aligned.glb")

    export_scene(
        output_path_raw,
        poses_gt,
        imgs_gt,
        poses_pred_raw,
        imgs_pred,
    )
    
    export_scene(
        output_path_aligned,
        poses_gt,
        imgs_gt,
        poses_pred_aligned,
        imgs_pred,
    )

    # ✅ ➕ 计算当前 img_prefix 的 mAA
    centers_gt = np.array([pose[:3, 3] for pose in poses_gt])
    centers_pred = np.array([pose[:3, 3] for pose in poses_pred_aligned])
    ths = thresholds_data[args.dataset][gt_scene_name]

    err = np.linalg.norm(centers_pred - centers_gt, axis=1, keepdims=True)
    err = np.repeat(err, len(ths), axis=1)  # shape=(N, len(ths))
    
    # ✅ 打印每张图像的得分表格
    # ➤ 找出被剔除的图像（仅用于标注）
    sorted_indices = np.argsort(err[:, 0] if err.ndim == 2 else err)
    removed_indices = set(sorted_indices[:to_dec])

    print("\n📊 Per-image registration table:")
    ths_str = [f"{t:.2f}" for t in ths]
    header = "| Image Name".ljust(26) + "|" + "".join([f" {t:>5} " for t in ths_str]) + "| Error  | mAA (%) | Removed?"
    print(header)
    print("-" * len(header))

    # 按图像名排序
    image_err_pairs = sorted(enumerate(zip(imgs_gt, err)), key=lambda x: os.path.basename(x[1][0]))

    for idx, (path, e) in image_err_pairs:
        name = os.path.basename(path)
        accs = [(ei <= t) for ei, t in zip(e, ths)]
        acc_str = "".join([f"  {'✅ ' if ok else '❌ '}  " for ok in accs])
        abs_err = float(e[0])
        acc_ratio = sum(accs) / len(ths) * 100
        flag = "✅ Removed" if idx in removed_indices else ""
        print(f"| {name:<24}|{acc_str}| {abs_err:6.2f} | {acc_ratio:6.2f} | {flag}")



    # 打印当前img_prefix的mAA
    mAA = mAA_on_cameras(err, ths, len(poses_gt), skip_top_thresholds=skip_top_thresholds, to_dec=to_dec)

    print(f"✅ Scene mAA for {args.dataset}/{gt_scene_name}: {mAA * 100:.2f}%")
    
    # 规模占比
    # 🧮 当前 scene 图像数量
    num_scene_images = len(poses_gt)

    # 🧮 当前 dataset 总图像数量
    num_total_images = sum([len(scene_data) for scene_data in gt_data[args.dataset].values()])
    scene_ratio = num_scene_images / num_total_images * 100

    # 🧮 当前 dataset 的整体 mAA（已由 score_transf 返回）
    dataset_mAA = dataset_transf[args.dataset].get("__avg__", None)
    if dataset_mAA is None:
        print("⚠️ Cannot find dataset-level mAA score.")
    else:
        relative_ratio = mAA / dataset_mAA * 100
        print(f"📊 Scene coverage in dataset: {num_scene_images}/{num_total_images} ({scene_ratio:.2f}%)")
        print(f"📈 Scene mAA vs dataset mAA: {mAA * 100:.2f}% / {dataset_mAA * 100:.2f}% ({relative_ratio:.2f}%)")
