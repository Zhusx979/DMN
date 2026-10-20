"""Multi-class segmentation overlay visualization."""

import cv2
import numpy as np


def vis_save_nii_slice(original_slice, pred_slice, save_path):
    """
    可视化单个nii切片（五分类：背景0 + 4个前景类1~4）
    只在四个前景类上叠加颜色，背景保持原图灰度。

    参数:
    - original_slice: numpy array, 形状 (H, W) 或 (H, W, 1)，原始灰度切片
    - pred_slice: numpy array, 形状 (H, W)，整数标签 0~4
    - save_path: str，保存路径（如 'vis_result.png'）
    """
    # 颜色定义（BGR格式）：1蓝(主动脉)、2绿(胆囊)、3红(左肾)、4青(右肾)
    colors = {
        1: [30, 144, 255],  # blue
        2: [0, 255, 0],  # green
        3: [255, 0, 0],  # red
        4: [0, 255, 255]  # cyan
    }

    # 统一处理输入形状：如果有通道维度，去掉
    if original_slice.ndim == 3 and original_slice.shape[-1] == 1:
        original_slice = original_slice.squeeze(-1)
    if original_slice.ndim != 2:
        raise ValueError("original_slice 应为 (H, W) 或 (H, W, 1) 的灰度图")

    if pred_slice.ndim != 2 or pred_slice.shape != original_slice.shape:
        raise ValueError("pred_slice 必须是 (H, W) 且与 original_slice 形状匹配")

    # 归一化到0-255并转为uint8
    img = original_slice.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 防零除
    img = (img * 255.0).astype(np.uint8)

    # 转为三通道BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 只对前景类1~4叠加颜色，背景(0)保持原图
    mask = np.zeros_like(img_bgr)
    for label, color in colors.items():
        class_mask = (pred_slice == label)
        mask[class_mask] = color

    # 融合：前景位置替换为颜色，背景保留原图
    vis_img = np.where(pred_slice[..., np.newaxis] >= 1, mask, img_bgr)

    # BGR转RGB（OpenCV保存时默认BGR，但多数查看器期望RGB，这里转成RGB保存）
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    # 保存
    success = cv2.imwrite(save_path, vis_img_rgb)
    if not success:
        print(f"保存失败: {save_path}")
    else:
        print(f"已保存可视化结果: {save_path}")