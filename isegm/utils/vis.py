from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO

def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
 
def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(255, 0, 0), radius=4):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def vis_mask_on_image(image, mask, vis_trimap=False, mask_color=(255, 0, 0), trimap_color=(0, 255, 255)):
    mask = mask.astype(np.float32)
    mask_3 = np.repeat(mask[..., np.newaxis], 3, 2)

    color_mask = np.zeros_like(mask_3)
    color_mask[mask > 0] = mask_color

    fusion_mask = image * 0.3 + color_mask * 0.7
    fusion_mask = image * (1 - mask_3) + fusion_mask * mask_3

    if vis_trimap:
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fusion_mask = cv2.drawContours(fusion_mask, contours, -1, trimap_color, 2)

    return fusion_mask


def add_tag(image, tag='nodefined', tag_h=40):
    image = image.astype(np.uint8)
    H, W = image.shape[0], image.shape[1]
    tag_blanc = np.ones((tag_h, W, 3)).astype(np.uint8) * 255
    cv2.putText(tag_blanc, tag, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    image = cv2.vconcat([image, tag_blanc])
    return image


def vis_result_base(image, pred_mask, instances_mask, iou, num_clicks, clicks_list, prev_prediction, last_y,
                    last_x, gcns):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_color = (132, 52, 84)
    gt_color = (63, 196, 63)
    init_color = (200, 200, 200)
    trimap_color = (4, 172, 244)
    out_sz = [128, 128]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sz = int(np.sqrt(gcns[0]['pos_sim'].shape[-1]))
    shallow_pos_sim = gcns[0]['pos_sim'].view(-1, sz, sz)
    shallow_pos_sim = F.interpolate(shallow_pos_sim.unsqueeze(0), size=out_sz).squeeze()
    shallow_pos_sim = shallow_pos_sim[0].cpu().numpy()

    shallow_neg_sim = gcns[0]['neg_sim'].view(-1, sz, sz)
    shallow_neg_sim = F.interpolate(shallow_neg_sim.unsqueeze(0), size=out_sz).squeeze()
    shallow_neg_sim = shallow_neg_sim[0].cpu().numpy()

    deep_pos_sim = gcns[1]['pos_sim'].view(-1, sz, sz)
    deep_pos_sim = F.interpolate(deep_pos_sim.unsqueeze(0), size=out_sz).squeeze()
    deep_pos_sim = deep_pos_sim[0].cpu().numpy()

    deep_neg_sim = gcns[1]['neg_sim'].view(-1, sz, sz)
    deep_neg_sim = F.interpolate(deep_neg_sim.unsqueeze(0), size=out_sz).squeeze()
    deep_neg_sim = deep_neg_sim[0].cpu().numpy()

    shallow_att = gcns[0]['attn'].squeeze().mean(0)
    shallow_pos_att = torch.flip(shallow_att[:, :gcns[0]['adj_mat_p'].shape[0]].sum(-1).view(sz, sz), dims=[1])
    shallow_pos_att = F.interpolate(shallow_pos_att.unsqueeze(0).unsqueeze(0), size=out_sz).squeeze().cpu().numpy()
    shallow_neg_att = torch.flip(shallow_att[:, gcns[0]['adj_mat_p'].shape[0]:].sum(-1).view(sz, sz), dims=[1])
    shallow_neg_att = F.interpolate(shallow_neg_att.unsqueeze(0).unsqueeze(0), size=out_sz).squeeze().cpu().numpy()

    deep_att = gcns[1]['attn'].squeeze().mean(0)
    deep_pos_att = torch.flip(deep_att[:, :gcns[1]['adj_mat_p'].shape[0]].sum(-1).view(sz, sz), dims=[1])
    deep_pos_att = F.interpolate(deep_pos_att.unsqueeze(0).unsqueeze(0), size=out_sz).squeeze().cpu().numpy()
    deep_neg_att = torch.flip(deep_att[:, gcns[1]['adj_mat_p'].shape[0]:].sum(-1).view(sz, sz), dims=[1])
    deep_neg_att = F.interpolate(deep_neg_att.unsqueeze(0).unsqueeze(0), size=out_sz).squeeze().cpu().numpy()

    fusion_gt = vis_mask_on_image(image, instances_mask, vis_trimap=True, mask_color=gt_color,
                                  trimap_color=trimap_color)
    fusion_pred = vis_mask_on_image(image, pred_mask, vis_trimap=True, mask_color=gt_color,
                                  trimap_color=trimap_color)

    fusion_pred = cv2.resize(fusion_pred, out_sz)
    for i in range(len(clicks_list)):
        click_tuple = clicks_list[i]

        if click_tuple.is_positive:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        coord = click_tuple.coords
        x, y = coord[1], coord[0]
        # scale due to 128 resize operation
        x = int(x * 128 / image.shape[1])
        y = int(y * 128 / image.shape[0])
        if x < 0 or y < 0:
            continue
        cv2.circle(fusion_pred, (x, y), 2, color, -1)
        # cv2.putText(fusion_pred, f'clicks: {str(i+1)}', (x-10, y-10),  cv2.FONT_HERSHEY_COMPLEX, 1 , color,1 )
    if last_x != -1:
        cv2.circle(fusion_pred, (last_x, last_y), 2, (255, 255, 255), -1)

    data_array = [fusion_pred, shallow_pos_sim, shallow_neg_sim, deep_pos_sim, deep_neg_sim,
                  shallow_pos_att, shallow_neg_att, deep_pos_att, deep_neg_att]
    legends = ['Pred', 'Pos Sim S', 'Neg Sim S', 'Pos Sim D', 'Neg Sim D',
                  'Pos Att S', 'Neg Att S', 'Pos Att D', 'Neg Att D']
    fig, axs = plt.subplots(1, len(data_array), figsize=(40, 8), constrained_layout=True)  # 创建3行1列的子图布局
    for i in range(len(data_array)):
        if i <= 0:
            axs[i].imshow(data_array[i].astype(np.uint8))
            axs[i].text(15, 15, f'{len(clicks_list)} clicks, IoU={iou:1.2f}', fontsize=20, color='black', backgroundcolor='white', alpha=0.5)
        else:
            axs[i].imshow(data_array[i], cmap='plasma', interpolation='nearest')
        axs[i].set_title(f'{legends[i]}', fontsize=8)
        axs[i].axis('off')

    # plt.show()
    # 创建一个内存缓冲区
    buffer = BytesIO()
    # 保存图像到缓冲区，裁剪掉空白
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.01)
    buffer.seek(0)  # 将缓冲区的位置重置到开始处
    # 使用OpenCV从内存缓冲区读取图像
    image_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    out_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    return out_image
