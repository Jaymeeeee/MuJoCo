"""Render the two distinct attention-related diagnostics of an action-only DiT.

The Transformer matrix is action-token self-attention. The RGB overlays show
the RGB-D encoder's SpatialSoftmax pooling, which is not action-to-image
attention. This module only renders supplied arrays and never executes a model.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .visualize_attention import _colors, _overlay
from .visualize_diffusion_attention import keypoint_pixels


FRAME_SIZE = (1360, 1056)
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
WHITE, MUTED, ACCENT, HIGHLIGHT = '#f0f4fa', '#bac5d4', '#68dfd3', '#fff47b'


@lru_cache(maxsize=24)
def _font(size):
    return ImageFont.truetype(FONT_PATH, size)


def _probabilities(value, shape, label, *, row_stochastic=False):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f'{label} must be finite nonnegative probabilities with shape {shape}')
    sums = array.sum(axis=-1) if row_stochastic else array.sum()
    if not np.allclose(sums, 1., atol=2e-3, rtol=2e-3):
        raise ValueError(f'{label} probabilities must sum to one')
    return array


def _rgb(value, label):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f'{label} must be an HWC RGB uint8 image')
    return array


def _point_overlay(rgb, mean, points, grid_shape, scale):
    image = Image.fromarray(_overlay(rgb, mean, scale)).resize((320, 320), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Keypoints must be finite Kx2 arrays')
    if (np.abs(points) > 1.0001).any():
        raise ValueError('SpatialSoftmax keypoints must be normalized to [-1, 1]')
    for index, (x, y) in enumerate(keypoint_pixels(points, grid_shape, image.size)):
        color = tuple(_colors(np.asarray(index/max(1, len(points)-1))).astype(np.uint8))
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color, outline='white', width=1)
    return image


def _dashed_line(draw, start, stop, *, fill=MUTED, width=1, dash=7):
    x0, y0 = start
    x1, y1 = stop
    distance = max(abs(x1-x0), abs(y1-y0))
    if distance == 0:
        return
    for offset in range(0, int(distance), dash*2):
        end = min(offset+dash, distance)
        a = (x0+(x1-x0)*offset/distance, y0+(y1-y0)*offset/distance)
        b = (x0+(x1-x0)*end/distance, y0+(y1-y0)*end/distance)
        draw.line((a, b), fill=fill, width=width)


def _color_bar(canvas, draw, x, y, label, scale):
    gradient = np.repeat(_colors(np.linspace(0, 1, 180))[None], 12, axis=0).astype(np.uint8)
    canvas.paste(Image.fromarray(gradient), (x, y))
    draw.text((x, y+16), '0', font=_font(13), fill=MUTED)
    draw.text((x+146, y+16), f'{scale:g}', font=_font(13), fill=MUTED)
    draw.text((x+194, y-7), label, font=_font(15), fill=MUTED)


def render_frame(*, live_images, source_images, spatial_maps, keypoints, grid_shapes,
                 action_attention, action_offset, time_s, source_time_s, chunk_index,
                 spatial_color_max=.05, action_color_max=.05):
    """Return a 1360x1056 RGB frame with fixed, separately declared scales.

    ``action_attention[q, k]`` is the head-averaged last-block attention at
    diffusion timestep zero. Every row sums to one. Only queries 0..31 are
    executed from this 64-action forecast, and the highlighted row follows the
    action queue. Spatial overlays always use ``source_images``, not live views.
    Neither display rescales its colors from individual frames or camera data.
    """
    keys = tuple(spatial_maps)
    if len(keys) != 2 or {key.rsplit('.', 1)[-1] for key in keys} != {'head', 'wrist'}:
        raise ValueError('Expected exactly the head and wrist camera keys')
    keys = tuple(sorted(keys, key=lambda key: key.rsplit('.', 1)[-1] != 'head'))
    if any(set(value) != set(keys) for value in (source_images, keypoints, grid_shapes)):
        raise ValueError('All spatial camera dictionaries must use identical keys')
    if not set(keys).issubset(live_images):
        raise ValueError('Missing live camera images')
    if type(action_offset) is not int or not 0 <= action_offset < 32:
        raise ValueError('Executed cached action offset must be in 0..31')
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError('chunk_index must be a nonnegative integer')
    if any(not math.isfinite(x) for x in (time_s, source_time_s, spatial_color_max, action_color_max)):
        raise ValueError('Display time and probability scales must be finite')
    if source_time_s < 0 or time_s < source_time_s or min(spatial_color_max, action_color_max) <= 0:
        raise ValueError('Invalid source time or color scale')
    attention = _probabilities(action_attention, (64, 64), 'Action attention', row_stochastic=True)
    canvas = Image.new('RGB', FRAME_SIZE, '#121923')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), 'DiT Diffusion · OOD · 动作注意力与视觉特征', font=_font(29), fill=WHITE)
    draw.text((20, 53), f'仿真 {time_s:5.1f} s    预测段 {chunk_index:02d}    当前执行 q={action_offset:02d} / 31',
              font=_font(21), fill=ACCENT)
    draw.line((702, 92, 702, 962), fill='#354051', width=2)

    for column, key in enumerate(keys):
        x = 20+336*column
        name = '头部相机' if key.rsplit('.', 1)[-1] == 'head' else '腕部相机'
        live = _rgb(live_images[key], f'Live {key}')
        source = _rgb(source_images[key], f'Source {key}')
        grid_shape = tuple(grid_shapes[key])
        if len(grid_shape) != 2 or min(grid_shape) < 1:
            raise ValueError('Spatial grids must be positive HxW shapes')
        mean = _probabilities(spatial_maps[key], grid_shape, f'Spatial pooling {key}')
        draw.text((x, 91), f'实时画面 · {name}', font=_font(19), fill=WHITE)
        canvas.paste(Image.fromarray(live).resize((320, 320), Image.Resampling.BILINEAR), (x, 120))
        draw.text((x, 452), f'预测源图＋空间汇聚 · {name}', font=_font(18), fill=ACCENT)
        canvas.paste(_point_overlay(source, mean, keypoints[key], grid_shape, spatial_color_max), (x, 485))
        draw.text((x, 817), f'{grid_shape[0]}×{grid_shape[1]}，{len(keypoints[key])} 个特征点等权平均',
                  font=_font(16), fill=WHITE)
        draw.text((x, 844), f'max {float(mean.max()):.4f} · 截断 {int((mean>spatial_color_max).sum())}/{mean.size} 格',
                  font=_font(15), fill=MUTED)
    draw.text((20, 880), f'预测源图：{source_time_s:.1f} s    图像年龄：{time_s-source_time_s:.1f} s',
              font=_font(18), fill=ACCENT)
    draw.text((20, 908), '缓存动作沿用该预测段的源图；点是特征期望位置。', font=_font(17), fill=MUTED)
    _color_bar(canvas, draw, 20, 942, '空间汇聚概率 · 固定色标', spatial_color_max)

    draw.text((730, 91), 'Transformer 动作 → 动作自注意力', font=_font(22), fill=WHITE)
    draw.text((730, 126), '第 6 层 · 8 头均值 · 最后去噪步 τ=0', font=_font(18), fill=ACCENT)
    mx, my, cell = 774, 171, 8
    matrix = Image.fromarray(_colors(np.clip(attention/action_color_max, 0, 1)).astype(np.uint8))
    canvas.paste(matrix.resize((512, 512), Image.Resampling.NEAREST), (mx, my))
    draw.rectangle((mx-1, my-1, mx+512, my+512), outline='#8994a4', width=1)
    _dashed_line(draw, (mx+32*cell, my), (mx+32*cell, my+512), fill='#d5dce5', width=2)
    _dashed_line(draw, (mx, my+32*cell), (mx+512, my+32*cell), fill='#d5dce5', width=2)
    qy = my+action_offset*cell
    draw.rectangle((mx-2, qy, mx+513, qy+cell-1), outline=HIGHLIGHT, width=2)
    draw.polygon([(mx-11,qy+4),(mx-18,qy),(mx-18,qy+8)], fill=HIGHLIGHT)
    for token in (0, 16, 32, 48, 63):
        px = mx+token*cell+cell/2
        py = my+token*cell+cell/2
        draw.text((px, my+517), str(token), font=_font(13), fill=MUTED, anchor='mt')
        draw.text((mx-24, py), str(token), font=_font(13), fill=MUTED, anchor='rm')
    draw.text((718, 154), 'q', font=_font(16), fill=MUTED)
    draw.text((mx+190, 713), '被关注的动作 token k', font=_font(16), fill=MUTED)
    draw.text((730, 742), f'q={action_offset:02d} 的权重：max {float(attention[action_offset].max()):.4f}；均匀值 1/64≈0.0156',
              font=_font(16), fill=WHITE)
    bx, by, height = mx, 779, 100
    draw.rectangle((bx, by, bx+512, by+height), fill='#1d2836')
    uniform_y = by+height-height*min(1., (1/64)/action_color_max)
    _dashed_line(draw, (bx, uniform_y), (bx+512, uniform_y), fill='#687587', width=1)
    for k, value in enumerate(attention[action_offset]):
        level = min(1., float(value)/action_color_max)
        color = tuple(_colors(np.asarray(level)).astype(np.uint8))
        draw.rectangle((bx+k*8+1, by+height-height*level, bx+(k+1)*8-1, by+height), fill=color)
    draw.line((bx,by+height,bx+512,by+height), fill=MUTED)
    _dashed_line(draw,(bx+32*8,by),(bx+32*8,by+height),fill=WHITE,width=2)
    draw.text((730, 890), '虚线 32：本段执行 q0–31；q32–63 仅预测', font=_font(16), fill=MUTED)
    draw.text((730, 916), f'矩阵 max {float(attention.max()):.4f} · 显示截断 {int((attention>action_color_max).sum())}/4096 格',
              font=_font(15), fill=MUTED)
    _color_bar(canvas, draw, 774, 945, '动作注意力 · 固定色标', action_color_max)

    draw.line((20, 986, 1340, 986), fill='#354051', width=1)
    draw.text((20, 997), '左下：RGB-D 编码器的空间汇聚。右侧：动作 token 间的自注意力；此模型没有动作到图像的交叉注意力。',
              font=_font(17), fill=WHITE)
    draw.text((20, 1027), '热区与权重展示模型内部计算，不能单凭这些图断言某个像素导致了某个动作；不同相机的总概率不可比较。',
              font=_font(15), fill=MUTED)
    return np.asarray(canvas)


def _effect_map(value, label):
    """Effect magnitudes retain physical units and need not sum to one."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f'{label} must be a finite, nonnegative HxW effect map')
    return array


def _effect_overlay(rgb, values, scale):
    """Render coarse intervention cells as blocks, without pixel-level smoothing."""
    image = Image.fromarray(rgb).resize((320, 320), Image.Resampling.BILINEAR)
    grid = np.asarray(Image.fromarray(values).resize((320, 320), Image.Resampling.NEAREST))
    level = np.clip(grid/scale, 0, 1)
    alpha = (.70*np.sqrt(level))[...,None]
    rgb_array = np.asarray(image)
    mixed = np.round(rgb_array*(1-alpha)+_colors(level)*alpha).astype(np.uint8)
    image = Image.fromarray(mixed)
    draw = ImageDraw.Draw(image)
    rows, columns = values.shape
    for column in range(1, columns):
        x = round(column*320/columns)
        draw.line((x,0,x,319), fill='#bac5d4', width=1)
    for row in range(1, rows):
        y = round(row*320/rows)
        draw.line((0,y,319,y), fill='#bac5d4', width=1)
    return image


def render_sensitivity_frame(*, live_images, source_images, arm_maps, gripper_maps,
                             arm_vmax_deg, gripper_vmax_mm, time_s, source_time_s,
                             chunk_index, intervention_label='局部遮挡；配对使用同一采样噪声'):
    """Render fixed-state image-region intervention effects in physical units.

    Maps describe the first 32 predicted actions as a whole, not the current
    cached action. ``arm_maps`` are joint-angle RMS changes in degrees and
    ``gripper_maps`` are aperture RMS changes in millimetres. The caller chooses
    one fixed ``arm_vmax_deg`` and one fixed ``gripper_vmax_mm`` for the entire
    video and both cameras. The two physical quantities are never normalized
    together. Every heatmap cell is displayed as its original coarse patch.
    """
    keys = tuple(arm_maps)
    if len(keys) != 2 or {key.rsplit('.',1)[-1] for key in keys} != {'head','wrist'}:
        raise ValueError('Expected exactly the head and wrist camera keys')
    keys = tuple(sorted(keys,key=lambda key:key.rsplit('.',1)[-1]!='head'))
    if any(set(value)!=set(keys) for value in (source_images,gripper_maps)):
        raise ValueError('All intervention camera dictionaries must use identical keys')
    if not set(keys).issubset(live_images):
        raise ValueError('Missing live camera images')
    if type(chunk_index) is not int or chunk_index<0:
        raise ValueError('chunk_index must be a nonnegative integer')
    if any(not math.isfinite(value) for value in (time_s,source_time_s,arm_vmax_deg,gripper_vmax_mm)):
        raise ValueError('Display times and effect scales must be finite')
    if source_time_s<0 or time_s<source_time_s or min(arm_vmax_deg,gripper_vmax_mm)<=0:
        raise ValueError('Invalid source time or effect scale')
    if not isinstance(intervention_label,str) or not intervention_label:
        raise ValueError('intervention_label must describe the actual intervention')
    source = {key:_rgb(source_images[key],f'Source {key}') for key in keys}
    live = {key:_rgb(live_images[key],f'Live {key}') for key in keys}
    arm = {key:_effect_map(arm_maps[key],f'Arm {key}') for key in keys}
    gripper = {key:_effect_map(gripper_maps[key],f'Gripper {key}') for key in keys}
    if any(arm[key].shape!=gripper[key].shape for key in keys):
        raise ValueError('Arm and gripper intervention maps must use the same patch grid')

    canvas=Image.new('RGB',FRAME_SIZE,'#121923')
    draw=ImageDraw.Draw(canvas)
    draw.text((20,10),'DiT Diffusion · OOD · 图像区域扰动对动作预测的影响',font=_font(28),fill=WHITE)
    draw.text((20,52),f'仿真 {time_s:5.1f} s    预测段 {chunk_index:02d}    源图 {source_time_s:.1f} s    图像年龄 {time_s-source_time_s:.1f} s',
              font=_font(21),fill=ACCENT)
    for index,key in enumerate(keys):
        x=20+272*index
        name='头部相机' if key.rsplit('.',1)[-1]=='head' else '腕部相机'
        draw.text((x,88),f'实时画面 · {name}',font=_font(18),fill=WHITE)
        canvas.paste(Image.fromarray(live[key]).resize((256,256),Image.Resampling.BILINEAR),(x,118))
    panel_x=590
    draw.text((panel_x,108),'固定状态下的遮挡干预',font=_font(25),fill=ACCENT)
    lines=[
        '比较同一状态下，遮挡前后的前 32 步预测动作。',
        '每格对应一块被遮挡区域，保持其余输入与噪声一致。',
        '更亮：遮挡该区域后，动作预测变化更大。',
        '热图覆盖整段 32 步预测，缓存执行时沿用源图。',
    ]
    for index,line in enumerate(lines):
        draw.text((panel_x,155+35*index),line,font=_font(18),fill=WHITE if index==2 else MUTED)
    label_font=_font(17)
    if draw.textbbox((0,0),intervention_label,font=label_font)[2]>738:
        raise ValueError('intervention_label is too long for the visualization panel')
    draw.text((panel_x,310),intervention_label,font=label_font,fill=MUTED)
    draw.text((panel_x,342),'单位分别为角度和毫米；两个指标各用自己的固定色标。',font=_font(17),fill=MUTED)

    draw.line((20,393,1340,393),fill='#354051',width=1)
    draw.text((20,406),'关节角度预测变化 · RMS（°）',font=_font(22),fill=WHITE)
    draw.text((692,406),'夹爪开度预测变化 · RMS（mm）',font=_font(22),fill=WHITE)
    for metric_index,(maps,scale,unit) in enumerate(((arm,arm_vmax_deg,'°'),(gripper,gripper_vmax_mm,'mm'))):
        for camera_index,key in enumerate(keys):
            x=20+metric_index*672+camera_index*336
            name='头部相机' if key.rsplit('.',1)[-1]=='head' else '腕部相机'
            values=maps[key]
            draw.text((x,445),f'预测源图 · {name}',font=_font(18),fill=ACCENT)
            canvas.paste(_effect_overlay(source[key],values,scale),(x,477))
            draw.text((x,811),f'{values.shape[0]}×{values.shape[1]} 遮挡区域 · max {float(values.max()):.3f} {unit}',
                      font=_font(15),fill=WHITE)
            draw.text((x,838),f'区域均值 {float(values.mean()):.3f} {unit} · 截断 {int((values>scale).sum())}/{values.size} 格',
                      font=_font(15),fill=MUTED)
        bx=20+metric_index*672
        _color_bar(canvas,draw,bx,887,f'固定范围 0–{scale:g} {unit}',scale)
    draw.line((20,937,1340,937),fill='#354051',width=1)
    draw.text((20,950),'这是区域遮挡对固定状态预测的影响，不是注意力概率；粗网格不能说明格内某一个像素的贡献。',font=_font(18),fill=WHITE)
    draw.text((20,984),'关节 RMS 汇总前 32 步各关节变化；夹爪 RMS 单独汇总开度变化。色标在整段视频与两相机之间保持一致。',font=_font(16),fill=MUTED)
    draw.text((20,1016),'颜色只表示变化幅度，不表示改善或恶化；没有执行遮挡后的动作，也不能直接推断抓取或插入成功率。',font=_font(16),fill=MUTED)
    return np.asarray(canvas)
