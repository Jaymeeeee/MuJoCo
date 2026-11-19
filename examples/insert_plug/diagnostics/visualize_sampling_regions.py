"""Render proposed InD XY regions in the unchanged initial MuJoCo scene.

Usage: python -m examples.insert_plug.diagnostics.visualize_sampling_regions
Annotations exist only in the rendering scene; no physics steps are taken.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..environment.scene import DEFAULT_CONFIG, _camera, build_scene, load_config


REGIONS = {
    'source': dict(x=[0.41, 0.49], y=[-0.11, -0.03], rgb=[245, 145, 35]),
    'target': dict(x=[0.41, 0.49], y=[0.03, 0.11], rgb=[30, 190, 145]),
}
TITLES = {'head': '固定第三视角 · Head', 'top': '俯视 · 世界 X 向右 / Y 向上',
          'oblique': '斜视 · 区域与支座的位置关系'}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def corners(region, z):
    x0, x1 = region['x']
    y0, y1 = region['y']
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])


def project(model, data, name, points, width):
    camera = model.camera(name).id
    local = (np.asarray(points) - data.cam_xpos[camera]) @ data.cam_xmat[camera].reshape(3, 3)
    depth = -local[:, 2]
    focal = width / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
    pixels = np.c_[focal * local[:, 0] / depth + (width - 1) / 2,
                   -focal * local[:, 1] / depth + (width - 1) / 2]
    return pixels, depth


def annotations(mj, scene, z):
    def geom(kind, size, pos, rgb, alpha=1):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Rendering annotation capacity exceeded')
        item = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(item, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                       np.eye(3).ravel(), np.array([*(np.asarray(rgb) / 255), alpha], dtype=np.float32))
        item.emission = 0.45
        item.specular = 0
        item.category = mj.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
        return item

    def line(a, b, rgb, width=1.4, arrow=False):
        kind = mj.mjtGeom.mjGEOM_ARROW if arrow else mj.mjtGeom.mjGEOM_LINE
        item = geom(kind, [0, 0, 0], [0, 0, 0], rgb)
        mj.mjv_connector(item, kind, width, np.asarray(a), np.asarray(b))

    for region in REGIONS.values():
        x0, x1 = region['x']
        y0, y1 = region['y']
        rgb = region['rgb']
        geom(mj.mjtGeom.mjGEOM_BOX, [.04, .04, .0001],
             [(x0 + x1) / 2, (y0 + y1) / 2, z - .0003], rgb, .18)
        pts = corners(region, z)
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            line(a, b, rgb, 4)
        for t in np.linspace(0, 1, 11)[1:-1]:
            line([x0 + .08 * t, y0, z], [x0 + .08 * t, y1, z], rgb)
            line([x0, y0 + .08 * t, z], [x1, y0 + .08 * t, z], rgb)
    origin = np.array([.375, -.135, z + .0005])
    line(origin, origin + [.03, 0, 0], [230, 65, 65], .001, arrow=True)
    line(origin, origin + [0, .03, 0], [50, 115, 240], .001, arrow=True)
    return origin


def font(size):
    return ImageFont.truetype(FONT, size)


def sheet(images, output, size):
    gap, margin, header, panel_title, footer = 20, 28, 162, 58, 155
    width = 2 * margin + len(images) * size + (len(images) - 1) * gap
    canvas = Image.new('RGB', (width, header + panel_title + size + footer), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), 'InD 位置范围 · 两个 8 × 8 cm 正方形', font=font(34), fill='#152437')
    for i, (name, label) in enumerate((('source', '初始位置'), ('target', '目标位置'))):
        region = REGIONS[name]
        y = 78 + i * 35
        draw.rounded_rectangle((margin, y + 5, margin + 22, y + 27), radius=4, fill=tuple(region['rgb']))
        text = f"{label}    X ∈ [0.41, 0.49] m     Y ∈ [{region['y'][0]:.2f}, {region['y'][1]:.2f}] m"
        draw.text((margin + 36, y), text, font=font(23), fill='#263b50')
    for i, (name, picture) in enumerate(images.items()):
        x = margin + i * (size + gap)
        draw.text((x + 8, header + 10), TITLES[name], font=font(24), fill='#263b50')
        canvas.paste(picture.resize((size, size), Image.Resampling.LANCZOS), (x, header + panel_title))
    y = header + panel_title + size + 20
    for text in ('彩色区域：孔位中心 XY 范围在桌面上的投影；不是支座外轮廓。',
                 '每区 10 × 10 格，每格 8 mm；两区域最近边缘相距 6 cm。',
                 '当前支座位于区域中心。网格仅作范围示意，尚未生成采样位置对。'):
        draw.text((margin, y), text, font=font(23), fill='#435468')
        y += 37
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backend', choices=['osmesa', 'egl'], default='osmesa')
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = args.backend
    if args.backend == 'osmesa':
        os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    import mujoco as mj

    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    output = (args.output or Path('outputs/insert_plug/previews') /
              f"insert_plug_ind_regions_{started:%Y%m%d_%H%M%S}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    width = 1024
    config_hash = hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()
    config = copy.deepcopy(load_config())
    config['cameras']['width'] = config['cameras']['height'] = width
    z = float(config['table']['surface_z']) + .001
    images, reports = {}, {}
    with TemporaryDirectory(prefix='mujoco-ind-region-preview-') as temporary:
        scene_path = build_scene(config, Path(temporary) / 'scene.xml')
        tree = ET.parse(scene_path)
        world = tree.getroot().find('worldbody')
        _camera(world, 'region_top', dict(position=[.45, 0, .46], look_at=[.45, 0, 0], fovy=40))
        _camera(world, 'region_oblique', dict(position=[.77, -.36, .35], look_at=[.45, 0, .015], fovy=40))
        tree.write(scene_path, encoding='unicode')
        model = mj.MjModel.from_xml_path(str(scene_path))
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, model.key('home').id)
        mj.mj_forward(model, data)
        before = data.qpos.copy()
        renderer = mj.Renderer(model, height=width, width=width)
        try:
            for label, camera in [('head', 'third_person'), ('top', 'region_top'), ('oblique', 'region_oblique')]:
                renderer.update_scene(data, camera=camera)
                if label == 'head':
                    Image.fromarray(renderer.render().copy()).save(output / 'head_original.png')
                origin = annotations(mj, renderer.scene, z)
                img = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(img)
                for axis, direction, color in [('X', [.034, 0, 0], '#ed4949'), ('Y', [0, .034, 0], '#3073ef')]:
                    pixel, _ = project(model, data, camera, [origin + direction], width)
                    draw.text(tuple(pixel[0]), f'+{axis}', font=font(22), fill=color, anchor='mm',
                              stroke_width=2, stroke_fill='white')
                img.save(output / f'{label}_regions.png')
                images[label] = img
                reports[label] = {}
                for name, region in REGIONS.items():
                    pixels, depth = project(model, data, camera, corners(region, z), width)
                    inside = bool(np.all(depth > 0) and np.all(pixels >= 0) and np.all(pixels <= width - 1))
                    if not inside:
                        raise RuntimeError(f'{name} region exceeds {label} image')
                    reports[label][name] = dict(corners_pixels=pixels.tolist(), all_corners_in_frame=inside)
        finally:
            renderer.close()
        assert float(data.time) == 0.0 and np.array_equal(before, data.qpos)
        assert hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest() == config_hash
    sheet({'head': images['head']}, output / 'third_person_regions.png', 1024)
    sheet(images, output / 'three_views.png', 720)
    result = dict(started_at=started.isoformat(), output=str(output), regions=REGIONS,
                  region_plane_z_m=z, interpretation='table projection of allowed hole-center XY; not fixture footprint',
                  grid_cells_per_axis=10, grid_spacing_m=.008, samples_generated=False,
                  scene_config_sha256=config_hash, physics_steps=0, initial_state_unchanged=True,
                  backend=args.backend, image_size=[width, width], projections=reports,
                  visibility_scope='region corner frame containment; not occlusion or execution validation')
    (output / 'visualization.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), all_region_corners_in_frame=True, physics_steps=0)))


if __name__ == '__main__':
    main()
