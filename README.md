# MuJoCo 插销仿真

Franka FR3 使用 Robotiq 2F-85 夹爪，将螺栓从起始孔取出并插入目标孔。以下命令在 MuJoCo 仓库根目录运行。

## 运行场景

首次使用时，初始化子模块并安装环境。

```bash
git submodule update --init --recursive
bash scripts/setup_mujoco_env.sh
```

日常使用时激活环境，按需选择一条场景命令。

```bash
conda activate mujoco
insert-plug view  # 查看静态场景
insert-plug demo --backend glfw --viewer --realtime --no-images  # 在窗口观看专家演示
insert-plug snapshot --backend egl  # 无窗口截图，保存到 outputs/insert_plug/scene_preview/
```

## 采集与保存

```bash
insert-plug collect --backend egl --episodes 10
```

默认保存到仓库根目录的 **`datasets/insert_plug/run_<时间戳>/`**。每条轨迹各占一个子目录，如 `episode_000000/`，内含同名的 HDF5 观测与动作、YAML 配置、XML 场景和第三视角 MP4。

按预设网格批量采集时，下面的命令在每个位置记录两条轨迹。

```bash
python -m examples.insert_plug.data.collect_positions \
  --episodes-per-position 2 --workers 3 --output datasets/insert_plug/grid_run01
```

## 配置与常用参数

| 配置文件 | 用途 |
| --- | --- |
| [config/insert_plug.yaml](config/insert_plug.yaml) | 场景、起始孔和目标孔位置、相机、控制与采样频率 |
| [config/insert_plug_positions.yaml](config/insert_plug_positions.yaml) | 固定网格布局，批量采集时可用 `--positions-config` 切换 |
| [config/insert_plug_positions_v2.yaml](config/insert_plug_positions_v2.yaml) | 方环布局，供 v2 和力矩采集使用 |

| `demo` / `collect` 参数 | 用途 |
| --- | --- |
| `--config`、`--output` | 指定场景 YAML 和本次采集的批次目录 |
| `--episodes`、`--seed` | 轨迹条数和起始种子，种子也用作轨迹编号 |
| `--randomize-mm` | 两个孔位在 X、Y 方向的最大随机偏移量，单位毫米，默认 0 |
| `--backend` | `auto` 自动选择，`glfw` 用于桌面窗口，`egl` 或 `osmesa` 用于无窗口运行 |
| `--width`、`--sample-hz`、`--control-hz` | 设置相机图像边长（像素）、记录频率和控制频率 |
| `--video` | 额外生成首条轨迹的双视角预览，保存到 `outputs/insert_plug/previews/` |

运行 `insert-plug collect --help` 可查看完整参数。力矩采集和 LeRobot 转换示例见[插销任务说明](examples/insert_plug/README.md)。
