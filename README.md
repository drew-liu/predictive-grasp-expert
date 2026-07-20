# Predictive Grasp Expert

## 1. 项目说明

该程序基于 ManiSkill 和 Panda 机械臂，用于模拟对“同时平移并绕 z 轴旋转”的目标方块进行预测式抓取。

当前版本属于**状态驱动专家基线**。程序直接读取仿真中的目标位置、姿态、线速度和角速度，预测未来接触位置与姿态，并控制机械臂完成接近、速度同步、角速度同步和夹爪闭合。

本程序使用 ManiSkill 默认场景，不需要额外背景图片或视频文件。运行时加上 `--render` 即可直接查看抓取过程。

---

## 2. 文件结构

```text
predictive_grasp_expert/
├── predictive_grasp_expert.py
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── environment.yml
```

- `predictive_grasp_expert.py`：核心仿真与控制程序。
- `README.md`：运行说明与后续任务说明。
- `requirements.txt`：核心 Python 依赖。
- `requirements-lock.txt`：本项目已验证版本的依赖。
- `environment.yml`：用于创建 Python 3.11 Conda 环境。

---

## 3. 当前程序包含的三种方法

通过 `--variant` 选择：

- `predictive_intercept`：预测未来接触位置和姿态，但不进行终端线速度与角速度同步。
- `linear_sync`：在预测接触的基础上同步目标平面线速度。
- `full_sync`：进一步同步目标绕 z 轴的角速度。

建议首先运行：

```text
full_sync
```

---

## 4. 创建环境

进入该文件夹后执行：

```bash
conda env create -f environment.yml
conda activate predictive_grasp
python -m pip install -r requirements-lock.txt
```

如已经有可用的 ManiSkill 环境，也可以直接在原环境中运行。

检查主要版本：

```bash
python - <<'PY'
import gymnasium
import mani_skill
import numpy
import sapien
import torch

print("gymnasium:", gymnasium.__version__)
print("mani_skill:", getattr(mani_skill, "__version__", "unknown"))
print("numpy:", numpy.__version__)
print("sapien:", getattr(sapien, "__version__", "unknown"))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

已验证的主要版本为：

```text
Python 3.11
gymnasium 0.29.1
mani-skill 3.0.0b22
numpy 2.4.6
sapien 3.0.2
torch 2.10.0+cu128
```

---

## 5. 基础检查

```bash
python -m py_compile predictive_grasp_expert.py

python predictive_grasp_expert.py --help >/dev/null   && echo "runtime import check passed"
```

---

## 6. 直接查看抓取过程

运行下面的已验证工况：

```bash
python predictive_grasp_expert.py   --render   --variant full_sync   --seed 0   --steps 500   --debug_every 20   --drift_speed 0.02   --spin_speed 0.05   --y_offset -0.30   --cube_z 0.14   --no-show_markers
```

正常情况下，程序会依次进入：

```text
observe
→ go_wait
→ wait_hold
→ launch
→ descend
→ close
→ post_grasp
→ done
```

最后应看到类似输出：

```text
final_phase=done
grasp_latched=1
```

---

## 7. 保存日志

```bash
mkdir -p logs results

python predictive_grasp_expert.py   --variant full_sync   --seed 0   --steps 500   --debug_every 20   --drift_speed 0.02   --spin_speed 0.05   --y_offset -0.30   --cube_z 0.14   --no-show_markers   --log_csv logs/full_sync_test.csv   --summary_csv results/full_sync_test_summary.csv
```

---

## 8. 当前实现说明

当前版本已经修正夹爪抓取面朝向：

- 夹爪 yaw 使用刚性 `panda_hand` 局部轴在世界 XY 平面的投影；
- 不再使用左右 finger link 原点连线估计平面朝向；
- 方块面方向按照 180° 周期处理；
- 角速度直接读取 `panda_hand.angular_velocity[2]`。

因此，夹爪应对齐方块的面，而不是斜着抓取对角线。

---

## 9. 当前限制

当前程序仍然是固定方向基线：

- 目标初始位置主要由 `y_offset` 设置；
- 目标沿世界坐标系 `+Y` 方向平移；
- 目标绕世界坐标系 z 轴旋转；
- 动态阶段包含针对 y 方向调节的速度伺服；
- 当前尚未支持平面内任意方向来向。

这些限制是下一步需要扩展的主要内容。

---

## 10. 第一阶段开发任务：随机来向

请先不要修改原文件，建议复制为：

```bash
cp predictive_grasp_expert.py predictive_grasp_random_direction.py
```

然后按以下顺序扩展。

### 第一步：复现当前基线

先确认原程序能够稳定运行，并理解：

- 环境初始化；
- 目标平移与旋转；
- 未来接触状态预测；
- `full_sync` 的线速度和角速度同步；
- 各控制阶段的切换逻辑。

### 第二步：支持可指定的平面运动方向

增加类似参数：

```text
--motion_angle_deg
```

将当前只有 y 分量的目标速度扩展为：

```text
vx = speed × cos(theta)
vy = speed × sin(theta)
```

先测试八个固定方向：

```text
0°、45°、90°、135°、180°、225°、270°、315°
```

### 第三步：合理设置初始位置

不要独立随机初始位置和运动方向。

建议先确定一个可抓取区域中心，再根据运动速度和预留时间反推出目标初始位置，使不同来向的轨迹都经过可抓取区域。

### 第四步：推广速度同步控制

当前程序中仍有针对 y 方向的速度控制，需要推广到平面 XY：

- 可以直接分别控制 x、y 速度；
- 或者在“沿目标运动方向 / 垂直运动方向”坐标系下设计控制。

不能只修改目标速度初始化而保留 y 方向专项控制。

### 第五步：加入随机模式

八个固定方向均能稳定运行后，再加入由随机种子控制的连续随机方向。

建议第一轮只随机来向，暂时固定：

- 线速度大小；
- 角速度大小；
- 目标高度；
- 方块尺寸；
- 其他控制参数。

---

## 11. 第一阶段建议交付内容

完成随机方向扩展后，建议提供：

1. 修改后的程序；
2. 八个固定来向的测试结果；
3. 各工况的运动方向、随机种子和成功情况；
4. 说明原程序中哪些部分依赖固定 `+Y` 方向；
5. 连续随机方向的初步测试结果。

---

## 12. 常见提示

运行时可能出现：

```text
pkg_resources is deprecated
```

这是当前 SAPIEN 依赖产生的警告，不影响本程序运行。

如果打开 Marker，ManiSkill 也可能提示 Marker 没有初始位姿。使用：

```text
--no-show_markers
```

可以关闭 Marker，并避免相关提示。
