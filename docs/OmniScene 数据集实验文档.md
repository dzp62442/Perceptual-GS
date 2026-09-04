# OmniScene 数据集实验方案（Perceptual-GS / `comp_svfgs`）

## 1. 文档目的与本阶段边界

本文档描述如何在 Perceptual-GS 中接入 OmniScene，并以逐场景优化方式完成 Center150 对比实验。当前阶段只确定方案，不实现代码、不创建 Center150 子集、不启动训练。文档通过审阅后，再按本文实施。

本次适配遵循以下边界：

- Perceptual-GS 是逐场景优化方法；每个 OmniScene bin 独立视为一个场景，不能照搬 DepthSplat 把多个 bin 组成 batch 训练前馈网络的调用方式。
- Center150 清单只由 `~/Projects/SVF-GS` 生成。本项目只读取和严格校验 `bins_center150_v1.json`，不提供生成、重排或修复清单的逻辑。
- 每个场景固定使用 6 张中心帧环视图像训练，使用 12 张相邻时刻图像加 6 张输入图像，共 18 张图像评估。
- 第一版不加载动态物体掩码，不加载相对深度，不计算 PCC。
- Metric3D-v2 绝对尺度深度只用于构建初始化点云；置信度严格大于 `0.3` 且深度有限、严格大于 0 的像素才有效。
- Center150 的单场景训练不保存 checkpoint。完整场景直接跳过；不完整场景删除其结果目录并从第 0 次迭代重跑，但保留预处理缓存。
- 断点续跑采用“产物与实验协议严格、代码状态宽松”的判断方式：不计算或比对源码文件哈希，不要求 Git commit、分支或工作区 clean/dirty 状态一致。代码与 Git 信息最多作为提示性元数据记录，不作为阻止跳过或续跑的硬条件。
- 默认总优化次数为 10000，在 1000、5000、10000 次迭代评估。只缩短总预算，不重新缩放 Perceptual-GS 原有 densification 间隔、阈值或学习率日程，以免改变基线方法本身。

## 2. 已阅读实现与已核对事实

### 2.1 DepthSplat 的 OmniScene 实现

已对照以下文件：

- `~/Projects/depthsplat/docs/OmniScene数据集实验文档.md`
- `~/Projects/depthsplat/src/dataset/dataset_omniscene.py`
- `~/Projects/depthsplat/src/dataset/utils_omniscene.py`
- `~/Projects/depthsplat/config/dataset/omniscene.yaml`
- `~/Projects/depthsplat/config/experiment/omniscene_112x200.yaml`
- `~/Projects/depthsplat/config/experiment/omniscene_224x400.yaml`
- `~/Projects/depthsplat/src/main.py` 和 `src/model/model_wrapper.py`

DepthSplat 的 loader 将一个 bin 直接组织为：

- `context`：6 个中心帧相机，包含图像、逐图像内外参等；
- `target`：每个相机的第 1、2 个相邻帧，共 12 张，再拼接 6 张 context，合计 18 张；
- target 顺序固定为 12 张相邻视角在前、6 张输入视角在后；
- `load_conditions` 负责 `/datasets/nuScenes` 前缀替换、`samples/sweeps` 到缩小图像和参数目录的映射、resize 及内参同步缩放。

DepthSplat 在一次 forward 中预测深度、生成 Gaussian 并渲染 target，因此不需要磁盘场景目录和初始化 PLY。Perceptual-GS 没有这条前馈链路，需要把 loader 返回的一个 bin 先物化为可被 `Scene` 读取的独立场景。

### 2.2 优化式 Gaussian 参考实现

已重点对照：

- `~/Projects/DropGaussian_release/comp_svfgs/dataset_omniscene.py`
- `~/Projects/DropGaussian_release/scripts/run_omniscene.py`
- `~/Projects/DropGaussian_release/train.py`
- `~/Projects/Octree-GS/comp_svfgs/dataset_omniscene.py`
- `~/Projects/Octree-GS/comp_svfgs/omniscene_preprocess.py`
- `~/Projects/Octree-GS/run_omniscene.py`
- `~/Projects/Octree-GS/train.py`
- `~/Projects/SVF-GS/data/omniscene_dataset.py`
- `~/Projects/SVF-GS/data/transforms/loading.py`

其中可复用的成熟做法包括：

- 将一个 bin 转换为一个 NeRF/Blender 风格的磁盘场景；
- 使用 6 个 context 视角的 Metric3D 深度和置信度构建米制初始化点云；
- 保存逐帧内参，不能把六路相机压成一个全局 `camera_angle_x`；
- 在训练进程内完成里程碑渲染、指标计算和纯训练耗时记录；
- 用协议文件隔离不同参数的实验，用严格的多产物条件判断场景是否完成；
- Center150 不完整场景从头重跑，完整场景在加载 RGB/深度前快速跳过。

### 2.3 当前数据的只读检查结果

2026-09-04 对真实数据做了只读结构检查：

- `bins_center150_v1.json` 包含 150 个唯一 bin，且覆盖 150 个不同 scene token；
- 150 个 `bin_infos_3.2m/*.pkl` 均存在；
- 900 个 context 相机记录和对应的缩小图像、参数 JSON、Metric3D 深度与置信度文件均存在；
- 2700 个三帧相机记录所需的缩小图像和参数文件均存在；
- 缩放到 112×200 后，逐图像 `fx` 范围约为 104.96～164.52，`fy` 范围约为 105.50～176.77，说明不能使用共享焦距；
- 当前数据的主点位于图像中心，`cx=100`、`cy=56`（数值误差小于 `1e-13`），但实现仍应保留完整逐帧 K，不能把主点居中当成永久约束；
- `sensor2lidar_transform` 的旋转正交、行列式约为 1，齐次矩阵末行合法。

## 3. DepthSplat 与 Perceptual-GS 的调用差异

| 维度 | DepthSplat（前馈） | Perceptual-GS（逐场景优化） |
| --- | --- | --- |
| bin 的含义 | DataLoader 中的一个样本 | 一个独立重建场景 |
| 输入组织 | `context` 张量直接送入网络 | 6 张图写入 prepared scene，训练期间反复采样 |
| Gaussian 来源 | 网络一次前向预测 | 从输入点云初始化后迭代优化 |
| 深度作用 | 可作为模型条件或预测监督 | 第一版只用于生成初始化点云 |
| target 使用 | 同一次 forward 渲染监督/评估 | 不参与优化，只在里程碑作为 test cameras 渲染 |
| 相机接口 | 张量形式的逐图像 K 与 `c2w` | 需要扩展 3DGS reader/Camera 以贯通逐图像 K |
| 运行粒度 | 一个训练任务遍历大量 bin | 启动器依次为每个 bin 启动一个优化任务 |
| 里程碑状态 | 网络权重全局共享 | 每个场景有自己的 1k/5k/10k Gaussian 状态 |
| 断点语义 | Lightning checkpoint | Center150 仅按完整场景跳过，不做样本内恢复 |

因此，本项目的数据适配分为两层：

1. `comp_svfgs` 负责把原始 OmniScene bin 解析、校验并预处理为 3DGS 可读场景；
2. `scripts/run_omniscene.py` 负责逐场景调用 Perceptual-GS 的训练入口，并在训练进程内完成里程碑渲染、评估和结果落盘。

## 4. 计划新增和修改的文件

```text
Perceptual-GS/
├── comp_svfgs/
│   ├── __init__.py
│   ├── dataset_omniscene.py       # split、bin、图像、K、c2w、绝对深度/置信度加载
│   └── omniscene_preprocess.py    # 场景物化、敏感度图、点云、缓存校验
├── docs/
│   └── OmniScene 数据集实验文档.md
├── scripts/
│   └── run_omniscene.py           # 单入口调度、完成态检查与汇总
├── tests/
│   └── test_omniscene_protocol.py # 数据协议、坐标和断点语义测试
└── output/                        # 运行时自动创建，已由 .gitignore 忽略
```

现有文件预计做最小范围修改：

- `scene/dataset_readers.py`：读取逐帧 K、稳健识别 train/test 路径、正确处理 RGB 图像的 alpha，并加载 prepared PLY；
- `scene/cameras.py`、`utils/camera_utils.py`、必要时 `utils/graphics_utils.py`：把逐帧 `fx/fy/cx/cy` 贯通到投影矩阵；
- `train.py`：增加仅由 OmniScene runner 开启的完整里程碑评估和累计训练计时；
- `metrics.py` 或一个可复用指标 helper：复用项目现有 PSNR/SSIM/LPIPS 定义，避免重新定义指标；
- `.gitignore`：现有 `output/` 规则已经覆盖运行结果，原则上不再扩大忽略范围。

不会把 OmniScene 逻辑塞进原有通用 `run.py`，也不会改变 main 分支原数据集的默认行为。

## 5. `dataset_omniscene.py` 设计

### 5.1 split 与 Center150 所有权

数据类提供 `train`、`val`、`test`、`demo`、`center150` 五种模式，默认 `center150`：

- `train`：读取 `bins_train_3.2m.json` 全量列表；
- `val`：兼容 DepthSplat 的 `bins[:30000:3000][:10]`；
- `test`：兼容 DepthSplat 的 mini-test 策略 `bins[0::14][:2048]`；
- `demo`：沿用 DepthSplat 的 `bins_dynamic_demo` 固定 token；
- `center150`：只读 `interp_12Hz_trainval/bins_center150_v1.json`。

Center150 加载时必须校验：

- JSON 中存在列表字段 `bins`；
- 恰好 150 项且 token 唯一；
- token 满足 `scene<hex>_bin<digits>`；
- 解析出的 scene token 也恰好 150 个且互不重复；
- 每个 `bin_infos_3.2m/<token>.pkl` 存在；
- 保持 JSON 原顺序，不自行排序、替换失败样本或重新生成子集。

### 5.2 路径转换

保留 DepthSplat `load_conditions` 的路径转换顺序，避免 nuScenes 原始路径与本地根目录不一致：

1. 将元数据中的 `/datasets/nuScenes` 前缀替换为 CLI 指定的数据根目录；
2. RGB：`samples`/`sweeps` → `samples_small`/`sweeps_small`；
3. 内参：`samples`/`sweeps` → `samples_param_small`/`sweeps_param_small`，后缀 `.jpg` → `.json`；
4. 绝对深度：`samples_small`/`sweeps_small` → `samples_dptm_small`/`sweeps_dptm_small`，后缀 `.jpg` → `_dpt.npy`；
5. 置信度：同一绝对深度路径的 `_dpt.npy` → `_conf.npy`。

第一版不访问 `samples_mask_small`、`sweeps_mask_small`、`samples_dpt_small` 或 `sweeps_dpt_small`。

### 5.3 resize 与逐图像内参

loader 接受 `(112, 200)` 和 `(224, 400)` 两种目标分辨率，默认前者。若原图尺寸为 `(H0,W0)`、目标尺寸为 `(H,W)`，则：

```text
sx = W / W0
sy = H / H0
fx' = fx * sx, cx' = cx * sx
fy' = fy * sy, cy' = cy * sy
```

RGB、绝对深度和置信度同步 resize；K 保持像素单位，不在 prepared scene 中归一化。每个 frame 都单独保存 `fl_x/fl_y/cx/cy/w/h`，不使用全局共享 `camera_angle_x`。

### 5.4 单个样本的内存结构

```text
sample
├── bin_token: str
├── context
│   ├── image:      float32 [6, 3, H, W]
│   ├── intrinsics: float32 [6, 3, 3]       # 像素单位
│   ├── extrinsics: float32 [6, 4, 4]       # 原始 OpenCV c2w
│   ├── depth_m:    float32 [6, H, W]
│   └── confidence: float32 [6, H, W]
└── target
    ├── image:      float32 [18, 3, H, W]
    ├── intrinsics: float32 [18, 3, 3]
    └── extrinsics: float32 [18, 4, 4]      # 原始 OpenCV c2w
```

视角顺序沿用 DepthSplat：

- 相机顺序：`FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT`；
- context：六路相机的 index 0；
- target 前 12 张：每个相机依次取 index 1、2；
- target 后 6 张：按相同相机顺序追加 context。

`n_views=6` 是第一版的协议不变量。Perceptual-GS 原训练代码没有必要的 `n_views` 子采样接口，只要 `transforms_train.json` 恰好包含 6 帧，它就会使用全部输入图像。启动器可显示记录 `n_views=6`，但不接受其他值，避免出现参数看似可改、实际未生效的情况。

## 6. 预处理缓存与场景目录

### 6.1 目录布局

以默认 Center150/112×200 为例：

```text
output/omniscene_prepared/center150_112x200/
└── 001_<bin_token>/
    ├── train/
    │   ├── 000.png ... 005.png
    │   └── sensitivity_maps/
    │       └── 000.png ... 005.png
    ├── test/
    │   ├── 000.png ... 017.png
    │   └── sensitivity_maps/
    │       └── 000.png ... 017.png
    ├── transforms_train.json
    ├── transforms_test.json
    ├── points3d.ply
    └── meta.json
```

Center150 场景名前缀固定为三位序号：`001_`、`002_`……`150_`，随后原样拼接 bin token。其他模式的前缀宽度至少两位，并按列表顺序生成。

`output/` 已被 Git 忽略，runner 在运行时创建，不提交缓存、模型或指标。

### 6.2 Perceptual sensitivity map

Perceptual-GS 不是普通 3DGS，训练时会读取每个 camera 对应的感知敏感度图。预处理必须复用 `preprocess.py` 的同一算法和常量：

- Gamma：1.5；
- Sobel 边缘；
- enhancement threshold：0.05；
- 5×5 平均池化；
- smoothing threshold：0.3。

6 张 train 和 18 张 test 图像都生成同名单通道 PNG。test sensitivity map 虽不参与图像重建优化，但当前评估代码会访问它，完整生成可以避免 reader 特判和评估缺文件。

该步骤可在 CPU 上完成，避免为简单预处理额外占用 GPU；数值流程与原实现保持一致。

### 6.3 坐标系闭环

OmniScene 的 `sensor2lidar_transform` 作为 OpenCV 相机坐标系下的 `c2w_cv`，世界坐标是当前 bin 中心关键帧的 LiDAR 坐标系。

Perceptual-GS 当前 Blender reader 会把输入 `c2w` 的 Y、Z 轴翻转后再求 `w2c`。为保持它的既有约定，prepared transforms 写入：

```text
flip_yz = diag(1, -1, -1, 1)
c2w_gl = c2w_cv @ flip_yz
```

reader 读取后再执行一次 Y/Z 翻转，恢复为与点云一致的 `c2w_cv`。禁止同时写原始 OpenCV pose 又保留 reader 翻转，否则相机会被重复转换，点云落到相机后方。

坐标实现完成后必须做端到端检查，而不是只比较 JSON：

- prepared `c2w_gl` 经 reader 后应与原始 `c2w_cv` 数值一致；
- 随机抽取有效深度像素，执行 pixel → camera → world → camera → pixel，重投影误差应接近浮点误差；
- 反投影点在对应 OpenCV camera 中应具有正 Z；
- 至少对一个真实 Center150 样本，将生成点云与独立参考公式、DropGaussian/Octree-GS 的已验证链路逐点或抽样比较。

### 6.4 Metric3D 初始化点云

只使用 6 个 context 视角。对每个像素 `(u,v)`：

```text
valid = isfinite(depth) and depth > 0 and confidence > 0.3
x = (u - cx) / fx * depth
y = (v - cy) / fy * depth
z = depth
p_world = c2w_cv @ [x, y, z, 1]^T
```

颜色取 resize 后 RGB 的同位置像素，以 `uint8 [0,255]` 写入 PLY；这是必要的，因为本项目的 `fetchPly` 会再除以 255。点云文件名固定为小写 `points3d.ply`，与 Perceptual-GS 的 `readNerfSyntheticInfo` 完全一致。

若深度、置信度缺失，或过滤后没有有效点，第一版直接报错，不静默回退随机点云。这样可以避免不同场景混用不同初始化方式而污染对比。若以后确需随机初始化，应增加显式调试参数并使用独立结果目录。

### 6.5 transforms 与 alpha 处理

每个 frame 写入：

```json
{
  "file_path": "train/000",
  "transform_matrix": "c2w_gl 4x4",
  "fl_x": 161.6,
  "fl_y": 169.6,
  "cx": 100.0,
  "cy": 56.0,
  "w": 200,
  "h": 112
}
```

当前 `readCamerasFromTransforms` 只读取一个顶层 `camera_angle_x`，并假定主点居中。实施时应扩展为优先读取逐帧 K，并将其传入 `Camera`/投影矩阵。即使当前 `cx/cy` 恰好位于中心，也不应丢弃这两个字段。

prepared 图像为普通 RGB。当前 reader 将所有图像强制转成 RGBA，导致普通 RGB 也得到全 1 alpha，进而触发 `train.py` 中额外的 alpha loss 和 alpha pruning。适配时必须只在源图真实含 alpha 通道时设置 `CameraInfo.alpha`；OmniScene 应明确得到 `alpha=None`。

### 6.6 缓存完整性

`meta.json` 至少记录：

- prepared format version；
- bin token 和三位序号；
- split、分辨率、`n_views=6`、`target_views=18`；
- depth confidence threshold；
- 坐标约定版本和 sensitivity 参数；
- train/test 图像名列表；
- 初始化点数；
- 原始数据根目录的真实路径。

缓存只有在 transforms、PLY、meta、6+18 张 RGB 和对应 sensitivity map 全部存在、非空、数量及 meta 参数一致时才有效。预处理先写临时目录，全部校验通过后再原子替换正式目录；不能仅凭场景目录存在就跳过。

## 7. Perceptual-GS 主流程接入

### 7.1 相机加载

`scene/dataset_readers.py` 的 OmniScene/Blender 路径需要完成：

- 从每个 frame 读取 `fl_x/fl_y/cx/cy/w/h`；
- 根据实际 transforms 文件或路径首段稳健判断 `train`/`test`，替换当前 `frame["file_path"][3]` 的脆弱判断；
- 按第 6.3 节完成一次且仅一次坐标转换；
- 从 `train/sensitivity_maps` 或 `test/sensitivity_maps` 加载同名敏感度图；
- 普通 RGB 返回 `alpha=None`；
- `readNerfSyntheticInfo` 优先读取已有 `points3d.ply`，不存在才进入原项目随机初始化逻辑。正式 OmniScene 流程会在进入训练前保证 PLY 存在。

`Camera` 应保留逐帧 K。`fx/fy` 用于给 rasterizer 提供对应的 `tan_fovx/tan_fovy`，`cx/cy` 用于构建可能非对称的投影矩阵。resize 后若仍通过 `-r 1` 加载，则不能再次缩放；若未来支持其他 `-r`，K 必须与图像同步二次缩放。

### 7.2 训练视角与预算

prepared `transforms_train.json` 只含 6 帧，因此 `Scene.getTrainCameras()` 恰好返回 6 个输入视角，原训练循环会循环随机抽取并用尽当前队列，不需要再对相机做子采样。

runner 传给训练入口的关键参数为：

```text
--eval
-s <prepared_scene>
-m <scene_result>
-r 1
--iterations 10000
--test_iterations 1000 5000 10000
--save_iterations 10000
--full_eval_metrics
```

不会传入 `--checkpoint_iterations` 或 `--start_checkpoint`。Perceptual-GS 原有 checkpoint 接口保留给其他实验，但 Center150 runner 禁止通过额外参数绕过这一约束。

Perceptual-GS 默认 `densify_until_iter=15000`，高于本实验的 10000 上限。这里不把它压缩到 5000 或 10000：评估的是原方法在固定优化预算下的轨迹，而不是重新调参后的变体。里程碑评估插入当前 `training_report` 所在位置，保持原训练更新、densification 和评估的相对顺序。

### 7.3 进程内里程碑评估

单个场景只启动一次训练进程，从 0 连续优化到 10k。在 1k、5k、10k 时直接使用内存中的当前 Gaussian：

1. 渲染全部 18 个 test cameras；
2. 保存 18 张 render 和 18 张 GT；
3. 使用项目现有定义计算 PSNR、SSIM、LPIPS；
4. 写入该里程碑的指标和累计纯训练耗时；
5. 恢复训练，直到下一里程碑；
6. 只在最终 10k 保存 `point_cloud/iteration_10000/point_cloud.ply` 及 Perceptual-GS 原有 sensitivity 状态。

不采用“分别训练 1k、5k、10k 三次”的方式，也不在里程碑后调用独立 `render.py` 重新加载模型。否则要么三个指标来自不同随机轨迹，要么必须保存中间 checkpoint/PLY，与本实验的无 checkpoint 要求冲突。

指标默认对全部 18 个 target 视角等权平均，与已完成的优化式基线协议保持一致。文件中可保留逐视角值用于排错，但正式汇总使用每个 bin 的 18 视角平均，再对 150 个 bin 等权平均。

### 7.4 训练耗时定义

训练耗时使用 CUDA 同步后的 wall-clock 累计值，计时范围是实际优化迭代，包括该方法正常训练所需的 forward、backward、optimizer、densification 和 `render_imp` 统计；排除：

- 原始数据预处理和 prepared scene 加载；
- 1k/5k/10k 的 18 视角评估；
- PNG、JSON、文本和 PLY 写盘；
- runner 的场景调度时间。

每个里程碑记录从第 1 次迭代到当前迭代的累计值，必须有限、非负且单调递增。现有 `total_time` 基于未同步的 CPU `time()`，不能直接作为正式 GPU 训练耗时；实施时需增加显式 CUDA 同步或等价的可靠计时。

## 8. 启动器与单阶段执行

### 8.1 默认命令

默认环境和命令规划为：

```bash
conda run -n perceptual_gs python scripts/run_omniscene.py
```

默认参数：

- mode：`center150`；
- data root：`datasets/omniscene`（本机用忽略的本地软链接指向真实数据目录）；
- resolution：`112x200`，可选 `224x400`；
- `n_views=6`；
- confidence threshold：`0.3`，有效判断使用 `>`；
- iterations：`10000`；
- eval iterations：`1000 5000 10000`；
- GPU：`0`；
- prepared root：`output/omniscene_prepared`；
- result root：`output/omniscene_results`；
- 传入 Perceptual-GS 的 `-r` 恒为 `1`。

计划支持的覆写示例：

```bash
conda run -n perceptual_gs python scripts/run_omniscene.py \
  --resolution 224x400 \
  --iterations 10000 \
  --eval-iterations 1000 5000 10000 \
  --gpu 0
```

还将支持 `--data-root`、`--prepared-root`、`--result-root`、`--mode`、`--conf-threshold` 和 `--extra-train-args`。数据路径与协议控制参数不能通过 `--extra-train-args` 重复传入。评估点必须严格递增、不能重复、不能超过总迭代数，并要求最后一个评估点等于总迭代数。

不同分辨率或非默认迭代协议使用不同结果目录；同一路径检测到会改变实验语义的协议变化时直接拒绝运行，不混用旧产物。这里的硬协议仅包括数据清单与数据根目录、分辨率、置信度阈值、训练/评估视角、总迭代数、评估点以及会影响结果的显式训练参数，不包括源码文件哈希、文件修改时间、Git commit、分支名称或工作区 clean/dirty 状态。若代码状态发生变化，runner 可以输出提示或把当前 Git 信息写入日志，但不得因此拒绝运行、强制重跑或把原本完整的场景判为不完整。

### 8.2 每个场景的执行顺序

```text
读取 token 列表
  → 先按 scene name 检查严格完成态
      → 已完成：不加载 RGB/深度，立即 [SKIP]
      → 未完成：加载当前 bin
          → prepared cache 完整：直接复用
          → prepared cache 缺失/失配：重新预处理并校验
          → 结果目录存在但不完整：仅删除结果目录，[RESTART]
          → 启动一次 train.py，从 0 连续优化到 10k
          → 核验三个里程碑与最终 PLY
          → 最后原子写 scene_complete.json
全部 150 场景严格完成
  → 生成全量汇总
```

预处理、加载、训练、渲染、评估都由同一个顶层 runner 顺序调度，用户不需要先执行单独的 prepare 命令。

## 9. 结果目录、完成态与汇总

### 9.1 单场景结果

```text
output/omniscene_results/center150_112x200/
├── center150_protocol.json
├── 001_<bin_token>/
│   ├── cfg_args
│   ├── input.ply
│   ├── cameras.json
│   ├── metrics_1000.txt
│   ├── metrics_5000.txt
│   ├── metrics_10000.txt
│   ├── training_time_1000.txt
│   ├── training_time_5000.txt
│   ├── training_time_10000.txt
│   ├── test/
│   │   ├── ours_1000/{renders,gt}/   # 各 18 张
│   │   ├── ours_5000/{renders,gt}/   # 各 18 张
│   │   └── ours_10000/{renders,gt}/  # 各 18 张
│   ├── point_cloud/iteration_10000/
│   │   ├── point_cloud.ply
│   │   └── sensitivity.ply
│   └── scene_complete.json
├── ...
├── 150_<bin_token>/
├── center150_metrics_summary.json
└── center150_metrics_summary.txt
```

`sensitivity.ply` 是现项目用 `torch.save` 保存的 sensitivity 状态，虽然后缀名为 `.ply`，仍按现状保留，不在本适配中重命名。

### 9.2 产物与实验协议的完成条件（代码状态宽松）

一个 Center150 场景只有同时满足以下条件才可跳过：

- prepared cache 通过第 6.6 节校验；
- 1k、5k、10k 指标文件均包含有限的 PSNR/SSIM/LPIPS；
- 三个累计训练时间有限、非负且单调；
- 每个里程碑的 render/GT 文件集合均恰好为 18 张，文件非空且可读取；
- 最终 10k `point_cloud.ply` 和 sensitivity 状态存在且非空；
- `scene_complete.json` 中的 token、分辨率、迭代点和协议版本一致；
- 根目录 `center150_protocol.json` 与当前 data root、分辨率、置信度阈值、迭代预算及额外训练参数一致。

不能仅凭进程退出码、结果目录存在、某个指标文件或旧 completion marker 判定完成。

上述“严格”只针对必要产物和会改变实验含义的协议字段，不扩展为代码仓库指纹校验。具体而言：

- 不扫描、哈希或逐文件比对 Python/CUDA/配置文件，也不因文件修改时间变化而失效已有完成态；
- 不要求当前 Git commit 与运行时一致，不要求仍在同一分支，也不要求工作区干净；
- 不因源码或 Git 状态变化自动删除已经完整的场景结果；
- 若 `scene_complete.json` 或日志记录 `git_commit`、`git_branch`、`git_dirty` 等信息，它们只用于人工追溯和宽松告警，不参与完成态布尔判断；
- 是否跳过仍以本节列出的产物完整性和语义协议兼容性为准。研究者若明确判断代码改动会影响结果，应主动切换结果目录或删除对应场景结果，而不是由 runner 猜测代码等价性。

### 9.3 不完整样本处理

- prepared cache 合法时始终保留；
- 结果目录不完整时，在确认它位于当前 experiment root 下后删除该单场景结果目录；
- 不搜索或加载 `chkpnt*.pth`，直接从 0 重跑；
- 不把失败/OOM 场景替换为其他 bin；
- 代码文件、Git commit、分支或 dirty 状态变化本身不触发清理和重跑；
- 任一场景失败时不生成伪装成 150 场景完整结果的最终汇总，修复后重复同一命令即可继续。

### 9.4 汇总格式

全部 150 个场景完成后生成 JSON 和便于阅读的 TXT。JSON 至少包含：

- 协议版本、数据根目录、分辨率、阈值、总迭代数、评估点、样本数和生成时间；
- 150 个场景的 scene name、bin token、各里程碑指标和训练耗时；
- 1k、5k、10k 各自的 150 场景等权平均 PSNR、SSIM、LPIPS 和累计训练耗时。

平均顺序为：先对每个 bin 的 18 个视角求均值，再对 150 个 bin 求均值，不能把全部图像打散后按图像数量加权。

## 10. 计划验证与验收标准

实现阶段先做 CPU/静态测试，再做一个真实样本的小迭代 GPU 冒烟，最后才允许启动正式 Center150。

### 10.1 数据与协议测试

- 五种 mode 的 token 来源和抽样规则正确；
- Center150 的数量、唯一性、scene 覆盖、顺序和 pkl 存在性校验；
- 路径替换覆盖 `samples` 与 `sweeps`；
- 112×200、224×400 的 RGB/depth/conf/K 尺寸一致；
- `confidence > 0.3`、有限正深度过滤准确；
- loader 返回 6 context、18 target，视角顺序稳定；
- 不访问动态 mask 和相对深度路径。

### 10.2 几何测试

- 逐帧 K 经 preprocess → transforms → reader 后保持一致；
- `c2w_cv → c2w_gl → reader` 只发生一对 Y/Z 翻转；
- 深度反投影/重投影误差接近浮点误差，点在相机前方；
- PLY 坐标、颜色、数量均有限且与独立参考实现一致；
- 使用实际 Perceptual rasterizer 对一个 prepared scene 做渲染冒烟，排除“公式正确但 reader/rasterizer 约定不一致”。

### 10.3 流程测试

- 构造临时 prepared/result 目录，验证缺任一指标、时间、图像或最终 PLY 都不能跳过；
- 完整场景在访问 RGB/depth 前快速跳过；
- 不完整结果被清理后从 0 重跑，prepared cache 保留；
- runner 生成的命令包含 `-r 1`、6 训练视角、10k 和三个评估点，不包含 checkpoint 参数；
- 会改变实验语义的非默认协议不会与默认目录混用；
- 在完整产物和语义协议不变时，切换 Git commit/分支、使工作区变脏或修改源码文件时间戳都不会阻止快速跳过；
- 150 个场景未全部完成时拒绝写最终汇总；
- `git diff --check`、Python 编译检查和 `perceptual_gs` 环境内的单元测试通过。

### 10.4 真实样本冒烟验收

在用户审阅文档并授权实现后，用 Center150 的一个真实 bin 做 1～数十次迭代冒烟，确认：

- 6 张 train 和 18 张 test 全部被正确加载；
- 初始化点云不是随机点云；
- 没有错误启用 alpha loss；
- 逐帧投影与图像尺寸一致；
- 18/18 render 和 GT、指标、训练耗时、最终 PLY 均按协议生成；
- 不产生任何样本内 checkpoint。

正式 150 场景实验不在实现和冒烟通过前启动。

## 11. 实施顺序

文档审阅通过后按以下顺序开发：

1. 实现纯数据 loader 和 Center150 严格校验；
2. 实现场景预处理、sensitivity map、Metric3D PLY 和缓存 meta；
3. 改造 Perceptual-GS 的逐帧相机内参、坐标和 alpha 加载链路；
4. 增加训练进程内的里程碑全量评估与可靠计时；
5. 实现 runner、完成态、无 checkpoint 重跑和最终汇总；
6. 增加协议/几何/调度测试；
7. 在 `perceptual_gs` 环境做静态检查和单样本 GPU 冒烟；
8. 将验证证据提交审阅，得到确认后再启动正式 Center150。
