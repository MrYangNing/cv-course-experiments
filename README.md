# 计算机视觉课程实验

从彩色图像的滤波与特征匹配，到目标检测、语义分割，再到注意力与模型压缩。本仓库包含三项实验的 Python 源码、输入文件校验值和报告使用的参考指标。首次运行建议从实验一开始，确认图像与指标能正常生成，再运行 GPU 训练。

## 1. 获取代码并配置环境

在装有 Conda 的 PowerShell 中执行以下命令。实验使用 Windows x64、Python 3.11、PyTorch 2.8、CUDA 12.8 和 NVIDIA RTX 5090。实验一使用 CPU；检测、分割和分类脚本使用第一个 CUDA 设备。

```powershell
git clone https://github.com/MrYangNing/cv-course-experiments.git
cd cv-course-experiments
conda create -n cv --file conda-win64.lock
conda activate cv
python -c "import torch, cv2; print('PyTorch:', torch.__version__, 'OpenCV:', cv2.__version__, 'CUDA:', torch.cuda.is_available())"
```

已有匹配的 `cv` 环境时直接激活。Windows 锁定文件记录本次运行所用的具体 Conda 包；在其他平台可参考 `environment.yml` 调整。后续命令均在仓库根目录、同一环境中执行。GPU 训练前，上面的 CUDA 检查应返回 `True`。

## 2. 实验一：从图像处理开始

```powershell
python work/personal/foundation.py
python work/personal/image_series.py
```

脚本读取 scikit-image 的 coffee、astronaut、chelsea 和 rocket。先查看 `work/personal/channels.png` 与 `filters.png`，再看 `matches.png` 中的特征连线。`foundation.json` 保存主图的 PSNR、SSIM、边缘点数和匹配误差，`image_series.json` 保存四张图的 72 组滤波对照。主图加入椒盐噪声后，3×3 中值滤波的 PSNR 约为 **29.781560 dB**。

## 3. 实验二：目标检测与语义分割

```powershell
python prepare_data.py detection
python work/personal/detection.py
python work/personal/repeat_detection.py
python work/personal/detection_examples.py

python prepare_data.py segmentation
python work/personal/segmentation.py
python work/personal/seg96.py
```

- **检测**：第一条训练脚本从 COCO128 构造人物、汽车、杯子三类数据，按 48/12/14 张划分训练、验证和测试集，用 YOLOv8n 微调 12 轮。`repeat_detection.py` 复用同一划分训练另两个种子，须在首次训练成功后运行。查看 `detection.json`、`det_seed2024.json`、`det_seed2025.json` 和 `det_examples.png`。报告的三次平均 mAP@0.50 为 **0.119032**。
- **分割**：两个脚本分别以 128×128 和 96×96 训练。报告使用 Oxford-IIIT Pet 官方图像归档开头 **13,631,251 字节**中的完整可解码图像；准备脚本复取相同字节并核验 SHA-256，文件名中的 `.incomplete` 表示有意保留的归档片段。用官方标注匹配得到 57/14/59 张，名单见 `reference/seg_split.json`。输出 `segmentation.json`、`s96_segmentation.json` 和分割示例图。128×128 无跳连基线的测试 mIoU 为 **0.633550**。

## 4. 实验三：注意力、轻量卷积与蒸馏

```powershell
python prepare_data.py classification
foreach ($seed in 2024,2025,2026) {
    $env:CV_SEED = "$seed"
    python work/personal/classification.py
    if ($LASTEXITCODE -ne 0) { throw "训练在种子 $seed 处中止" }
}
python work/personal/summary_graphs.py
```

每类从 CIFAR-10 的原训练集抽取 1,000 张训练、200 张验证，从原测试集抽取 200 张测试。固定样本划分，改变训练种子。每次依次训练 ResNet-18、SE、CBAM、ECA、标准 CNN、可分离 CNN 和蒸馏学生，每组 6 轮。种子 2026 另运行温度和权重扫描；如需先跑一遍完整流程，可只运行该种子。

每个种子的逐图预测、指标和热图位于 `work/personal/seed<种子>/`。`summary_graphs.py` 读取种子 2026 的结果生成验证曲线和结构图。核对 Top-1 时，用 JSON 的 `test_correct / test_n`；每次测试共 2,000 张。SE 的三次平均为 **63.5667%**，可分离学生直接训练为 **49.25%**，主蒸馏配置为 **46.45%**。GPU 算子、软件版本与负载可能影响重新训练的数值和计时。

## 5. 数据准备和运行问题

首次下载需要能够访问 Ultralytics 的 GitHub Releases、Oxford Pet 数据服务器和 fast.ai 的 CIFAR-10 镜像。运行 `python prepare_data.py all` 可一次准备全部外部输入，文件通过 `reference/input_hashes.json` 校验后才会放到实验脚本读取的位置。若已经取得配套实验数据包，可将其解压，再用下列命令从本地导入：

```powershell
python prepare_data.py all --from-dir "D:/实验数据包解压目录"
```

该目录应直接包含 `yolov8n.pt` 与 `work/provenance/downloads/`。遇到校验不符，请检查下载是否完整、压缩包版本是否一致。遇到找不到数据或权重时，先确认当前目录是仓库根目录，并重新运行对应的准备命令。若要比较新配置，请先另存已有结果；同一种子重复运行会使用相同输出路径。

原始数据和预训练权重从各自发布方获取，其使用条件见对应项目。[COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) · [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) · [YOLOv8](https://docs.ultralytics.com/models/yolov8/)

## 6. 课程要求核对版的补充实验与附录

在仓库根目录运行。直接使用本地完整包时，可以跳过训练，先核对配套记录，再重绘报告中的图。

| 报告位置 | 命令 | 所需输入和输出 |
|---|---|---|
| 表 8a、11a | `python extra_compare.py` | 读取 Pet 归档、分割名单、分类名单和种子 2026 的预测。终端输出 Otsu/GrabCut 指标及精确 McNemar 检验与 Holm 校正。 |
| 表 12a | 查阅 `reference/classification_2024.json` 至 `classification_2026.json` | Base、DS、KD 对应教师、直接训练学生和蒸馏学生。每个种子的 test_correct 除以 2,000 得到表中准确率。 |
| 附录 C 三幅图 | `python work/personal/appendix_figures.py` | 需要实验一的 image_series、实验二的检测示例与测试图、实验三的逐图预测及分类名单；生成 work/personal/expanded/ 下的图和逐框、逐类计数。 |
| 图 9a、10a、11a | `python work/personal/course_check/plots.py` | 读取同目录 examples.npz，绘制 U-Net、误差叠加、残差注意力结构图，写出 analysis.json。 |
| 从头生成图 10a 的输入 | `python work/personal/course_check/rerun.py` | 先运行 `python prepare_data.py segmentation`；复跑 U-Net/CE，保存 59 张测试图的 ID、RGB、真值和预测到 examples.npz。 |

主实验的 128×128 U-Net/CE 测试 mIoU 为 **0.611200**，彩色图对应的独立复跑为 **0.597232**。主表的结果在 `reference/segmentation.json`，复跑结果在 `reference/supplement/segmentation.json`。要重绘报告中的同一批彩色案例，使用本地完整包自带的 examples.npz。重新训练会覆盖该文件，模型和极端样本可能变化。图中的逐像素颜色分别表示命中（绿）、漏检（红）、误检（橙）和忽略边界（灰）。

补充参考记录在 `reference/supplement/`：extra_compare.json 对应传统基线与配对检验，derived.json 对应附录 C，analysis.json 对应彩色案例。本地完整包另含署名 Word、实验输入、保存的权重、训练日志和原始预测；公开仓库提供生成这些结果的代码和环境。

## 7. 宠物分割三种子复验（表 8b）

固定 57/14/59 张训练、验证、测试图，分别训练 96 和 128 像素的方形输入，通过三种随机种子对照三种网络／损失配置。各运行保留最佳验证轮次的权重、25 轮验证轨迹和测试预测。对应表格的未舍入值在 `reference/pet_repeat_summary.json`。

从完整包根目录运行下面的命令核对已有权重，需要 cv 环境和 CUDA 显卡。公开仓库仅含程序与数值记录，首次从仓库运行时，先用 `python prepare_data.py segmentation` 准备数据，再执行训练命令。

```powershell
conda activate cv
python work/personal/pet_experiment.py --verify
```

如需从头训练并保留原归档，指定独立目录：

```powershell
python work/personal/pet_experiment.py --output work/personal/new_pet_runs
python work/personal/pet_experiment.py --output work/personal/new_pet_runs --verify
```

三个单次运行入口共用同一份数据读取、训练和评价代码。可使用 `--verify --output 新目录` 重绘种子 2026 的三种子实验输出。旧图 10、10a 与表 8 对应的原始文件仍可从完整包查阅。新训练结果应与表 8b 对照。
