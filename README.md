# mHC
实现 mHC: Manifold-Constrained Hyper-Connections
 
相关链接：
- https://arxiv.org/abs/2409.19606
- https://arxiv.org/pdf/2512.24880
 
## 简介
mHC（Manifold-Constrained Hyper-Connections）是一种替代传统残差连接（Residual Connection）的连接方式。
本仓库包含一个 PyTorch 实现，并提供两个最小可跑的示例：
- 图像分类：`ImageHyperConnectionTransformer` 在 MNIST 上训练
- 文本建模：`HyperConnectionDecodeTransformer` 在 tinyshakespeare 上做字符级 next-token 训练
 
## 依赖
- Python 3.10+
- PyTorch
- torchvision
 
安装示例（按你的 CUDA / CPU 环境选择合适的 torch 版本）：
 
```bash
pip install torch torchvision
```
 
## 快速开始
 
### 1) 图像：MNIST 训练（最小示例）
训练脚本：`mHC/image_train.py`
 
在 `mHC/` 目录下运行：
 
```bash
python image_train.py
```
 
你应该能看到类似输出（acc 随 epoch 上升即可）：
```text
epoch 1/5 | train loss ... acc ... | test loss ... acc ...
epoch 2/5 | ...
```
 
默认会把 MNIST 下载到 `./data`。
 
### 2) 文本：tinyshakespeare 训练（最小示例）
训练脚本：`mHC/text_train.py`
 
在 `mHC/` 目录下运行：
 
```bash
python text_train.py
```
 
说明：
- 脚本会自动下载 tinyshakespeare 到 `./data/tinyshakespeare.txt`（如果本地不存在）
- 训练过程中会输出 loss / ppl，并打印一段生成文本用于快速验证训练效果
 
## 代码结构
- `mhyperconn.py`
  - `HyperConnection`：核心超连接模块（包含 mHC 的流形约束逻辑）
  - `ImageHyperConnectionTransformer`：图像分类模型（patch embedding + 多层 block + pooling + classifier）
  - 还包含 `HyperConnectionTransformer`（文本）与 `HyperConnectionDecodeTransformer`（自回归解码器）以及一些辅助组件（norm、drop path 等）
- `image_train.py`
  - MNIST 训练最小例子：展示如何构造 `ImageHyperConnectionTransformer` 并跑通训练
- `text_train.py`
  - tinyshakespeare 训练最小例子：展示如何构造 `HyperConnectionDecodeTransformer` 并跑通字符级语言模型训练
 
## 如何使用 ImageHyperConnectionTransformer
`ImageHyperConnectionTransformer` 的核心用法就是：输入 `(B, C, H, W)`，输出 `(B, num_classes)`。
MNIST 的最小配置（见 `image_train.py`）要点：
- `in_channels=1`：MNIST 是灰度图
- `num_classes=10`：MNIST 10 类
- `image_size` 必须与你喂给模型的图像尺寸一致（示例里用 `Resize((32, 32))`）
- `patch_size` 决定 token 数量（例如 32x32，patch=4x4，则 token 网格是 8x8）
示例（等价于 `image_train.py` 中的构造方式）：
 
```python
from mhyperconn import ImageHyperConnectionTransformer
 
model = ImageHyperConnectionTransformer(
    image_size=(32, 32),
    patch_size=(4, 4),
    in_channels=1,
    num_classes=10,
    dim=96,
    n_layers=6,
    n_heads=4,
    rate=2,
    dropout=0.1,
    pool_size=4,
    mask_ratio=0.0,
)
logits = model(images)  # images: (B, 1, 32, 32)
```
 
## 常见参数建议
- `dim / n_layers / n_heads`：越大越慢、越吃显存；MNIST 任务不需要很大
- `rate`：超连接扩展率，会显著影响计算量/显存
- `mask_ratio`：训练时 token masking（默认示例关闭：`0.0`）
 
## 常见问题
- 训练时如果你机器上同时跑了很多 Python 进程（`nvidia-smi` 里看到多个占用），显存/算力会被抢占，训练会变慢。
- 如果只是想验证能训练成功，建议先用较小的 `epochs`（例如 1~5）。
- `text_train.py` 的训练相对更慢（序列模型 + 训练步数更多）；想更快可调小 `block_size / dim / n_layers / rate`。
- `text_train.py` 首次运行需要网络下载数据；如果网络受限，可以手动下载文件放到 `./data/tinyshakespeare.txt`。
 
## 引用
如果你在论文或项目中使用了本实现，请引用对应论文（见上方 arXiv 链接）。