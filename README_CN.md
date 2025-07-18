# 基于二次样条分位数函数和自回归循环神经网络的风电功率非参数概率预测

## 项目简介

本项目实现了论文《Nonparametric Probabilistic Forecasting for Wind Power Generation using Quadratic Spline Quantile Function and Autoregressive Recurrent Neural Network》中提出的方法。

本实现主要基于 [TimeSeries 仓库](https://github.com/zhykoties/TimeSeries)。我们深深感谢张云凯、江乔和马雪莹的工作。同时感谢 [GEFCom2014](https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133#ec000005) 提供开源数据。

## 作者信息

* **王可** (<w110k120@stu.xjtu.edu.cn>) - *西安交通大学，西安*
* **张耀** (<yaozhang_ee@ieee.org>) - *西安交通大学，西安*
* **林帆** (<lf1206@stu.xjtu.edu.cn>) - *西安交通大学，西安*
* **王建学** (<jxwang@mail.xjtu.edu.cn>) - *西安交通大学，西安*
* **朱墨润** (<1491974695@qq.com>) - *西安交通大学，西安*

## 方法介绍

我们提出了一种用于风电功率概率预测的非参数灵活方法：

1. **样条分位数函数**：首先通过样条分位数函数指定风电功率输出的分布，避免假设参数形式，同时为风电功率密度提供灵活的形状。

2. **自回归循环神经网络**：使用自回归循环神经网络建立从输入特征到二次样条分位数函数参数的非线性映射。

3. **CRPS损失函数**：设计了一种基于连续排序概率得分（CRPS）的新型损失函数来训练预测模型。

4. **计算效率优化**：为了提高训练中的计算效率，我们推导了计算CRPS损失函数所需积分的闭式解。

## 实验结果

四种QSQF模型在可靠性、锐度和CRPS方面的主要结果如下表所示：

| 指标 | QSQF-A | QSQF-B | QSQF-AB | QSQF-C |
| :---: | :----: | :----: | :-----: | :----: |
| MRAE  | 0.0401 | 0.0506 | 0.0397  | 0.0286 |
| NAPS  | 0.3423 | 0.3530 | 0.3577  | 0.3698 |
| CRPS  | 0.0811 | 0.0832 | 0.0868  | 0.0764 |

## 环境配置

### 系统要求

- Python 3.6.7（推荐）或 Python 3.6+
- CUDA支持（可选，用于GPU加速）

### 依赖包安装

#### 方法一：自动配置（推荐）

运行环境配置脚本：
```bash
python setup_env.py
```

该脚本会自动：
- 检查Python版本
- 安装所有依赖包
- 检查CUDA支持
- 验证数据目录结构
- 显示可用模型类型
- 创建配置模板（base_model + example_model）

#### 方法二：手动配置

1. **创建虚拟环境**（推荐）：
```bash
# 使用conda
conda create -n qsqf python=3.6.7
conda activate qsqf

# 或使用venv
python3.6 -m venv qsqf_env
source qsqf_env/bin/activate  # Linux/Mac
# qsqf_env\Scripts\activate  # Windows
```

2. **安装依赖包**：
```bash
pip install -r requirements.txt
```

### 依赖包详情

#### Python 3.6.7（原始版本）
- numpy 1.14.3 - 数值计算
- pandas 0.23.0 - 数据处理
- scipy 1.1.0 - 科学计算
- torch 0.4.1 - 深度学习框架
- tqdm 4.26.0 - 进度条显示
- matplotlib 2.2.2 - 数据可视化

#### Python 3.6+（兼容版本）
- numpy >=1.19.0,<2.0.0 - 数值计算
- pandas >=1.0.0,<2.0.0 - 数据处理
- scipy >=1.5.0,<2.0.0 - 科学计算
- torch >=1.7.0,<2.0.0 - 深度学习框架
- tqdm >=4.50.0 - 进度条显示
- matplotlib >=3.3.0,<4.0.0 - 数据可视化

## 项目结构

```
QSQF/
├── README.md              # 英文说明文档
├── README_CN.md           # 中文说明文档
├── setup_env.py           # 环境配置脚本
├── requirements.txt       # 依赖包列表
├── controller.py          # 主控制程序
├── kernel.py              # 核心训练和评估逻辑
├── utils.py               # 工具函数
├── dataloader.py          # 数据加载器
├── data_prepare.py        # 数据预处理
├── search_params.py       # 参数搜索
├── model/                 # 模型定义
│   ├── __init__.py
│   ├── net_qspline_A.py   # QSQF-A模型
│   ├── net_qspline_B.py   # QSQF-B模型
│   ├── net_qspline_AB.py  # QSQF-AB模型
│   ├── net_qspline_C.py   # QSQF-C模型
│   └── net_lspline.py     # 线性样条模型
├── data/                  # 数据目录
│   ├── Zone1/             # 区域1数据
│   ├── Zone2/             # 区域2数据
│   └── ...                # 其他区域数据
└── experiments/           # 实验结果
    └── param_search/      # 参数搜索结果
```

## 使用方法

### 快速开始

1. **运行环境配置脚本**：
```bash
python setup_env.py
```

2. **开始训练**：
```bash
# 快速测试（推荐新手）
python controller.py --model-dir example_model

# 完整训练
python controller.py --model-dir base_model
```

### 详细步骤

#### 1. 数据准备

确保数据文件已正确放置在 `data/` 目录下的相应区域文件夹中。

### 2. 模型训练

项目支持多种QSQF模型变体：

- **QSQF-A**: 使用 `QAspline` 配置
- **QSQF-B**: 使用 `QBspline` 配置  
- **QSQF-AB**: 使用 `QABspline` 配置
- **QSQF-C**: 使用 `QCDspline` 配置
- **线性样条**: 使用 `Lspline` 配置

### 3. 运行训练

```bash
# 使用自动配置脚本创建的环境
python controller.py --model-dir base_model
python controller.py --model-dir example_model

# 或手动指定模型目录
python controller.py --model-dir experiments/your_model_config
```

### 4. 参数配置

每个模型都需要相应的配置文件：
- `params.json`: 模型参数配置
- `dirs.json`: 目录路径配置

## 注意事项

1. **Python版本**：推荐使用Python 3.6.7，但Python 3.6+版本也支持运行
2. **依赖包版本**：脚本会自动根据Python版本选择合适的依赖包版本
3. **PyTorch版本**：Python 3.6使用PyTorch 0.4.1，更高版本使用PyTorch 1.7+
4. **GPU支持**：项目支持CUDA加速，但CPU运行也是可行的
5. **数据格式**：确保数据文件格式符合项目要求

## 故障排除

### 常见问题

1. **Python版本兼容性**：脚本会自动检测Python版本并安装合适的依赖包
2. **版本冲突**：如果遇到依赖包冲突，脚本会自动尝试安装兼容版本
3. **CUDA问题**：如果GPU不可用，程序会自动切换到CPU模式
4. **内存不足**：可以调整batch_size参数减少内存使用

### 获取帮助

如果遇到问题，请检查：
- Python版本是否为3.6+（推荐3.6.7）
- 所有依赖包是否正确安装
- 数据文件是否完整且格式正确
- 配置文件是否存在且格式正确

## 许可证

本项目遵循LICENSE文件中规定的许可证条款。 