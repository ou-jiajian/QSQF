#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSQF项目环境配置脚本
自动检测和配置运行环境
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    current_version = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor == 6:
        print(f"✅ Python版本正确: {current_version}")
        return True
    elif version.major == 3 and version.minor >= 6:
        print(f"⚠️  Python版本兼容: {current_version} (推荐3.6.7)")
        print("   当前版本应该可以运行，但建议使用Python 3.6.7以获得最佳兼容性")
        return True
    else:
        print(f"❌ Python版本不兼容: {current_version}, 需要Python 3.6+")
        return False

def check_pip():
    """检查pip是否可用"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip可用")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip不可用")
        return False

def install_requirements():
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    # 根据Python版本选择requirements文件
    version = sys.version_info
    if version.major == 3 and version.minor == 6:
        requirements_file = "requirements.txt"
        print("   使用原始requirements.txt (Python 3.6)")
    else:
        requirements_file = "requirements_compatible.txt"
        print(f"   使用兼容版本requirements_compatible.txt (Python {version.major}.{version.minor})")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                      check=True)
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        print("   尝试安装兼容版本...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_compatible.txt"], 
                          check=True)
            print("✅ 兼容版本依赖包安装完成")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ 兼容版本安装也失败: {e2}")
            return False

def check_cuda():
    """检查CUDA支持"""
    print("🔍 检查CUDA支持...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU模式")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_data_structure():
    """检查数据目录结构"""
    print("📁 检查数据目录结构...")
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data目录不存在")
        return False
    
    zone_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Zone")]
    if len(zone_dirs) == 0:
        print("⚠️  未找到Zone数据目录")
        return False
    
    print(f"✅ 找到{len(zone_dirs)}个数据区域: {[d.name for d in zone_dirs]}")
    return True

def show_available_models():
    """显示可用的模型类型"""
    print("\n📋 可用的模型类型:")
    models = {
        "QAspline": "QSQF-A模型 - 基础二次样条模型",
        "QBspline": "QSQF-B模型 - 改进的二次样条模型", 
        "QABspline": "QSQF-AB模型 - 组合二次样条模型",
        "QCDspline": "QSQF-C模型 - 高级二次样条模型",
        "Lspline": "线性样条模型 - 线性样条基准模型"
    }
    
    for model_type, description in models.items():
        print(f"  • {model_type}: {description}")

def create_config_template():
    """创建配置模板"""
    print("📝 创建配置模板...")
    
    # 创建base_model目录
    base_model_dir = Path("base_model")
    base_model_dir.mkdir(exist_ok=True)
    
    # 创建example_model目录
    example_model_dir = Path("example_model")
    example_model_dir.mkdir(exist_ok=True)
    
    # base_model配置（完整训练）
    params_base = {
        "line": "QAspline",
        "batch_size": 32,
        "num_epochs": 100,
        "lr": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1
    }
    
    dirs_base = {
        "data_dir": "data",
        "dataset": "Zone1",
        "model_dir": "base_model",
        "model_save_dir": "base_model"
    }
    
    # example_model配置（快速测试）
    params_example = {
        "line": "QAspline",
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 0.001,
        "hidden_size": 32,
        "num_layers": 2,
        "dropout": 0.1
    }
    
    dirs_example = {
        "data_dir": "data",
        "dataset": "Zone1",
        "model_dir": "example_model",
        "model_save_dir": "example_model"
    }
    
    import json
    # 保存base_model配置
    with open(base_model_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params_base, f, indent=2, ensure_ascii=False)
    
    with open(base_model_dir / "dirs.json", "w", encoding="utf-8") as f:
        json.dump(dirs_base, f, indent=2, ensure_ascii=False)
    
    # 保存example_model配置
    with open(example_model_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params_example, f, indent=2, ensure_ascii=False)
    
    with open(example_model_dir / "dirs.json", "w", encoding="utf-8") as f:
        json.dump(dirs_example, f, indent=2, ensure_ascii=False)
    
    print("✅ 配置模板创建完成（base_model + example_model）")

def main():
    """主函数"""
    print("🚀 QSQF风电功率预测项目 - 环境配置脚本")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ 请安装Python 3.6.7")
        return False
    
    # 检查pip
    if not check_pip():
        print("\n❌ 请确保pip可用")
        return False
    
    # 安装依赖包
    if not install_requirements():
        print("\n❌ 依赖包安装失败")
        return False
    
    # 检查CUDA
    check_cuda()
    
    # 检查数据目录
    if not check_data_structure():
        print("\n⚠️  请确保数据文件已正确放置")
    
    # 显示可用模型
    show_available_models()
    
    # 创建配置模板
    create_config_template()
    
    print("\n" + "=" * 60)
    print("✅ 环境配置完成！")
    print("\n🎯 快速开始:")
    print("1. 快速测试（推荐新手）:")
    print("   python controller.py --model-dir example_model")
    print("\n2. 完整训练:")
    print("   python controller.py --model-dir base_model")
    print("\n3. 查看训练日志:")
    print("   tail -f example_model/train.log")
    print("\n📚 更多信息请查看 README_CN.md")
    
    return True

if __name__ == "__main__":
    main() 