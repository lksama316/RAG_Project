# diagnose_mineru_v3.py
import subprocess
import sys
import torch
import os
from pathlib import Path


def check_environment():
    print("=" * 50)
    print("MinerU 环境诊断 v3")
    print("=" * 50)

    # 1. Python 版本
    print(f"\n1. Python 版本: {sys.version}")

    # 2. PyTorch & CUDA
    print(f"\n2. PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   当前设备: {torch.cuda.get_device_name(0)}")
        # 修正属性名
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024 ** 3
        print(f"   显存总量: {total_mem:.1f} GB")

        # 检查当前显存使用
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f"   显存已分配: {allocated:.1f} GB")
        print(f"   显存已缓存: {reserved:.1f} GB")

        # 关键警告
        if total_mem < 8:
            print(f"   ⚠️  显存不足！MinerU 推荐至少 8GB，当前仅 {total_mem:.1f}GB")

    # 3. 检查 mineru 安装
    try:
        import mineru
        print(f"\n3. MinerU 版本: {mineru.__version__}")

        # 检查 mineru 配置
        from mineru.utils.config import get_default_config
        config = get_default_config()
        print(f"   默认配置:")
        print(f"   - 设备: {config.get('device', 'auto')}")
        if 'vllm_config' in config:
            print(f"   - GPU 内存使用率: {config['vllm_config'].get('gpu_memory_utilization', 'default')}")
    except ImportError as e:
        print(f"\n3. MinerU 未正确安装: {e}")
        return

    # 4. 检查关键依赖
    print("\n4. 关键依赖检查:")
    deps = {
        'transformers': 'transformers',
        'modelscope': 'modelscope',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'paddleocr': 'paddleocr',
        'paddlepaddle': 'paddle'
    }

    for display_name, import_name in deps.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✓ {display_name}: {version}")
        except ImportError:
            print(f"   ✗ {display_name}: 未安装")

    # 5. 检查模型缓存
    print("\n5. 模型缓存目录:")
    cache_dirs = [
        Path.home() / ".cache" / "modelscope" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".mineru" / "models",
    ]

    for p in cache_dirs:
        if p.exists():
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            print(f"   ✓ {p} ({total_size / 1024 ** 3:.1f} GB)")
        else:
            print(f"   ✗ {p}")

    # 6. 系统资源
    import shutil
    total, used, free = shutil.disk_usage(Path.home())
    print(f"\n6. 磁盘空间: {free / 1024 ** 3:.1f} GB 可用")

    import psutil
    mem = psutil.virtual_memory()
    print(f"   系统内存: {mem.available / 1024 ** 3:.1f} GB 可用 / {mem.total / 1024 ** 3:.1f} GB 总计")


if __name__ == "__main__":
    check_environment()