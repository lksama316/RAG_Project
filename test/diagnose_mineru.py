# diagnose_mineru_v6.py
import subprocess
import sys
import torch
import os
from pathlib import Path
import importlib.util
import importlib.metadata


def get_package_version(package_name):
    """安全获取包版本 - 使用多种方法"""
    # 方法1: 使用 importlib.metadata (Python 3.8+)
    try:
        return importlib.metadata.version(package_name)
    except:
        pass

    # 方法2: 尝试直接导入并获取 __version__
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'unknown')
    except:
        pass

    # 方法3: 尝试从 __init__.py 读取
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            init_path = Path(spec.origin)
            if init_path.exists():
                with open(init_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import re
                    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
    except:
        pass

    return 'unknown'


def get_installed_packages():
    """获取已安装的包列表 - 不依赖 pkg_resources"""
    packages = {}
    try:
        # 使用 pip 列出已安装的包
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    packages[name.lower()] = version
    except:
        pass
    return packages


def check_mineru_installation():
    """详细检查 MinerU 安装"""
    print("\n" + "=" * 60)
    print("3. MinerU 详细检查")
    print("=" * 60)

    try:
        import mineru
        print(f"   ✓ MinerU 已安装")
        print(f"   路径: {mineru.__file__}")

        # 获取版本
        version = get_package_version('mineru')
        print(f"   版本: {version}")

        # 检查 MinerU 的子模块
        mineru_dir = Path(mineru.__file__).parent
        print(f"   安装目录: {mineru_dir}")

        # 列出 MinerU 的主要模块
        modules = []
        for item in mineru_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                modules.append(item.name)
            elif item.suffix == '.py' and not item.name.startswith('_'):
                modules.append(item.stem)

        if modules:
            print(f"   主要模块: {', '.join(sorted(modules)[:15])}")

        # 检查关键配置文件
        print("   配置文件检查:")
        config_files = [
            mineru_dir / 'config.py',
            mineru_dir / 'settings.py',
            mineru_dir / 'utils' / 'config.py',
            mineru_dir / 'utils' / 'settings.py',
            mineru_dir / 'common' / 'config.py',
        ]

        config_found = False
        for cfg in config_files:
            if cfg.exists():
                print(f"     ✓ {cfg.relative_to(mineru_dir.parent)}")
                config_found = True

        if not config_found:
            print(f"     ℹ️  未找到标准配置文件")

        # 尝试导入配置
        print("   配置导入测试:")
        config_imports = [
            ('mineru.config', 'Config'),
            ('mineru.settings', 'Settings'),
            ('mineru.utils.config', 'get_default_config'),
            ('mineru.common.config', 'Config'),
        ]

        for module_name, attr_name in config_imports:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                if hasattr(module, attr_name):
                    print(f"     ✓ from {module_name} import {attr_name}")
                else:
                    print(f"     ⚠️  {module_name} 存在但没有 {attr_name}")
            except ImportError as e:
                print(f"     ✗ {module_name}: {str(e).split(':')[0]}")

        # 检查 MinerU 的入口点
        print("   入口点检查:")
        try:
            import mineru.cli
            print(f"     ✓ mineru.cli 可用")
        except ImportError:
            print(f"     ✗ mineru.cli 不可用")

        return True

    except ImportError as e:
        print(f"   ✗ MinerU 未安装: {e}")
        print(f"   💡 安装: pip install mineru")
        return False


def check_mineru_commands():
    """检查 MinerU 命令行工具"""
    print("\n" + "=" * 60)
    print("4. MinerU 命令行工具")
    print("=" * 60)

    commands = ['mineru', 'mineru-cli', 'magic-pdf']
    found_any = False

    for cmd in commands:
        try:
            import shutil
            path = shutil.which(cmd)
            if path:
                print(f"   ✓ {cmd}: {path}")
                found_any = True
                # 尝试获取版本
                try:
                    result = subprocess.run([cmd, '--version'],
                                            capture_output=True,
                                            text=True,
                                            timeout=3)
                    if result.stdout:
                        print(f"     -> {result.stdout.strip()}")
                except:
                    pass
            else:
                print(f"   ✗ {cmd}: 未找到")
        except:
            print(f"   ✗ {cmd}: 检查失败")

    if not found_any:
        print(f"   💡 确保 MinerU 正确安装，或使用 python -m mineru 运行")


def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("5. 关键依赖检查")
    print("=" * 60)

    deps = {
        'transformers': 'transformers',
        'modelscope': 'modelscope',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'paddleocr': 'paddleocr',
        'paddlepaddle': 'paddle',
        'vllm': 'vllm',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'click': 'click',
        'pdfplumber': 'pdfplumber',
        'pypdf': 'pypdf',
    }

    missing = []
    installed = get_installed_packages()

    for pkg_name, import_name in deps.items():
        try:
            if import_name == 'cv2':
                import cv2
                version = cv2.__version__
            elif import_name == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')

            # 获取实际版本号
            if version == 'unknown' and pkg_name in installed:
                version = installed[pkg_name]

            print(f"   ✓ {pkg_name}: {version}")
        except ImportError:
            print(f"   ✗ {pkg_name}: 未安装")
            missing.append(pkg_name)

    if missing:
        print(f"\n   ⚠️  缺失 {len(missing)} 个依赖:")
        for m in missing:
            print(f"     - {m}")
        print(f"\n   💡 安装命令: pip install {' '.join(missing)}")
    else:
        print(f"\n   ✓ 所有必需依赖已安装")

    return missing


def check_model_caches():
    """检查模型缓存"""
    print("\n" + "=" * 60)
    print("6. 模型缓存目录")
    print("=" * 60)

    # Windows 和 Linux 缓存路径
    home = Path.home()
    cache_paths = [
        home / ".cache" / "modelscope" / "hub",
        home / ".cache" / "huggingface" / "hub",
        home / ".mineru" / "models",
        home / ".cache" / "torch" / "hub",
        home / "AppData" / "Local" / "modelscope" / "hub",
        home / "AppData" / "Local" / "huggingface" / "hub",
        home / "AppData" / "Roaming" / "mineru" / "models",
    ]

    found_caches = 0
    total_size = 0

    for p in cache_paths:
        if p.exists() and p.is_dir():
            try:
                files = list(p.rglob('*'))
                file_count = len([f for f in files if f.is_file()])
                if file_count > 0:
                    dir_size = sum(f.stat().st_size for f in files if f.is_file())
                    size_gb = dir_size / 1024 ** 3
                    print(f"   ✓ {p}")
                    print(f"     - 文件数: {file_count}")
                    print(f"     - 大小: {size_gb:.2f} GB")
                    found_caches += 1
                    total_size += dir_size
                else:
                    print(f"   ✓ {p} (空目录)")
            except Exception as e:
                print(f"   ⚠️  {p} (无法读取: {e})")
        else:
            # 不显示不存在的路径，避免信息过多
            pass

    if found_caches == 0:
        print(f"   ℹ️  未找到模型缓存")
        print(f"   💡 首次运行 MinerU 会自动下载模型到缓存目录")
    else:
        print(f"\n   总计缓存: {total_size / 1024 ** 3:.2f} GB")


def check_system_resources():
    """检查系统资源"""
    print("\n" + "=" * 60)
    print("7. 系统资源")
    print("=" * 60)

    import shutil

    # 磁盘空间
    try:
        # 获取当前工作目录的磁盘使用情况
        current_dir = Path.cwd()
        total, used, free = shutil.disk_usage(current_dir)
        print(f"   磁盘空间 ({current_dir}):")
        print(f"     - 总空间: {total / 1024 ** 3:.1f} GB")
        print(f"     - 可用: {free / 1024 ** 3:.1f} GB")
        print(f"     - 使用率: {((total - free) / total * 100):.1f}%")

        if free / 1024 ** 3 < 10:
            print(f"     ⚠️  可用空间不足 10GB，可能影响模型下载")
    except:
        print(f"   磁盘空间: 无法获取")

    # 系统内存 - 使用更通用的方法
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n   系统内存:")
        print(f"     - 总计: {mem.total / 1024 ** 3:.1f} GB")
        print(f"     - 可用: {mem.available / 1024 ** 3:.1f} GB")
        print(f"     - 使用率: {mem.percent}%")

        if mem.available / 1024 ** 3 < 8:
            print(f"     ⚠️  可用内存不足 8GB")

        # CPU 信息
        print(f"\n   CPU:")
        print(f"     - 物理核心: {psutil.cpu_count(logical=False)}")
        print(f"     - 逻辑核心: {psutil.cpu_count(logical=True)}")
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"     - 使用率: {cpu_percent}%")

    except ImportError:
        print(f"\n   ℹ️  psutil 未安装，跳过详细系统检测")
        print(f"   💡 安装: pip install psutil")
    except Exception as e:
        print(f"\n   ⚠️  系统资源检测失败: {e}")


def check_environment_variables():
    """检查环境变量"""
    print("\n" + "=" * 60)
    print("8. 环境变量")
    print("=" * 60)

    env_vars = {
        'CUDA_VISIBLE_DEVICES': 'GPU 设备',
        'TORCH_CUDA_ARCH_LIST': 'CUDA 架构',
        'PYTORCH_CUDA_ALLOC_CONF': 'PyTorch 内存分配',
        'OMP_NUM_THREADS': 'OpenMP 线程',
        'MKL_NUM_THREADS': 'MKL 线程',
        'MINERU_MODEL_DIR': 'MinerU 模型目录',
        'MODELSCOPE_CACHE': 'ModelScope 缓存',
        'HF_HOME': 'HuggingFace 缓存',
        'HF_ENDPOINT': 'HF 镜像端点',
    }

    set_count = 0
    for var, desc in env_vars.items():
        value = os.environ.get(var)
        if value is not None:
            print(f"   ✓ {var} = {value}")
            print(f"     -> {desc}")
            set_count += 1
        else:
            # 只在有重要变量未设置时提示
            pass

    if set_count == 0:
        print(f"   ℹ️  未设置 MinerU 相关环境变量（使用默认配置）")

    if 'HF_ENDPOINT' not in os.environ:
        print(f"\n   💡 如果下载模型慢，可设置镜像:")
        print(f"      set HF_ENDPOINT=https://hf-mirror.com")


def main():
    print("=" * 70)
    print(" MinerU 完整环境诊断 v6")
    print("=" * 70)

    # 1. Python 环境
    print(f"\n1. Python 环境:")
    print(f"   版本: {sys.version.split()[0]}")
    print(f"   路径: {sys.executable}")
    print(f"   平台: {sys.platform}")

    # 2. PyTorch & CUDA
    print(f"\n2. PyTorch & CUDA:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   显卡: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024 ** 3
        print(f"   显存: {total_mem:.1f} GB")

        if total_mem < 8:
            print(f"   ⚠️  显存 {total_mem:.1f}GB < 推荐 8GB")
            print(f"   💡 建议使用 CPU 模式或降低配置")
    else:
        print(f"   ℹ️  CUDA 不可用，将使用 CPU 模式")

    # 3-8. 详细检查
    check_mineru_installation()
    check_mineru_commands()
    check_dependencies()
    check_model_caches()
    check_system_resources()
    check_environment_variables()

    # 9. 总结和建议
    print("\n" + "=" * 70)
    print(" 📋 总结与建议")
    print("=" * 70)

    # 检查 MinerU 是否可用
    try:
        import mineru
        mineru_ok = True
    except:
        mineru_ok = False

    if mineru_ok:
        print("✅ MinerU 已安装")

        # GPU 建议
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if total_mem < 8:
                print("\n1. 针对 6GB 显存的优化配置:")
                print("   - 使用 CPU 模式（推荐）:")
                print("     mineru --device cpu /path/to/pdf")
                print("   - 或 GPU 小批量模式:")
                print("     set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256")
                print("     mineru --device cuda --batch_size 1 /path/to/pdf")
        else:
            print("\n1. 使用 CPU 模式:")
            print("   mineru --device cpu /path/to/pdf")

        print("\n2. 常用命令:")
        print("   - 处理单个文件: mineru /path/to/file.pdf")
        print("   - 处理目录: mineru /path/to/folder -o /output/dir")
        print("   - 指定设备: mineru /path/to/file.pdf --device cuda")

        print("\n3. 首次使用注意事项:")
        print("   - 首次运行会自动下载模型 (约 3-5GB)")
        print("   - 确保网络连接稳定")
        print("   - 可设置镜像加速下载: set HF_ENDPOINT=https://hf-mirror.com")

    else:
        print("❌ MinerU 未正确安装")
        print("\n安装步骤:")
        print("1. pip install mineru")
        print("2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. 验证: python -c 'import mineru; print(\"OK\")'")

    print("\n" + "=" * 70)
    print(" ✅ 诊断完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()