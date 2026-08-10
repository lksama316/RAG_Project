import re
from pathlib import Path

# 读取你的MD文件
md_path = r"E:\AI_Study\pdf\output\联想海豚用户手册\auto\联想海豚用户手册.md"
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

# 提取所有图片引用
image_refs = re.findall(r"!.*?\(.*?\.(jpg|png|gif|bmp|jpeg|webp).*?\)", content, re.IGNORECASE)

print(f"找到 {len(image_refs)} 个图片引用")
print("\n前10个图片引用:")
for i, ref in enumerate(image_refs[:10]):
    # 获取完整匹配
    full_match = re.search(r"!.*?\(.*?" + re.escape(ref) + r".*?\)", content, re.IGNORECASE)
    if full_match:
        print(f"{i+1}. {full_match.group()[:100]}")  # 只显示前100字符

# 同时列出images文件夹中的实际文件名
images_dir = Path(r"E:\AI_Study\pdf\output\联想海豚用户手册\auto\images")
if images_dir.exists():
    actual_files = [f.name for f in images_dir.iterdir() if f.is_file()]
    print(f"\nimages文件夹中有 {len(actual_files)} 个文件")
    print(f"前5个实际文件名: {actual_files[:5]}")