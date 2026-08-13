# test/bge_test.py
from utils.embedding_utils import generate_embeddings, get_bge_m3_ef
import os


def test_model():
    print("正在加载模型...")
    try:
        model = get_bge_m3_ef()
        print("✓ 模型加载成功!")

        # 测试编码
        texts = ['hello', '我很开心', '今天天气不错']
        embeddings = model.encode_documents(texts)
        print(f"✓ 生成嵌入成功")
        print(f"  稠密向量维度: {len(embeddings['dense'][0])}")
        print(f"  稀疏向量形状: {embeddings['sparse'].shape}")

        # 测试 generate_embeddings 函数
        result = generate_embeddings(texts)
        print(f"✓ generate_embeddings 成功")
        print(f"  第一个文本的稠密向量维度: {len(result['dense'][0])}")
        print(f"  第一个文本的稀疏向量: {list(result['sparse'][0].keys())[:5]}...")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()