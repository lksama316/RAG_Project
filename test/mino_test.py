# test_minio_connection.py
from minio import Minio
from config.minio_config import minio_config


def test_minio_connection():
    try:
        client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=False
        )
        print(f"✅ MinIO 连接成功: {minio_config.endpoint}")

        # 列出所有 bucket
        buckets = client.list_buckets()
        print(f"可用的 buckets: {[b.name for b in buckets]}")

        # 检查目标 bucket
        if client.bucket_exists(minio_config.bucket_name):
            print(f"✅ Bucket '{minio_config.bucket_name}' 存在")
        else:
            print(f"⚠️ Bucket '{minio_config.bucket_name}' 不存在")

        return client
    except Exception as e:
        print(f"❌ MinIO 连接失败: {e}")
        return None


if __name__ == "__main__":
    test_minio_connection()