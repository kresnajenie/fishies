from minio import Minio

# Create client with access key and secret key with specific region.
client = Minio(
    "192.168.64.2:9000",
    access_key="5xtqqLRKsyVqCxsENyv1",
    secret_key="T4jitgsR4QBgWnZ0vktNoUB3OcKjZjKKTssWi0de",
    region="bintulab",
    secure=False
)


response = None
try:
    # response = client.get_object("test-data", "cbm2_labeled.h5ad")
    response = client.get_object("test-data", "cbm2_labeled.h5ad")
    # Read data from response.
    print(response.data)
finally:
    response = None