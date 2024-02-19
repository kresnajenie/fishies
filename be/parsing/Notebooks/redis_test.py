import redis

# Connect to Redis
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
# redis_client = redis.StrictRedis(host='redis-14060.c326.us-east-1-3.ec2.cloud.redislabs.com', port=14060, db=0, password="qsFkNCM7CxGBQNliVx2ZmVTVcKoihLvR")
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
# redis_client = redis.StrictRedis(host='192.168.64.2', port=6379, db=0)

# redis_client = redis.Redis(
#   host='usw1-liked-emu-33798.upstash.io',
#   port=33798,
#   password='1d9190660a1042079c224cdd0bc0fff0'
# )
# Function to save data to Redis
def save_to_redis(key, value):
    redis_client.set(key, value)

# Function to retrieve data from Redis
def retrieve_from_redis(key):
    return redis_client.get(key)

# Example usage
# retrieved_value = retrieve_from_redis('test_key')

# if retrieved_value:
#     print(f"Retrieved value: {retrieved_value.decode('utf-8')}")
# else:
#     print("No value found for the key.")

if __name__ == "__main__":
    save_to_redis("test", "hello")