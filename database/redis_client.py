import redis
import pandas as pd
import os


REDIS_HOST = os.getenv("REDIS_HOST", "redis_cache")  # fixed default
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL = 60 * 60 * 24  # 24 hours


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True
    )


def save_df_to_redis(key: str, df: pd.DataFrame):
    """Save a DataFrame to Redis."""
    try:
        get_redis_client().setex(key, REDIS_TTL, df.to_json())
        print(f"Saved '{key}' to Redis")
    except Exception as e:
        print(f"Redis save failed: {e}")


def load_df_from_redis(key: str):
    """Load a DataFrame from Redis."""
    try:
        data = get_redis_client().get(key)

        if not data:
            print(f"'{key}' not found in Redis")
            return None

        print(f"Loaded '{key}' from Redis")
        return pd.read_json(data)

    except Exception as e:
        print(f"Redis load failed: {e}")
        return None


def delete_key(key: str):
    """Delete a Redis key."""
    try:
        get_redis_client().delete(key)
        print(f"Deleted '{key}' from Redis")
    except Exception as e:
        print(f"Redis delete failed: {e}")


def cache_exists(key: str) -> bool:
    """Check whether a Redis key exists."""
    try:
        return bool(get_redis_client().exists(key))
    except Exception:
        return False