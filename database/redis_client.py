from io import StringIO
import json
import redis
import pandas as pd
import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis_cache")
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


def save_split_data_to_redis(
    key: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
):
    """Save train-test split data to Redis."""
    try:
        split_data = {
            "x_train": x_train.to_json(),
            "x_test": x_test.to_json(),
            "y_train": y_train.to_json(),
            "y_test": y_test.to_json(),
        }

        get_redis_client().setex(
            key,
            REDIS_TTL,
            json.dumps(split_data)
        )

        print(f"Saved split data '{key}' to Redis")

    except Exception as e:
        print(f"Redis save failed: {e}")


def load_df_from_redis(key: str) -> pd.DataFrame | None:
    """Load a DataFrame from Redis."""
    try:
        data = get_redis_client().get(key)

        if not data:
            print(f"'{key}' not found in Redis")
            return None

        print(f"Loaded '{key}' from Redis")
        return pd.read_json(StringIO(data))

    except Exception as e:
        print(f"Redis load failed: {e}")
        return None


def load_split_data_from_redis(
    key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    """Load train-test split data from Redis."""
    try:
        data = get_redis_client().get(key)

        if not data:
            print(f"'{key}' not found in Redis")
            return None

        split_data = json.loads(data)

        x_train = pd.read_json(StringIO(split_data["x_train"]))
        x_test = pd.read_json(StringIO(split_data["x_test"]))
        y_train = pd.read_json(
            StringIO(split_data["y_train"]),
            typ="series"
        )
        y_test = pd.read_json(
            StringIO(split_data["y_test"]),
            typ="series"
        )

        print(f"Loaded split data '{key}' from Redis")

        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"Redis load failed: {e}")
        return None
    


def save_encoded_data_to_redis(
    key: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
):
    """Save encoded train-test data to Redis."""
    try:
        encoded_data = {
            "x_train": x_train.to_json(),
            "x_test": x_test.to_json(),
            "y_train": y_train.to_json(),
            "y_test": y_test.to_json(),
        }

        get_redis_client().setex(
            key,
            REDIS_TTL,
            json.dumps(encoded_data)
        )

        print(f"Saved encoded data '{key}' to Redis")

    except Exception as e:
        print(f"Redis save failed: {e}")


def load_encoded_data_from_redis(key: str):
    """Load encoded train-test data from Redis."""
    try:
        data = get_redis_client().get(key)

        if not data:
            print(f"'{key}' not found in Redis")
            return None

        encoded_data = json.loads(data)

        return (
            pd.read_json(StringIO(encoded_data["x_train"])),
            pd.read_json(StringIO(encoded_data["x_test"])),
            pd.read_json(StringIO(encoded_data["y_train"]), typ="series"),
            pd.read_json(StringIO(encoded_data["y_test"]), typ="series"),
        )

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