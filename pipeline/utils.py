"""Pipeline shared utilities"""
import os

POOL_NAMES = {
    "small": "stock_pool_small.txt",
    "500": "stock_pool_500.txt",
    "full": "stock_pool.txt",
}


def get_pool_file(pool: str) -> str:
    """Get path to stock pool file."""
    return os.path.join(os.path.dirname(__file__), "data", POOL_NAMES[pool])


def get_pool_input_path(pool: str) -> str:
    """Get path to step1 output CSV for a given pool."""
    return os.path.join(os.path.dirname(__file__), "data", f"stock_pool_{pool}.csv")
