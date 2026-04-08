"""
数据存储模块
"""
import os
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime


class DataStorage:
    """股票数据存储类"""

    def __init__(self, data_dir: str = "./data"):
        """
        初始化存储

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def save(self, symbol: str, data: pd.DataFrame, overwrite: bool = False) -> bool:
        """
        保存数据到 CSV

        Args:
            symbol: 股票代码
            data: DataFrame 数据
            overwrite: 是否覆盖现有文件

        Returns:
            保存成功与否
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"Data already exists for {symbol}. Use overwrite=True to replace.")

        data.to_csv(file_path, index=False)
        return True

    def load(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从 CSV 加载数据

        Args:
            symbol: 股票代码

        Returns:
            DataFrame 或 None
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if not os.path.exists(file_path):
            return None

        return pd.read_csv(file_path)

    def delete(self, symbol: str) -> bool:
        """删除股票数据"""
        import os
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def list_symbols(self) -> list:
        """获取已存储的所有股票代码列表"""
        files = os.listdir(self.data_dir) if os.path.exists(self.data_dir) else []
        return [f.rsplit(".", 1)[0] for f in files if f.endswith(".csv")]

    def get_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取数据元信息"""
        df = self.load(symbol)
        if df is None:
            return None

        return {
            "symbol": symbol,
            "start_date": df["date"].min() if "date" in df.columns else None,
            "end_date": df["date"].max() if "date" in df.columns else None,
            "records_count": len(df),
            "columns": list(df.columns),
            "last_updated": datetime.now().isoformat()
        }


def create_storage(data_dir: str = "./data") -> DataStorage:
    """创建存储对象"""
    return DataStorage(data_dir)
