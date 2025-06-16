"""pytest 配置文件

此文件用於設置測試環境和共享的測試夾具（fixtures）。
"""

import os
import sys
import pytest
from typing import Generator

# 添加項目根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """返回測試數據目錄的路徑"""
    return os.path.join(project_root, "tests", "data")

@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """創建臨時目錄並返回其路徑"""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="function")
def clean_registry() -> Generator[None, None, None]:
    """清理配置註冊表"""
    from core.configuration import VLMConfigRegistry
    registry = VLMConfigRegistry()
    registry.clear()
    yield
    registry.clear() 