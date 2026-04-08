from setuptools import setup, find_packages

setup(
    name="gp_quant",
    version="0.1.0",
    author="gp-quant team",
    description="股票量化交易模型框架",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "ta-lib>=0.4.24",
        "yfinance>=0.1.70",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
        ]
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "gp-quant=cli.main:main",
        ],
    },
)
