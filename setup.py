from setuptools import setup, find_packages

setup(
    name="gp_quant",
    version="0.1.0",
    author="gp-quant team",
    description="股票量化交易模型框架",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "akshare>=1.14.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.3.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "ml": ["torch>=2.0.0"],
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "gp-quant=gp_quant.cli.main:main",
        ],
    },
)
