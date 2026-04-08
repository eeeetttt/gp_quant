"""
CLI 命令行工具
"""
import click
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gp_quant.data.fetcher import create_fetcher
from gp_quant.data.processor import create_processor
from gp_quant.data.storage import create_storage
from gp_quant.strategy.base import SimpleMovingAverageStrategy, MomentumStrategy
from gp_quant.strategy.indicators import TechnicalIndicators


@click.group()
@click.version_option(version="0.1.0", prog_name="gp-quant")
def main():
    """gp-quant 股票量化交易框架"""
    pass


@main.command()
@click.argument("symbol", type=str, default="000001.SZ")
@click.option("--start-date", type=str, default="2023-01-01", help="开始日期")
@click.option("--end-date", type=str, default="2024-01-01", help="结束日期")
@click.option("--source", type=str, default="akshare", help="数据源 (akshare, local)")
@click.option("--output", type=str, default=None, help="输出文件路径")
def fetch(symbol, start_date, end_date, source, output):
    """获取股票数据"""
    try:
        fetcher = create_fetcher(source=source)
        data = fetcher.fetch(symbol, start_date, end_date)

        click.echo(f"成功获取 {symbol} 的数据")
        click.echo(f"日期范围：{data['date'].min()} 至 {data['date'].max()}")
        click.echo(f"数据条数：{len(data)}")

        if output:
            data.to_csv(output, index=False)
            click.echo(f"已保存到 {output}")
        else:
            click.echo(data.head(10).to_string())

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("input-file", type=click.Path(exists=True), default=None)
@click.argument("symbol", type=str, default="000001.SZ")
@click.option("--indicator", type=str, multiple=True, help="技术指标")
@click.option("--output", type=str, default=None, help="输出文件路径")
def analyze(symbol, input_file, indicator, output):
    """分析技术指标"""
    try:
        import pandas as pd

        # 加载数据
        if input_file:
            df = pd.read_csv(input_file)
        else:
            fetcher = create_fetcher(source="akshare")
            df = fetcher.fetch(symbol, "2023-01-01", "2024-01-01")

        # 计算指标
        indicators = TechnicalIndicators(df)

        for ind in indicator:
            df = indicators.add_indicator(ind)
            click.echo(f"已计算指标：{ind}")

        # 输出结果
        if output:
            df.to_csv(output, index=False)
            click.echo(f"已保存到 {output}")
        else:
            click.echo(df.tail(10).to_string())

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option("--strategy", type=str, default="sma", help="策略类型 (sma, momentum)")
@click.option("--fast-window", type=int, default=5, help="快线周期")
@click.option("--slow-window", type=int, default=20, help="慢线周期")
def strategy(strategy, fast_window, slow_window):
    """创建策略"""
    try:
        if strategy == "sma":
            strat = SimpleMovingAverageStrategy(fast_window=fast_window, slow_window=slow_window)
        elif strategy == "momentum":
            strat = MomentumStrategy(lookback_period=20, threshold=2.0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        click.echo(f"策略名称：{strat.name}")
        click.echo(f"策略参数：{strat.get_parameters()}")

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("symbol", type=str, default="000001.SZ")
@click.option("--start-date", type=str, default="2023-01-01", help="开始日期")
@click.option("--end-date", type=str, default="2024-01-01", help="结束日期")
@click.option("--initial-capital", type=float, default=100000.0, help="初始资金")
@click.option("--output", type=str, default=None, help="输出文件路径")
def backtest(symbol, start_date, end_date, initial_capital, output):
    """运行回测"""
    try:
        from gp_quant.backtest.engine import BacktestEngine

        # 获取数据
        fetcher = create_fetcher(source="akshare")
        market_data = fetcher.fetch(symbol, start_date, end_date)

        # 计算指标
        indicators = TechnicalIndicators(market_data)
        indicators.add_indicator("rsi")
        indicators.add_indicator("macd")
        indicators.add_indicator("bollinger")
        df = indicators.get_all_indicators()

        # 生成信号
        df["signal"] = "HOLD"
        rsi_signals = df["rsi"].apply(lambda x: "BUY" if x < 30 else ("SELL" if x > 70 else "HOLD"))
        df["signal"] = rsi_signals

        # 运行回测
        engine = BacktestEngine(initial_capital=initial_capital)
        engine.set_market_data(df)
        engine.set_signals(df)
        results = engine.run()

        # 输出结果
        click.echo("\n" + "=" * 50)
        click.echo("回测结果")
        click.echo("=" * 50)
        click.echo(f"总交易次数：{results['total_trades']}")
        click.echo(f"胜率：{results['win_rate']:.2f}%")
        click.echo(f"总盈亏：{results['total_pnl']:.2f}")
        click.echo(f"总收益率：{results['total_return']:.2f}%")
        click.echo(f"夏普比率：{results['sharpe_ratio']:.4f}")
        click.echo(f"最大回撤：{results['max_drawdown']:.2f}%")

        if output:
            import json
            with open(output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            click.echo(f"\n结果已保存到 {output}")

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--feature-config", type=str, default=None, help="特征配置 JSON 文件")
@click.option("--output", type=str, default=None, help="输出文件路径")
def features(feature_config, output):
    """特征工程"""
    try:
        from gp_quant.ml.features import FeatureEngineer, FeatureConfig
        import pandas as pd
        import json

        # 加载数据
        fetcher = create_fetcher(source="akshare")
        df = fetcher.fetch("000001.SZ", "2023-01-01", "2024-01-01")

        # 特征工程
        config = FeatureConfig()
        if feature_config:
            with open(feature_config) as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    setattr(config, key, value)

        engineer = FeatureEngineer(config)
        features_df = engineer.create_features(df)

        # 输出结果
        click.echo(f"特征数量：{len(features_df.columns) - 1}")
        click.echo(f"目标变量：{config.target_column}")

        if output:
            features_df.to_csv(output, index=False)
            click.echo(f"已保存到 {output}")
        else:
            click.echo(features_df.head(10).to_string())

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option("--model-type", type=str, default="random_forest", help="模型类型")
@click.option("--train-ratio", type=float, default=0.8, help="训练集比例")
@click.option("--epochs", type=int, default=100, help="训练轮数")
def train(model_type, train_ratio, epochs):
    """训练模型"""
    try:
        from gp_quant.ml.trainer import ModelTrainer, TrainConfig
        from gp_quant.ml.features import FeatureEngineer, FeatureConfig
        import pandas as pd
        import numpy as np

        # 获取数据并创建特征
        fetcher = create_fetcher(source="akshare")
        df = fetcher.fetch("000001.SZ", "2023-01-01", "2024-01-01")

        engineer = FeatureEngineer(FeatureConfig(target_horizon=5))
        features_df = engineer.create_features(df)

        # 准备数据
        X, y = engineer.prepare_data(features_df)

        # 训练模型
        config = TrainConfig(
            model_type=model_type,
            task_type="classification",
            train_ratio=train_ratio,
            epochs=epochs,
            random_state=42
        )
        trainer = ModelTrainer(config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)

        click.echo(f"训练集大小：{X_train_scaled.shape[0]}")
        click.echo(f"测试集大小：{X_test_scaled.shape[0]}")

        trainer.train(X_train_scaled, y_train)

        # 评估
        metrics = trainer.evaluate(X_test_scaled, y_test)
        click.echo("\n模型评估结果:")
        for key, value in metrics.items():
            click.echo(f"{key}: {value:.4f}")

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--model-path", type=str, default="./models/best_model.pkl", help="模型文件路径")
def predict(model_path):
    """预测"""
    try:
        from gp_quant.ml.predictor import ModelPredictor
        import json

        predictor = ModelPredictor(model_path=model_path)

        click.echo(f"模型信息：{json.dumps(predictor.get_model_info(), ensure_ascii=False, indent=2)}")

    except Exception as e:
        click.echo(f"错误：{str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
