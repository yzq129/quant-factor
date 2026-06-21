"""
统一策略运行入口
示例：
    python scripts/run_pipeline.py original
    python scripts/run_pipeline.py mined
    python scripts/run_pipeline.py pure_ic
    python scripts/run_pipeline.py all
"""
import sys
import os

# 把项目根目录加入 Python 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from factor_engine.strategy.original import OriginalStrategy
from factor_engine.strategy.mined import MinedStrategy
from factor_engine.strategy.pure_ic import PureICStrategy


STRATEGIES = {
    'original': OriginalStrategy,
    'mined': MinedStrategy,
    'pure_ic': PureICStrategy,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pipeline.py <strategy_name|all>")
        print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)
    
    name = sys.argv[1].lower()
    
    if name == 'all':
        for n, cls in STRATEGIES.items():
            print(f"\n### Running {n} strategy ###")
            strategy = cls()
            strategy.run()
    elif name in STRATEGIES:
        strategy = STRATEGIES[name]()
        strategy.run()
    else:
        print(f"Unknown strategy: {name}")
        print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)


if __name__ == '__main__':
    main()
