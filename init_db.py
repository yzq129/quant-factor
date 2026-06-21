"""
初始化MySQL因子库表结构
"""
import pymysql

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "Qiqi20102",
    "database": "vnpy",
    "charset": "utf8mb4"
}

FACTOR_COLS = [
    "bp_lr", "ep_deducted_ttm", "fcfp_ttm", "ocfp_ttm",
    "amount_mean_20d", "asset_ln", "revenues_ln",
    "currentratio", "ocf_to_operating_profit",
    "price_chg1200d", "price_chg120d", "price_chg180d",
    "capex2sales", "netincome_chg1y", "op_profit_chg1y"
]

CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS csi500_constituents (
    trade_date DATE NOT NULL,
    code VARCHAR(10) NOT NULL,
    PRIMARY KEY (trade_date, code),
    INDEX idx_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS factor_raw_daily (
    trade_date DATE NOT NULL,
    code VARCHAR(10) NOT NULL,
    {', '.join([f'{c} FLOAT' for c in FACTOR_COLS])},
    market_cap FLOAT,
    future_5d_return FLOAT,
    PRIMARY KEY (trade_date, code),
    INDEX idx_date (trade_date),
    INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS factor_processed_daily (
    trade_date DATE NOT NULL,
    code VARCHAR(10) NOT NULL,
    {', '.join([f'{c}_neu FLOAT' for c in FACTOR_COLS])},
    market_cap FLOAT,
    PRIMARY KEY (trade_date, code),
    INDEX idx_date (trade_date),
    INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS factor_ic_monthly (
    month_end DATE NOT NULL,
    factor_name VARCHAR(30) NOT NULL,
    ic_value FLOAT,
    p_value FLOAT,
    PRIMARY KEY (month_end, factor_name),
    INDEX idx_factor (factor_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS factor_selected (
    update_date DATE NOT NULL,
    factor_name VARCHAR(30) NOT NULL,
    weight FLOAT,
    category VARCHAR(10),
    PRIMARY KEY (update_date, factor_name),
    INDEX idx_factor (factor_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS stock_score_daily (
    trade_date DATE NOT NULL,
    code VARCHAR(10) NOT NULL,
    score FLOAT,
    rank_in_pool INT,
    PRIMARY KEY (trade_date, code),
    INDEX idx_date (trade_date),
    INDEX idx_score (score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def init_database():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            for stmt in CREATE_TABLES_SQL.strip().split(';\n'):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            conn.commit()
        print("[OK] MySQL因子库表结构创建成功")
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()
