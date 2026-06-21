"""
MySQL数据库操作封装
从配置文件读取连接信息
"""
import pymysql
import pandas as pd
from sqlalchemy import create_engine

from factor_engine.config import get_config


def _get_db_config():
    """获取数据库配置"""
    return get_config().get_database_config()


def get_engine():
    """创建 SQLAlchemy engine"""
    cfg = _get_db_config()
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        f"?charset={cfg.get('charset', 'utf8mb4')}"
    )
    return create_engine(url)


def get_connection():
    """创建 pymysql 连接"""
    return pymysql.connect(**_get_db_config())


def _infer_mysql_type(dtype):
    """根据 pandas dtype 推断 MySQL 列类型"""
    dtype_str = str(dtype)
    if 'int' in dtype_str:
        return 'BIGINT'
    elif 'float' in dtype_str:
        return 'DOUBLE'
    elif 'bool' in dtype_str:
        return 'TINYINT'
    elif 'datetime' in dtype_str:
        return 'DATETIME'
    else:
        return 'VARCHAR(255)'


def _create_table_from_df(df, table_name, conn):
    """根据 DataFrame 结构自动创建表"""
    col_defs = []
    for col in df.columns:
        mysql_type = _infer_mysql_type(df[col].dtype)
        col_defs.append(f"`{col}` {mysql_type}")
    
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        {', '.join(col_defs)}
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def save_dataframe(df, table_name, if_exists="append"):
    """保存DataFrame到MySQL，使用pymysql批量插入；replace 时若表不存在则自动创建"""
    if df.empty:
        print(f"[WARN] Empty dataframe, skip saving to {table_name}")
        return
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # replace 时删除并重建表，确保列结构与 DataFrame 一致
            if if_exists == "replace":
                cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                _create_table_from_df(df, table_name, conn)
            
            # 构建INSERT语句
            cols = df.columns.tolist()
            col_str = ', '.join([f"`{c}`" for c in cols])
            placeholders = ', '.join(['%s'] * len(cols))
            sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
            
            # 转换数据为列表，将NaN转为None
            data = df.astype(object).where(pd.notnull(df), None).values.tolist()
            
            # 批量插入，每批1000条
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                cur.executemany(sql, batch)
            
            conn.commit()
        print(f"[OK] Saved {len(df)} rows to {table_name}")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to save to {table_name}: {e}")
        raise
    finally:
        conn.close()


def read_sql(query):
    """执行SQL查询返回DataFrame"""
    engine = get_engine()
    return pd.read_sql(query, engine)


def get_trade_dates(start, end):
    """获取数据库中已有的交易日期列表"""
    query = f"""
    SELECT DISTINCT trade_date FROM factor_raw_daily
    WHERE trade_date BETWEEN '{start}' AND '{end}'
    ORDER BY trade_date
    """
    df = read_sql(query)
    if df.empty:
        return []
    return df["trade_date"].dt.strftime("%Y-%m-%d").tolist()


def get_latest_date(table_name):
    """获取某表最新日期"""
    query = f"SELECT MAX(trade_date) as max_date FROM {table_name}"
    df = read_sql(query)
    if df.empty or df["max_date"].isna().all():
        return None
    return df["max_date"].iloc[0].strftime("%Y-%m-%d")


def execute(query):
    """执行非查询 SQL"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
    finally:
        conn.close()
