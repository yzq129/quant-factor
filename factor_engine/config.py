"""
配置加载器
支持从 config/config.yaml 和 config/secrets.yaml 加载配置
可通过环境变量覆盖部分配置
"""
import os
import yaml
from pathlib import Path


class Config:
    """配置管理类"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        """加载配置文件"""
        # 项目根目录：当前文件的上级目录
        project_root = Path(__file__).parent.parent.resolve()
        
        config_path = project_root / "config" / "config.yaml"
        secrets_path = project_root / "config" / "secrets.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
        
        # 加载 secrets（如果存在）
        if secrets_path.exists():
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f) or {}
            self._merge_dicts(self._config, secrets)
        
        # 设置项目根目录
        self._config.setdefault('paths', {})
        self._config['paths']['project_root'] = str(project_root)
        
        # 环境变量覆盖
        self._apply_env_overrides()
    
    def _merge_dicts(self, base, override):
        """递归合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        # 数据库密码
        db_password = os.environ.get('DB_PASSWORD')
        if db_password:
            self._config.setdefault('database', {})['password'] = db_password
        
        # Tushare token
        ts_token = os.environ.get('TUSHARE_TOKEN')
        if ts_token:
            self._config.setdefault('tushare', {})['token'] = ts_token
    
    def get(self, key_path, default=None):
        """
        通过点分路径获取配置
        例如：get('database.host') -> 'localhost'
        """
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_database_config(self):
        """获取完整数据库配置"""
        db = self.get('database', {}).copy()
        if 'password' not in db:
            raise ValueError("Database password not found in config/secrets.yaml or env DB_PASSWORD")
        return db
    
    def get_tushare_token(self):
        """获取 Tushare token"""
        token = self.get('tushare.token')
        if not token:
            raise ValueError("Tushare token not found in config/secrets.yaml or env TUSHARE_TOKEN")
        return token
    
    def get_path(self, key):
        """获取绝对路径"""
        rel_path = self.get(f'paths.{key}', '')
        if not rel_path:
            return ''
        if os.path.isabs(rel_path):
            return rel_path
        project_root = self.get('paths.project_root')
        return os.path.join(project_root, rel_path)
    
    def all(self):
        """获取完整配置字典"""
        return self._config.copy()


# 全局配置访问函数
def get_config():
    """获取配置单例"""
    return Config()
