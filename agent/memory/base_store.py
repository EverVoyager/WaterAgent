"""MySQL store 公共基类：事务型连接 + 自动建表 + 降级模式。

从 session_store.py 已验证的实现抽取（with 体成功 commit、异常 rollback、
最后 close；幂等建表；MySQL 未配置时 enabled=False）。
"""
import logging
import threading
from contextlib import contextmanager, suppress

logger = logging.getLogger(__name__)


class BaseStore:
    """MySQL 存储基类。线程安全（每次操作独立连接）。

    子类需提供 _create_sqls: list[str]（幂等建表语句）。
    降级策略：MySQL 不可用时抛 RuntimeError（硬失败）；
    若需软失败请在调用层捕获。
    """

    _create_sqls: list[str] = []

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self._config = {
            "host": host, "port": port, "user": user,
            "password": password, "database": database,
            "charset": "utf8mb4",
        }
        self._initialized = False
        self._init_lock = threading.Lock()
        self._enabled = bool(password)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def _get_conn(self):
        """事务型连接上下文管理器：with 体成功自动 commit，异常 rollback，最后 close。"""
        if not self._enabled:
            raise RuntimeError(f"{type(self).__name__} 未启用（MYSQL_PASSWORD 为空）")
        import pymysql
        import pymysql.cursors
        conn = None
        try:
            conn = pymysql.connect(**self._config, cursorclass=pymysql.cursors.DictCursor)
            yield conn
            conn.commit()
        except Exception:
            if conn:
                with suppress(Exception):
                    conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _ensure_tables(self) -> None:
        """首次使用时自动建表（线程安全，只执行一次）。"""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            if not self._enabled:
                raise RuntimeError(f"{type(self).__name__} 未启用（MYSQL_PASSWORD 为空）")
            try:
                with self._get_conn() as conn, conn.cursor() as cur:
                    for sql in self._create_sqls:
                        cur.execute(sql)
                logger.info("[store] %s 表已就绪", type(self).__name__)
                self._initialized = True
            except Exception as e:
                logger.error("[store] %s 建表失败：%s", type(self).__name__, e)
                raise RuntimeError(f"{type(self).__name__} 建表失败：{e}") from e
