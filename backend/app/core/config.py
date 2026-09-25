"""应用配置管理。"""
import logging
from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# 占位符 API Key（用于检测未配置情况）
_PLACEHOLDER_KEYS = {"", "sk-placeholder", "your_api_key_here", "xxxx"}


class Settings(BaseSettings):
    """应用配置，从环境变量读取。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用基础信息
    APP_NAME: str = "water-agents"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"  # development / production

    # 后端服务
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # CORS 允许来源
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # LLM 配置（OpenAI 兼容接口）
    LLM_API_KEY: str = "sk-placeholder"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"
    # 评判模型（双模型蒸馏的评判角色，用更强模型做语义质量打分）
    LLM_JUDGE_MODEL: str = "qwen-max"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    # 单次会话最大工具调用轮次（防止死循环）
    LLM_MAX_TOOL_ROUNDS: int = 5
    # Embedding 模型（用于法规 RAG 检索 + Skill 匹配）
    LLM_EMBEDDING_MODEL: str = "text-embedding-v3"
    # Embedding 独立凭证（与推理 LLM_API_KEY 分开授权/计费），留空回退到 LLM_* 配置。
    # 典型场景：推理切 DeepSeek（无 embedding API），embedding 留在 DashScope——
    # 此时必须配置 EMBEDDING_API_KEY + EMBEDDING_BASE_URL 指向 embedding 服务商
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""

    # Qdrant 向量库配置（用于法规 RAG 检索）
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "water_regulations"
    # 向量维度（需与 DashScope text-embedding-v3 一致）
    QDRANT_VECTOR_SIZE: int = 1024

    # 高德天气 API（阶段 F 接入实时天气）
    # 申请：https://lbs.amap.com/dev/key/app（Web服务 API 类型）
    # 留空则降级到 mock
    AMAP_API_KEY: str = ""
    # 吕梁市 adcode（高德城市编码）
    AMAP_CITY_CODE: str = "141100"

    # Tavily 联网搜索 API（用于 web_search 工具）
    # 申请：https://tavily.com（免费额度 1000 次/月）
    # 留空则降级到 mock
    TAVILY_API_KEY: str = ""

    # 水文数据源配置（阶段 F 接入实时水情）
    # 数据源：qqjjsj.com 每日发布黄河水文站水位流量（来源于水利部公开数据）
    HYDRO_SOURCE_URL: str = "http://www.qqjjsj.com/list226a1/"
    # 缓存 TTL（秒）：避免高频请求被封，默认 30 分钟
    HYDRO_CACHE_TTL: int = 1800

    # Rate Limiting（slowapi）
    RATE_LIMIT_PER_MINUTE: int = 30

    # 上下文 token 压缩（借鉴 Codex compact.rs）
    # history 总 token 超过此值时触发压缩。
    # 阈值依据（DeepSeek 现行模型 deepseek-flash / deepseek-v4-pro 上下文
    # 均为 1M、最大输出 384K）：容量已不是约束，绑定约束是——
    #   1) 成本与时延：planner→executor 每轮重灌 prompt，history 前缀
    #      append-only 可吃上下文缓存，但 64K 以上的常驻历史在缓存未命
    #      时 prefill 成本和 TTFT 明显上升；
    #   2) 质量：lost-in-the-middle——长中段无关对话稀释 planner 注意力，
    #      1M 窗口模型检索尚可、长上下文多跳推理仍会衰减。
    # 取 64K（窗口的 ~6%）：常驻 prompt（history + 系统/工具 schema +
    # 记忆注入 + 当轮工具结果 + 输出）峰值约 120K，远低于 1M 且 TTFT 友好；
    # 宁可早压（摘要便宜）不要长历史硬扛（质量衰减不可恢复）。
    # estimate_tokens 误差 ±20%，实际触发区间约 51K~77K。
    HISTORY_MAX_TOKENS: int = 65536
    # 压缩时保留最近几轮原文（1 轮 = 1 问 1答 = 2 条消息）。
    # 64K 预算下近窗从 2 轮放宽到 4 轮：追问类问题通常落在最近几轮，
    # 原文保留比摘要还原省一次工具往返
    HISTORY_KEEP_RECENT_ROUNDS: int = 4
    # 会话任务段落盘与按需还原（Context-Folding 工程化，
    # 详见 docs/context-compression-research.md）：
    # 超预算时早段折叠为冻结的结构化摘要（KV Cache 前缀稳定），
    # 全文（含工具数据）落 MD 文件；摘要内标注存档文件名，
    # 由 planner 按需调用 read_session_archive 工具读取（Claude Code 式
    # 摘要+磁盘指针），不再每请求做 query-段向量比较
    SESSION_ARCHIVE_ENABLED: bool = True
    SESSION_ARCHIVE_DIR: str = "session_archive"
    # 分段语义边界：相邻轮 query embedding 余弦低于阈值开新段
    SESSION_SEGMENT_SIM_THRESHOLD: float = 0.5
    # 按需还原（旧机制，默认关闭）：每请求把 query 与全部早段做 embedding
    # 匹配自动注入命中段全文。段数增多后比较成本线性增长，已改为
    # "摘要标注存档路径 + LLM 调用 read_session_archive 主动读取"；
    # 置 True 可恢复旧行为用于对照实验
    SESSION_RECALL_ON_QUERY: bool = False
    SESSION_ARCHIVE_MATCH_THRESHOLD: float = 0.5
    SESSION_ARCHIVE_TOP_K: int = 2
    SESSION_ARCHIVE_MAX_AGE_DAYS: int = 30

    # MySQL 配置（自进化长期记忆存储）
    # 留空则禁用自进化（reflection 仍可运行但记忆不持久化）
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "water_agent"
    # 自进化开关（关闭时反思循环不运行）
    SELF_EVOLUTION_ENABLED: bool = True
    # 长期记忆双层文件（对标 Claude Code CLAUDE.md + auto-memory）：
    # MEMORY_FILE 用户权威手册（Agent 只读），MEMORY_DIR Agent 自动记忆目录
    AUTO_MEMORY_ENABLED: bool = True
    MEMORY_FILE: str = "MEMORY.md"
    MEMORY_DIR: str = "memory"
    # Curator 后台治理（借鉴 Hermes Agent v0.12.0 Curator）：
    # 周期性剪枝僵尸记忆 / LLM 压缩合并 / 向量索引对账 / 写治理报告
    CURATOR_ENABLED: bool = True
    # 治理周期（小时），默认 168h = 7 天（对齐 Hermes Curator 的 7-day cycle）
    CURATOR_INTERVAL_HOURS: int = 168

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _normalize_cors(cls, v: str) -> str:
        # 统一去除空白
        return ",".join([item.strip() for item in v.split(",") if item.strip()])

    @model_validator(mode="after")
    def _validate_production_config(self) -> "Settings":
        """M13：production 模式下校验关键配置，避免使用占位符部署。"""
        if self.APP_ENV != "production":
            return self

        issues = []
        if self.LLM_API_KEY in _PLACEHOLDER_KEYS:
            issues.append("LLM_API_KEY 未配置（不能为空或占位符）")
        if self.AMAP_API_KEY in _PLACEHOLDER_KEYS:
            issues.append("AMAP_API_KEY 未配置（production 模式必需）")
        if self.BACKEND_HOST == "0.0.0.0":
            logger.warning("[config] production 模式 BACKEND_HOST=0.0.0.0，请确认网络暴露范围")

        if issues:
            raise ValueError(
                "Production 配置校验失败：\n  - " + "\n  - ".join(issues) +
                "\n请检查 .env 文件或环境变量后重启服务。"
            )
        logger.info("[config] production 配置校验通过")
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return self.CORS_ORIGINS.split(",")

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    def mask_secrets(self) -> dict:
        """M13：返回脱敏后的配置摘要，用于日志打印。"""
        def _mask(key: str) -> str:
            if not key or key in _PLACEHOLDER_KEYS:
                return "<empty>"
            if len(key) <= 8:
                return "***"
            return f"{key[:4]}***{key[-4:]}"

        return {
            "LLM_API_KEY": _mask(self.LLM_API_KEY),
            "EMBEDDING_API_KEY": _mask(self.EMBEDDING_API_KEY or self.LLM_API_KEY),
            "AMAP_API_KEY": _mask(self.AMAP_API_KEY),
            "LLM_BASE_URL": self.LLM_BASE_URL,
            "EMBEDDING_BASE_URL": self.EMBEDDING_BASE_URL or self.LLM_BASE_URL,
            "LLM_MODEL": self.LLM_MODEL,
            "QDRANT": f"{self.QDRANT_HOST}:{self.QDRANT_PORT}",
            "MYSQL": f"{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}" if self.MYSQL_PASSWORD else "<empty>",
            "SELF_EVOLUTION": self.SELF_EVOLUTION_ENABLED,
            "APP_ENV": self.APP_ENV,
        }


@lru_cache
def get_settings() -> Settings:
    """单例配置，避免重复读取 .env。"""
    settings = Settings()
    # 启动时打印脱敏配置
    logger.info("[config] loaded: %s", settings.mask_secrets())
    return settings
