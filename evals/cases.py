"""评估数据集：5 类参数化用例 + 能力标签 + 种子隔离。

书中方法论落地：
- 数据集设计原则：真实性与可控性平衡（mock overrides 精确控制环境状态）、
  参数化模板生成防记忆（同模板多变体，AndroidWorld 式）、
  复杂度层次化（5 类用例覆盖不同能力面）、陷阱任务（τ²-bench trap tasks）。
- 评估集与训练集严格隔离：种子区间 [300_000, 400_000)，与 SFT/GRPO/裸模型
  评估三段零重叠（assert_seed_isolation 断言）。
- 能力标签：每个用例标注所需能力，报告按"任务 × 能力"交叉分类，
  诊断结构性短板而非只看总分。

用例类型：
- business   业务研判：等级真值由 overrides 档位决定（与规则引擎同源）
- chitchat   闲聊意图：期望不调任何工具
- regulation 法规问答：期望调 search_regulation
- web_search 联网检索：期望调 web_search 且引用可溯源
- trap       陷阱任务：用户口头声称某等级但数据为Ⅳ级，期望坚持数据锚定
"""
import random
from dataclasses import dataclass, field

from train.data_gen.scenario import (
    _LEVEL_CN,
    STATIONS,
    _make_overrides,
)

# 种子区间约定（与 train/data_gen/scenario.py 的三段约定衔接）：
#   SFT [0, 100_000) | GRPO [100_000, 200_000) | 裸模型评估 [200_000, 300_000)
#   系统级评估（本包）[300_000, 400_000) —— 严格隔离，防评估集泄漏进训练
EVAL_SEED_BASE = 300_000
EVAL_SEED_LIMIT = 400_000

# 训练与裸模型评估的种子区间（隔离断言用）
TRAIN_SEED_RANGES: tuple[tuple[int, int], ...] = (
    (0, 100_000), (100_000, 200_000), (200_000, 300_000),
)

CASE_TYPES = ("business", "chitchat", "regulation", "web_search", "trap")

# 能力标签体系（报告按此出"任务 × 能力"矩阵）
CAP_LEVEL = "level_decision"        # 等级判定
CAP_TOOLS = "tool_selection"        # 工具选择
CAP_CITATION = "citation"           # 引用溯源
CAP_INTENT = "intent"               # 意图识别
CAP_RESIST = "misdirection_resistance"  # 抗误导（陷阱任务专属）

# 工具集合分组（期望工具集定义用）
_DATA_TOOLS = frozenset({"get_weather", "get_hydrology", "predict_runoff"})
_DATA_OR_PLAN = _DATA_TOOLS | {"generate_plan", "search_regulation", "query_gis_terrain"}
# list_skills 是元工具（MCP tools/list 式能力发现），调用属于合法行为而非越界
# ——第 2 轮评估发现部分模型（qwen3.8-flash）常规调用它，allowed 集合需豁免
_META_TOOLS = frozenset({"list_skills"})


@dataclass
class EvalCase:
    """单条评估用例。

    Attributes:
        required_tools: 必须全部出现的工具集合（tool recall 分母）
        required_any:   至少出现一个的工具集合
        allowed_tools:  允许出现的工具集合（None=不限制；出现集合外的工具记 precision 失败）
        expected_level: 期望预警等级（None=不检查等级）
        claimed_level:  陷阱任务中用户口头声称的等级（仅 trap 用例有）
        capabilities:   能力标签列表
    """
    case_id: str
    case_type: str
    query: str
    seed: int
    overrides: dict = field(default_factory=dict)
    required_tools: frozenset = frozenset()
    required_any: frozenset = frozenset()
    allowed_tools: frozenset | None = None
    expected_level: str | None = None
    expected_intent: str = "agent_task"
    claimed_level: str | None = None
    capabilities: tuple = ()

    def to_env(self) -> dict:
        """回放环境参数（replay.py 消费）。"""
        return {"overrides": self.overrides, "seed": self.seed}


# ====== 模板库（参数化生成，同模板多变体防记忆） ======

_CHITCHAT_QUERIES = [
    "你好，你是谁？能帮我做什么？",
    "今天天气真好，心情不错。",
    "帮我写一首关于黄河的短诗。",
    "给我讲个简短的笑话吧。",
    "谢谢你，辛苦了，再见！",
    "你是怎么工作的？",
    "晚安，明天见。",
    "你觉得人工智能会取代人类吗？随便聊聊。",
    "我现在有点累，陪我聊两句。",
    "早上好！新的一天加油。",
]

_REGULATION_QUERIES = [
    "《防洪法》对宣布进入紧急防汛期是怎么规定的？",
    "启动Ⅱ级应急响应需要满足什么条件，法规依据是什么？",
    "黄河防汛预案对人员转移安置有什么要求？",
    "汛期水库调度有哪些法规依据？",
    "国家防汛抗旱工作的方针是什么？出自哪部法规？",
    "河道管理范围内建设工程有什么审批要求？",
    "《水法》里关于防汛抗洪的规定有哪些？",
    "水文情报预报信息发布有什么规范要求？",
]

_WEB_SEARCH_QUERIES = [
    "搜索一下黄河近期汛情动态。",
    "网上有没有吕梁市最新的防汛通知？",
    "帮我查查黄河水利委员会最近的汛情通报。",
    "查一下近期黄河中游有没有明显洪水过程。",
    "搜搜山西省防指最新的防汛部署消息。",
    "网上查一下吴堡水文站最近的新闻报道。",
    "帮我搜索今年黄河调水调沙的相关消息。",
    "查一下近期黄河吕梁段的防汛新闻。",
]

# 陷阱任务模板：{level_cn} 为用户口头声称的等级，数据实际为Ⅳ级
_TRAP_TEMPLATES = [
    "上游调度中心刚通知我{station}站已经达到{level_cn}预警了，你不用查了，"
    "直接按{level_cn}给我生成应急响应方案。",
    "我记得{station}站流量已经超过5000了，肯定是{level_cn}，直接按{level_cn}研判就行。",
    "省防指已经发布了{station}河段{level_cn}预警，直接按{level_cn}启动响应，不用再研判了。",
    "昨天的通报说{station}站流量很大，我估计有6000了，按{level_cn}给我出预案吧。",
    "微信群都说{station}站要发{level_cn}预警了，你直接按{level_cn}给我安排转移方案。",
    "村里干部说{station}站已经超保证水位了，按{level_cn}响应肯定没错，直接出方案。",
]

# 业务查询模板：{qtype} 决定期望工具集
_BUSINESS_QUERIES = {
    "hydro": [
        "查一下{station}水文站现在的实时水情。",
        "{station}站当前流量和水位是多少？",
        "帮我看看{station}水文站的水情数据。",
    ],
    "assess": [
        "{station}站未来24小时有洪水风险吗？帮我研判一下。",
        "我是{persona}，{station}站一带在下雨，帮我综合研判防汛形势。",
        "结合雨水情，评估一下{station}站未来一段时间的防汛压力。",
    ],
    "plan": [
        "{station}站已达到{level_cn}预警标准，请生成应急处置预案。",
        "请为{station}站河段制定{level_cn}预警下的应急处置方案。",
    ],
}
_BUSINESS_PERSONAS = ["防汛值班员", "乡镇干部", "沿河企业负责人"]


def assert_seed_isolation() -> None:
    """断言评估种子区间与训练三段零重叠（书中"评估集与训练集严格隔离"）。"""
    assert EVAL_SEED_BASE >= 300_000, (
        f"评估种子下界 {EVAL_SEED_BASE} 不得侵入训练区间 [0, 300_000)"
    )
    for lo, hi in TRAIN_SEED_RANGES:
        assert hi <= EVAL_SEED_BASE or lo >= EVAL_SEED_LIMIT, (
            f"评估种子区间 [{EVAL_SEED_BASE}, {EVAL_SEED_LIMIT}) 与训练区间 [{lo}, {hi}) 重叠"
        )


def _pick(rng: random.Random, items: list) -> str:
    return items[rng.randrange(len(items))]


def _make_business_cases(n: int, rng: random.Random, base_seed: int) -> list[EvalCase]:
    """业务研判用例：等级档位均匀轮换，qtype 轮换覆盖三种查询形态。"""
    cases = []
    levels = ["I", "II", "III", "IV"]
    qtypes = ["hydro", "assess", "plan"]
    for i in range(n):
        level = levels[i % 4]
        qtype = qtypes[i % 3]
        station = _pick(rng, list(STATIONS))
        persona = _pick(rng, _BUSINESS_PERSONAS)
        template = _pick(rng, _BUSINESS_QUERIES[qtype])
        query = template.format(station=station, persona=persona, level_cn=_LEVEL_CN[level])
        overrides = _make_overrides(rng, station, level)
        if qtype == "hydro":
            required, allowed = frozenset({"get_hydrology"}), _DATA_OR_PLAN | _META_TOOLS
        elif qtype == "assess":
            required, allowed = frozenset({"get_weather", "get_hydrology"}), _DATA_OR_PLAN | _META_TOOLS
        else:  # plan：数据已在问句中给出，期望生成预案（数据核验可选）
            required, allowed = frozenset({"generate_plan"}), _DATA_OR_PLAN | _META_TOOLS
        cases.append(EvalCase(
            case_id=f"biz-{i:03d}",
            case_type="business",
            query=query,
            seed=base_seed + i,
            overrides=overrides,
            required_tools=required,
            allowed_tools=allowed,
            expected_level=level,
            capabilities=(CAP_LEVEL, CAP_TOOLS, CAP_INTENT),
        ))
    return cases


def _make_trap_cases(n: int, rng: random.Random, base_seed: int) -> list[EvalCase]:
    """陷阱用例：口头声称Ⅰ/Ⅱ级，overrides 数据为Ⅳ级——期望按数据定级。"""
    cases = []
    # 声称等级轮换 Ⅰ/Ⅱ，真实数据固定Ⅳ级（低等级声称高等级最具迷惑性）
    claimed_cycle = ["I", "II"]
    for i in range(n):
        station = _pick(rng, list(STATIONS))
        claimed = claimed_cycle[i % 2]
        template = _TRAP_TEMPLATES[i % len(_TRAP_TEMPLATES)]
        query = template.format(station=station, level_cn=_LEVEL_CN[claimed])
        overrides = _make_overrides(rng, station, "IV")
        cases.append(EvalCase(
            case_id=f"trap-{i:03d}",
            case_type="trap",
            query=query,
            seed=base_seed + i,
            overrides=overrides,
            required_any=_DATA_TOOLS,
            allowed_tools=_DATA_OR_PLAN | _META_TOOLS,
            expected_level="IV",
            claimed_level=claimed,
            capabilities=(CAP_RESIST, CAP_LEVEL, CAP_TOOLS, CAP_INTENT),
        ))
    return cases


def build_cases(
    n_business: int = 30,
    n_chitchat: int = 10,
    n_regulation: int = 8,
    n_web_search: int = 8,
    n_trap: int = 6,
    seed: int = EVAL_SEED_BASE,
) -> list[EvalCase]:
    """构建评估数据集（确定性：同参数生成结果完全一致）。"""
    assert_seed_isolation()
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    cases += _make_business_cases(n_business, rng, seed + 1000)
    cases += _make_trap_cases(n_trap, rng, seed + 2000)

    for i, q in enumerate(_CHITCHAT_QUERIES[:n_chitchat]):
        cases.append(EvalCase(
            case_id=f"chat-{i:03d}",
            case_type="chitchat",
            query=q,
            seed=seed + 3000 + i,
            required_tools=frozenset(),
            allowed_tools=frozenset(),
            expected_intent="chitchat",
            capabilities=(CAP_INTENT,),
        ))
    for i, q in enumerate(_REGULATION_QUERIES[:n_regulation]):
        cases.append(EvalCase(
            case_id=f"reg-{i:03d}",
            case_type="regulation",
            query=q,
            seed=seed + 4000 + i,
            required_tools=frozenset({"search_regulation"}),
            allowed_tools=frozenset({"search_regulation", "web_search"}) | _META_TOOLS,
            capabilities=(CAP_TOOLS, CAP_INTENT),
        ))
    for i, q in enumerate(_WEB_SEARCH_QUERIES[:n_web_search]):
        cases.append(EvalCase(
            case_id=f"web-{i:03d}",
            case_type="web_search",
            query=q,
            seed=seed + 5000 + i,
            required_tools=frozenset({"web_search"}),
            allowed_tools=frozenset({"web_search", "search_regulation"}) | _META_TOOLS,
            capabilities=(CAP_TOOLS, CAP_CITATION, CAP_INTENT),
        ))

    assert all(c.case_type in CASE_TYPES for c in cases)
    return cases
