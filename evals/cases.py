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

CASE_TYPES = (
    "business", "chitchat", "regulation", "web_search", "trap",
    "memory", "compression", "tool_edge",
)

# 能力标签体系（报告按此出"任务 × 能力"矩阵）
CAP_LEVEL = "level_decision"        # 等级判定
CAP_TOOLS = "tool_selection"        # 工具选择
CAP_CITATION = "citation"           # 引用溯源
CAP_INTENT = "intent"               # 意图识别
CAP_RESIST = "misdirection_resistance"  # 抗误导（陷阱任务专属）
CAP_MEMORY = "memory_recall"        # 记忆召回（跨会话事实/知识更新/时间推理）
CAP_NEEDLE = "needle_retention"     # 长上下文针保留（压缩后早轮信息不丢）

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
        expected_intent: 期望意图（None=不检查：tool_edge 边界查询两类意图皆可）
        history: 预置会话历史（memory/compression 用例：多轮脚本，压缩用例须超预算）
        memory_payload: 脚本化记忆注入内容（key ∈ longterm/experiences/semantic，
            模拟已积累的记忆库；experiments/memory.py 的 patch 消费）
        needle_substrings: 答案必须包含的子串（针检查：确定性，不依赖 judge）
        forbidden_substrings: 答案不得包含的子串（知识更新用例：旧值不得回显）
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
    expected_intent: str | None = "agent_task"
    claimed_level: str | None = None
    history: list = field(default_factory=list)
    memory_payload: dict = field(default_factory=dict)
    needle_substrings: tuple = ()
    forbidden_substrings: tuple = ()
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


# ====== 记忆召回用例（LongMemEval 式：脚本化记忆 + 事后提问，experiments/memory.py 消费） ======

# 站点档案事实（fact 子类）：注入长期记忆，问答案必须含精确数值
_MEMORY_FACTS = {
    "吴堡": ("警戒水位", "638.26"),
    "龙门": ("警戒流量", "10500"),
    "府谷": ("保证水位", "812.45"),
}
_MEMORY_FACT_QUERY_TPL = [
    "结合你记住的站点档案回答：{station}站的{item}是多少？不用查工具。",
    "{station}站的{item}是多少？请基于你记住的历史信息回答。",
    "你之前记过{station}站的{item}吧？说给我听听。",
    "不用查实时数据，凭记忆说下{station}站的{item}。",
]

# 知识更新（update 子类）：语义记忆为旧值，近期会话已更正——答案用新值、不得回显旧值
_MEMORY_UPDATE = {
    "吴堡": ("警戒流量", "6200", "6500"),
    "龙门": ("汛限水位", "382.10", "385.50"),
    "府谷": ("警戒流量", "5500", "5800"),
}
_MEMORY_UPDATE_QUERY_TPL = [
    "结合我们刚才的更正，{station}站的{item}现在是多少？",
    "{station}站的{item}以哪个数为准？旧值还是更正值？",
]

# 时间推理（temporal 子类）：情景记忆中的时间事件，问答需还原事件结论
_MEMORY_TEMPORAL = [
    ("丁家沟雨量站 9 月 2 日完成校准，此前雨量数据系统性偏低约 5%。",
     "丁家沟站 9 月 1 日之前的雨量数据还能直接用吗？", ("偏低",)),
    ("上游 8 月 28 日调度会决议：红旗沟水库即日起按 120 立方米每秒预泄腾库。",
     "红旗沟水库现在应该按多大流量预泄？", ("120",)),
    ("前日报汛电话修正：裴沟站 3 日 8 时流量由 2100 修正为 2450 立方米每秒。",
     "裴沟站 3 日 8 时的流量最终以哪个数为准？", ("2450",)),
    ("白乙沟水文站 9 月起迁址至下游 300 米新断面，新旧断面水位不作换算。",
     "白乙沟站的水位数据和迁址前怎么衔接？", ("不作换算",)),
    ("8 月 30 日测流缆道检修，当晚白乙沟流量缺测，已用插补值代替。",
     "8 月 30 日晚白乙沟的流量数据是实测的吗？", ("插补",)),
    ("9 月 1 日起全河段报汛频次由每小时 1 次加密为每小时 2 次。",
     "现在的报汛频次是多少？和 9 月前比有什么变化？", ("加密",)),
]


def _make_memory_cases(n: int, rng: random.Random, base_seed: int) -> list[EvalCase]:
    """记忆召回用例：fact/update/temporal 三子类轮换。

    记忆内容通过 memory_payload 脚本化注入（experiments/memory.py 把三个
    注入函数替换为 payload 读取）——受控内容优于真实记忆库的噪声，
    归因到"Harness+模型能否用好记忆内容"，库填充质量由单测另行覆盖。
    """
    stations = list(STATIONS)
    cases: list[EvalCase] = []
    # 子类配比 fact:update:temporal = 2:1:1
    for i in range(n):
        station = stations[i % len(stations)]
        subtype = ("fact", "fact", "update", "temporal")[i % 4]
        history: list = []
        if subtype == "fact":
            item, value = _MEMORY_FACTS[station]
            query = _MEMORY_FACT_QUERY_TPL[(i // 4) % len(_MEMORY_FACT_QUERY_TPL)] \
                .format(station=station, item=item)
            payload = {"longterm": (
                f"【长期记忆·站点档案】{station}站{item} {value}"
                f"（2024 年汛期核定，值班交接时登记）。"
            )}
            needles, forbidden = (value,), ()
        elif subtype == "update":
            item, old, new = _MEMORY_UPDATE[station]
            query = _MEMORY_UPDATE_QUERY_TPL[(i // 4) % len(_MEMORY_UPDATE_QUERY_TPL)] \
                .format(station=station, item=item)
            payload = {"semantic": (
                f"【语义记忆】{station}站{item} {old}（2024 年核定，可能过期）。"
            )}
            history = [
                {"role": "user", "content": (
                    f"更正：经 2025 年复核，{station}站{item}由 {old} 调整为 {new}，以此为准。"
                )},
                {"role": "assistant", "content": (
                    f"已了解，{station}站{item}以更正值 {new} 为准。"
                )},
            ]
            needles, forbidden = (new,), (old,)
        else:
            fact, query, needles = _MEMORY_TEMPORAL[(i // 4) % len(_MEMORY_TEMPORAL)]
            payload = {"experiences": f"【情景记忆】{fact}"}
            forbidden = ()
        cases.append(EvalCase(
            case_id=f"mem-{i:03d}",
            case_type="memory",
            query=query,
            seed=base_seed + i,
            history=history,
            memory_payload=payload,
            needle_substrings=needles,
            forbidden_substrings=forbidden,
            expected_intent=None,  # 记忆问答两类意图皆可，判定只看针
            capabilities=(CAP_MEMORY,),
        ))
    return cases


# ====== 压缩等价性用例（LLMLingua 式：长历史埋针 + 末轮提问，experiments/compression.py 消费） ======

_COMPRESSION_NEEDLES = {
    "吴堡": "638.26",
    "龙门": "385.90",
    "府谷": "812.45",
}
_COMPRESSION_QUERY_TPL = [
    "回顾我们前面聊过的内容：{station}站的警戒水位是多少？前期流域累积降雨量达到多少毫米？",
    "只根据我们之前的对话回答：{station}站警戒水位和入汛以来流域累积降雨量分别是多少？",
    "刚才对话里提到过{station}站的警戒水位和累积降雨量，分别是多少来着？",
    "不用查工具，凭我们前面的对话说：{station}站警戒水位多少？流域前期累积雨量多少？",
]

# 填充轮语料池（与针无关的值班问答，撑长历史至 token 预算之上）
_FILLER_TOPICS = [
    ("今天的值班安排有什么要注意的？",
     "今日值班重点关注三项：一是{station}站流量变幅，若小时涨幅超过 5% 需立即上报；"
     "二是上游区间降雨预报的落区调整，气象台 08 时会商已把强降雨落区向南修正约 30 公里，"
     "涉及我们河段的量级可能下调；三是撤离路线的临时管制信息，交警队在沿河三个路口"
     "设置了管制点，如启动响应需要协调放行抢险车辆。另请注意今日 14 时有上级汛情调度会，"
     "需要提前准备昨日水情简报和近三日雨水情对比材料，会上可能问到区间来水占比问题。"),
    ("最近几天的水情总体怎么样？",
     "近三日水情总体平稳中略有抬升。{station}站流量从 {q1} 立方米每秒缓涨至 {q2} 立方米每秒，"
     "水位相应抬升约 0.3 米，均在警戒值以下。支流来水占比约四成，较前期略有增加，"
     "主要是上游局地阵性降雨贡献。河道演进方面，洪水传播时间与往常一致，"
     "未出现漫滩或偎水情况。未来 24 小时如无新的强降雨，预计维持缓变态势；"
     "若上游出现 50 毫米以上量级降雨，需要关注断面起涨时间并做好加密测报准备。"),
    ("上游水库现在是什么状态？",
     "上游万家寨水库当前水位接近汛限，出库按调度规程控制在 {q1} 立方米每秒附近，"
     "入库流量与出库基本持平，库容处于安全区间。按现行调度方式，未来三日若无台风外围影响，"
     "水库将以发电流量为基础平稳运行。需要注意的是水库泄流时段的传播到本河段时间约 20 小时，"
     "如遇调度调整需要提前一天通知沿河做好防范。另外上游还有两座中型水库在拦洪运用，"
     "其泄流汇入后对本断面洪峰有一定削峰作用，具体数值模型组还在滚动复核。"),
    ("防汛物资准备得如何了？",
     "按照年度度汛方案，重点物资已完成了第二轮核查补充。编织袋、铅丝笼、救生衣等"
     "常规物资在三个中心仓库均有足额储备，抢险车辆和挖掘机等机械已落实社会化储备协议，"
     "联系人清单已更新。砂石料场落实了两处，均在 30 分钟运输半径内。存在的短板是"
     "夜间照明设备数量偏紧，已紧急采购一批移动灯塔，预计下周到货。各乡镇的应急"
     "队伍花名册已核实，共登记抢险队员六百余人，关键岗位实行 24 小时双人值守。"),
    ("天气展望怎么说？",
     "气象部门最新预报显示，未来三天本区域以多云天气为主，局地有分散性阵雨，"
     "累计雨量不大。中期预报看，周末前后可能有一次较明显的降雨过程，落区和量级"
     "还存在分歧，欧洲和中央台两家模式预报的累计雨量相差接近一倍，需要滚动关注。"
     "水汽条件方面，副高边缘的西南气流维持，低层湿度较好，一旦有扰动触发，"
     "局地短时强降雨的可能性不能排除。建议重点关注周末过程与上游来水的叠加影响，"
     "提前做好会商和测报加密的预案。"),
]


def _make_long_history(rng: random.Random, station: str, warning_level_value: str) -> list:
    """构造超 token 预算的长会话历史（针埋在早轮，撑长用值班问答填充）。"""
    q_low, q_high = 800 + rng.randrange(400), 1400 + rng.randrange(600)
    history: list[dict] = [
        {"role": "user", "content": (
            f"先记个底：{station}站的警戒水位是 {warning_level_value} 米，今天先聊聊总体形势。"
        )},
        {"role": "assistant", "content": (
            f"好的，已记录：{station}站警戒水位 {warning_level_value} 米。"
            f"下面结合当前雨水情给你一个总体判断：目前河道流量处于平稳段，"
            f"水位距警戒值还有一定余量，但前期土壤含水量偏高，一旦出现强降雨，"
            f"产流会明显加快，需要提前关注。"
        )},
        {"role": "user", "content": (
            "另外，今年入汛以来前期流域累积降雨量已达 127.4 毫米，比常年同期偏多。"
        )},
        {"role": "assistant", "content": (
            "了解，前期流域累积降雨量 127.4 毫米、较常年偏多这个信息很关键——"
            "土壤偏饱和意味着后续降雨的径流系数会抬高，同样的雨可能形成更大的洪峰，"
            "研判时我会把这个作为重要背景考虑。"
        )},
    ]
    # 填充轮：5 个主题循环 4 遍 = 20 轮，每轮附一段时段快报（数值 seed 确定性），
    # 总量需超 HISTORY_MAX_TOKENS（coverage.py 断言守门，不足会被 CI 拦下）
    for i in range(20):
        topic_q, topic_a = _FILLER_TOPICS[i % len(_FILLER_TOPICS)]
        hour = (8 + i) % 24
        bulletin = (
            f"附 {hour:02d} 时段快报：{station}站流量 {q_low + i * 17} 立方米每秒、"
            f"水位 {310 + (i % 9) * 0.1:.1f} 米、含沙量 {2 + (i % 7)} 千克每立方米；"
            f"区间面雨量 {(i % 5) * 0.6:.1f} 毫米；水温 {14 + (i % 6)} 度。"
            f"上游来水趋稳，下游传播正常，闸门开度无调整，机组运行平稳。"
            f"巡测人员在岗，报汛信道畅通，备用电源切换试验正常。"
            f"本段数据已同步至值班日志与水情数据库，供后续会商引用。"
        )
        history.append({"role": "user", "content": topic_q})
        history.append({"role": "assistant", "content": topic_a.format(
            station=station,
            q1=q_low + i * 10,
            q2=q_high + i * 15,
        ) + bulletin})
    return history


def _make_compression_cases(n: int, rng: random.Random, base_seed: int) -> list[EvalCase]:
    """压缩等价性用例：历史超 4000 token 预算、针埋早轮，末轮提问验针保留。"""
    stations = list(STATIONS)
    cases: list[EvalCase] = []
    for i in range(n):
        station = stations[i % len(stations)]
        level_value = _COMPRESSION_NEEDLES[station]
        query = _COMPRESSION_QUERY_TPL[(i // len(stations)) % len(_COMPRESSION_QUERY_TPL)] \
            .format(station=station)
        cases.append(EvalCase(
            case_id=f"comp-{i:03d}",
            case_type="compression",
            query=query,
            seed=base_seed + i,
            history=_make_long_history(rng, station, level_value),
            needle_substrings=(level_value, "127.4"),
            expected_intent=None,  # 凭历史作答不强制意图与工具
            capabilities=(CAP_NEEDLE,),
        ))
    return cases


# ====== 工具边界用例（BFCL 式：无关/幻觉/元工具/并行/串行，experiments 回归门禁消费） ======

# 纯超范围：与防汛值守无关，期望零工具调用（含元工具）
_EDGE_OUT_OF_SCOPE_PURE = [
    "帮我订一张明天去西安的高铁票。",
    "把这段会议录音转成文字纪要。",
    "帮我预测一下下周股市走势。",
    "给值班邮箱发一封防汛物资清单邮件。",
]
# 能力相邻：本系统没有对应工具，期望不乱调数据工具（允许 list_skills 自查）
_EDGE_OUT_OF_SCOPE_ADJACENT = [
    "调取吴堡站的雷达回波图分析一下降雨趋势。",
    "用卫星云图看看吕梁上空的水汽情况。",
    "画一张吴堡站的水位过程线图并导出图片。",
    "调取黄河全流域的实时视频监控看看河道情况。",
]
_EDGE_META_QUERIES = [
    "你现在有哪些技能？分别能干什么？",
    "你支持哪些工具？列一下。",
    "展示一下你的能力清单。",
    "看看你都会什么，有哪些本事。",
]
_EDGE_MULTI_QUERIES = [
    "把{station}站的实时水情、当地天气和周边地形一起查全，做综合研判。",
    "{station}站的水情、气象和地形资料一次性都给我调出来。",
    "综合{station}站的雨水情和河道地形，把需要的数据都查齐了再分析。",
    "给{station}站做全面体检：水情、天气、地形全要。",
]
_EDGE_SEQUENTIAL_QUERIES = [
    "预报{station}站未来 6 小时的径流过程。",
    "{station}站未来半天的来水过程预报一下。",
    "推演{station}站短时径流变化趋势。",
    "预估{station}站接下来几小时的流量过程。",
]


def _make_tool_edge_cases(n: int, rng: random.Random, base_seed: int) -> list[EvalCase]:
    """工具边界用例：out_of_scope(8) + meta(4) + multi(4) + sequential(4)。

    BFCL 的 relevance detection / hallucination 类别的领域化：无可满足工具时
    应克制不调用（precision 严格），元工具合法可调，多工具并行与串行依赖
    覆盖工具编排能力。
    """
    cases: list[EvalCase] = []
    stations = list(STATIONS)
    levels = ["II", "III"]

    def _add(cid: str, query: str, seed_off: int, **kw) -> None:
        cases.append(EvalCase(
            case_id=cid,
            case_type="tool_edge",
            query=query,
            seed=base_seed + seed_off,
            capabilities=(CAP_TOOLS,),
            **kw,
        ))

    for i, q in enumerate(_EDGE_OUT_OF_SCOPE_PURE[: n]):
        _add(f"edge-oop-{i:03d}", q, i,
             required_tools=frozenset(), allowed_tools=frozenset(),
             expected_intent=None)
    for i, q in enumerate(_EDGE_OUT_OF_SCOPE_ADJACENT[: max(0, n - 4)]):
        _add(f"edge-adj-{i:03d}", q, 100 + i,
             required_tools=frozenset(), allowed_tools=_META_TOOLS,
             expected_intent=None)
    for i, q in enumerate(_EDGE_META_QUERIES[: max(0, n - 8)]):
        _add(f"edge-meta-{i:03d}", q, 200 + i,
             required_tools=frozenset({"list_skills"}), allowed_tools=None,
             expected_intent=None)
    for i, q in enumerate(_EDGE_MULTI_QUERIES[: max(0, n - 12)]):
        station = stations[i % len(stations)]
        _add(f"edge-multi-{i:03d}", q.format(station=station), 300 + i,
             overrides=_make_overrides(rng, station, levels[i % 2]),
             required_tools=frozenset({"get_hydrology", "get_weather", "query_gis_terrain"}),
             allowed_tools=_DATA_OR_PLAN | _META_TOOLS)
    for i, q in enumerate(_EDGE_SEQUENTIAL_QUERIES[: max(0, n - 16)]):
        station = stations[i % len(stations)]
        _add(f"edge-seq-{i:03d}", q.format(station=station), 400 + i,
             overrides=_make_overrides(rng, station, levels[i % 2]),
             required_tools=frozenset({"predict_runoff"}),
             allowed_tools=_DATA_OR_PLAN | _META_TOOLS)
    return cases


def build_cases(
    n_business: int = 30,
    n_chitchat: int = 10,
    n_regulation: int = 8,
    n_web_search: int = 8,
    n_trap: int = 6,
    n_memory: int = 0,
    n_compression: int = 0,
    n_tool_edge: int = 0,
    seed: int = EVAL_SEED_BASE,
) -> list[EvalCase]:
    """构建评估数据集（确定性：同参数生成结果完全一致）。

    新三类默认 0 条：默认组合保持 62 条不变，确保与既有基线可比
    （regression.py 的组合一致性检查），由 --experiment / 显式参数启用。
    """
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

    cases += _make_memory_cases(n_memory, rng, seed + 6000)
    cases += _make_compression_cases(n_compression, rng, seed + 7000)
    cases += _make_tool_edge_cases(n_tool_edge, rng, seed + 8000)

    assert all(c.case_type in CASE_TYPES for c in cases)
    return cases
