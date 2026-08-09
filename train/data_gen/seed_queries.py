"""人工编写的种子查询（Self-Instruct 起点）。

业务种子覆盖 3 站点 × 4 等级 × 3 意图（single_tool/multi_tool/plan_only），
知识种子覆盖防汛概念/历史/法规/应急常识，供 query_expander 扩张为大规模多样查询。

intent 字段仅用于场景参数化（决定生成哪些 mock 工具数据），
不作为训练标签，不会出现在最终训练数据中。
"""
from dataclasses import dataclass


@dataclass
class SeedQuery:
    query: str
    station: str
    level: str  # I / II / III / IV；知识问答为 ""
    intent: str  # single_tool / multi_tool / plan_only / knowledge


SEEDS: list[SeedQuery] = [
    # === single_tool：单工具查询（10个）===
    SeedQuery("吴堡站现在水情怎么样？", "吴堡", "IV", "single_tool"),
    SeedQuery("查一下龙门水文站的实时流量和水位。", "龙门", "IV", "single_tool"),
    SeedQuery("府谷站当前流量多少？", "府谷", "IV", "single_tool"),
    SeedQuery("吴堡站水位有没有超警戒？", "吴堡", "II", "single_tool"),
    SeedQuery("龙门站现在的流量数据是多少？", "龙门", "III", "single_tool"),
    SeedQuery("府谷水文站实时水情如何？", "府谷", "IV", "single_tool"),
    SeedQuery("吴堡站今天的水位是多少？", "吴堡", "III", "single_tool"),
    SeedQuery("龙门站流量超过5000了吗？", "龙门", "I", "single_tool"),
    SeedQuery("府谷站现在水情平稳吗？", "府谷", "III", "single_tool"),
    SeedQuery("查一下吴堡水文站最新水位。", "吴堡", "II", "single_tool"),

    # === multi_tool：多工具综合研判（12个）===
    SeedQuery("吴堡站未来24小时有洪水风险吗？需要预警吗？", "吴堡", "II", "multi_tool"),
    SeedQuery("我是防汛值班员，龙门站一带在下雨，帮我研判一下防汛形势。", "龙门", "III", "multi_tool"),
    SeedQuery("府谷站降雨情况怎么样？会不会涨水？", "府谷", "III", "multi_tool"),
    SeedQuery("吴堡站现在水情和天气如何？需要启动响应吗？", "吴堡", "I", "multi_tool"),
    SeedQuery("龙门水文站流量持续上涨，未来趋势如何？", "龙门", "II", "multi_tool"),
    SeedQuery("我是乡镇干部，府谷站那边暴雨不断，风险大不大？", "府谷", "I", "multi_tool"),
    SeedQuery("吴堡站当前水情+降雨，综合研判一下。", "吴堡", "III", "multi_tool"),
    SeedQuery("龙门站达到警戒水位了吗？结合降雨分析一下。", "龙门", "II", "multi_tool"),
    SeedQuery("府谷站未来洪峰流量预测多少？", "府谷", "II", "multi_tool"),
    SeedQuery("吴堡站水情严峻，流量快到5000了，怎么办？", "吴堡", "I", "multi_tool"),
    SeedQuery("我是沿河企业负责人，龙门站那边要不要撤离？", "龙门", "I", "multi_tool"),
    SeedQuery("府谷站当前流量正常吗？未来24小时有风险吗？", "府谷", "IV", "multi_tool"),

    # === plan_only：预案生成（8个）===
    SeedQuery("吴堡站已达Ⅰ级预警，请生成防汛值班员的应急处置预案。", "吴堡", "I", "plan_only"),
    SeedQuery("龙门站Ⅱ级预警，需要乡镇干部做什么？", "龙门", "II", "plan_only"),
    SeedQuery("府谷站Ⅲ级响应，沿河企业负责人该怎么应对？", "府谷", "III", "plan_only"),
    SeedQuery("吴堡站Ⅳ级蓝色预警，日常防汛措施有哪些？", "吴堡", "IV", "plan_only"),
    SeedQuery("龙门站达到Ⅰ级红色预警，生成应急响应方案。", "龙门", "I", "plan_only"),
    SeedQuery("府谷站Ⅱ级橙色预警，防汛物资怎么调配？", "府谷", "II", "plan_only"),
    SeedQuery("吴堡站Ⅲ级黄色预警，堤防巡查要求是什么？", "吴堡", "III", "plan_only"),
    SeedQuery("龙门站Ⅳ级预警期间，监测频次怎么安排？", "龙门", "IV", "plan_only"),
]


# === 防汛知识问答种子（不调用工具，纯知识性问答）===
KNOWLEDGE_SEEDS: list[SeedQuery] = [
    # 防汛概念
    SeedQuery("什么是防汛？", "", "", "knowledge"),
    SeedQuery("防洪和防汛有什么区别？", "", "", "knowledge"),
    # 等级标准
    SeedQuery("防汛预警等级是怎么划分的？", "", "", "knowledge"),
    SeedQuery("黄河防汛Ⅳ级响应有什么要求？", "", "", "knowledge"),
    SeedQuery("Ⅰ级红色预警意味着什么？", "", "", "knowledge"),
    # 历史事件
    SeedQuery("黄河历史上有哪些大洪水？", "", "", "knowledge"),
    SeedQuery("2021年黄河秋汛是怎么回事？", "", "", "knowledge"),
    # 法规常识
    SeedQuery("《防洪法》主要规定了什么？", "", "", "knowledge"),
    SeedQuery("《黄河防汛条例》的核心内容是什么？", "", "", "knowledge"),
    # 站点常识
    SeedQuery("吴堡水文站的作用是什么？", "吴堡", "", "knowledge"),
    SeedQuery("龙门水文站在黄河防汛中有什么地位？", "龙门", "", "knowledge"),
    # 应急知识
    SeedQuery("洪水来了普通群众应该怎么办？", "", "", "knowledge"),
    SeedQuery("防汛物资一般包括哪些？", "", "", "knowledge"),
    SeedQuery("什么是蓄滞洪区？", "", "", "knowledge"),
    SeedQuery("堤防巡查主要查什么？", "", "", "knowledge"),
]


def get_seeds() -> list[SeedQuery]:
    """返回全部业务种子查询（不含知识问答）。"""
    return list(SEEDS)


def get_knowledge_seeds() -> list[SeedQuery]:
    """返回全部知识问答种子。"""
    return list(KNOWLEDGE_SEEDS)
