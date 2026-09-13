"""阈值校准专用：构造测试记忆向量（不碰 MySQL，只写 Qdrant，可清理）。

动机：线上记忆库仅个位数向量时，分布校准没有统计意义。本脚本批量构造
贴近真实业务的记忆文本（语义=领域知识 / 情景=历史事件 / 程序=任务步骤），
embed 后以 payload kind=calibration_seed 写入三个 collection，id 从 900000 起
（远超 MySQL 自增区间，不与真实数据冲突）。

用法（项目根目录，需 Qdrant + embedding 服务可用）：
    python backend/scripts/seed_calibration_memories.py            # 播种
    python backend/scripts/seed_calibration_memories.py --clean    # 清除播种数据
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / "backend" / ".env")

from agent.memory import vector_index  # noqa: E402
from app.core.llm import get_qdrant_client  # noqa: E402

_SEED_KIND = "calibration_seed"
_ID_BASE = 900_000

# 语义记忆：领域知识（对齐 agent_semantic 的 title+content 拼接格式）
_SEMANTIC = [
    ("预警等级划分", "黄河防汛预警分四级：Ⅰ级红色特别重大、Ⅱ级橙色重大、Ⅲ级黄色较大、Ⅳ级蓝色一般，阈值以系统 WARNING_THRESHOLDS 为准"),
    ("警戒水位与保证水位", "警戒水位是防汛部门关注的特征水位，保证水位是堤防能安全抗御的上限水位，超保证水位须立即组织抢险"),
    ("洪峰流量", "洪峰流量是一次洪水过程中的最大瞬时流量，反映洪水规模的核心指标，单位立方米每秒"),
    ("24小时降雨等级", "24小时降雨量超过100毫米达大暴雨量级，超过250毫米为特大暴雨，是山洪灾害的主要诱发因素"),
    ("SCS-CN 产流模型", "SCS-CN 模型用曲线数 CN 值估算降雨产流量，适用于缺乏实测径流资料的小流域"),
    ("吴堡水文站", "吴堡站是黄河中游干流重要控制站，位于陕西省吴堡县，控制黄河河口镇至龙门区间的上半段来水"),
    ("龙门水文站", "龙门站是黄河干流控制性水文站，位于晋陕峡谷出口，是黄河中游洪水预报的关键节点"),
    ("蓄滞洪区", "蓄滞洪区是分蓄河道超额洪水的区域性工程措施，启用须经防汛指挥机构批准并提前转移区内群众"),
    ("防汛应急响应", "Ⅰ级响应要求防指进入战时状态、组织受威胁群众转移；Ⅱ级要求全员到岗、危险区域群众限时转移"),
    ("堤防巡查要点", "堤防巡查重点检查散浸、管涌、裂缝、滑坡、渗漏等险情，高水位期间加密巡查频次"),
    ("防汛物资", "防汛抢险物资主要包括编织袋、土工布、块石、砂料、冲锋舟、救生衣、发电机组等"),
    ("山洪灾害特点", "山洪灾害突发性强、水量集中、流速大、冲刷破坏力强，预警窗口短，需提前转移避险"),
    ("黄河凌汛", "凌汛发生在封冻期和开河期，冰塞冰坝壅高水位，宁蒙河段和下游山东段是防凌重点"),
    ("秋汛特点", "秋汛由华西秋雨形成，历时长、总量大、峰值相对低，2021年黄河秋汛为典型"),
    ("洪水重现期", "重现期是某量级洪水平均多少年一遇的统计指标，百年一遇洪水指任一年发生概率为1%"),
    ("水位流量关系", "水位流量关系曲线描述测站水位与流量的对应关系，受河床冲淤影响需定期率定"),
    ("水库防洪调度", "水库防洪调度按批复的调度规程运用，汛限水位约束下的预泄、拦洪、错峰是核心手段"),
    ("洪水风险图", "洪水风险图标示不同频率洪水的淹没范围、水深和演进时间，是避险转移规划的基础"),
    ("预报精度评价", "洪水预报用确定性系数评价精度等级，甲级要求确定性系数大于0.9"),
    ("降水径流关系", "降水转化为径流受下垫面、前期土壤含水量和降雨强度影响，前期影响雨量是重要参数"),
    ("预警发布渠道", "预警信息通过广播、电视、短信、大喇叭和网格员逐户通知等渠道发布，确保覆盖到人"),
    ("分洪闸运用", "分洪闸在河道流量超过安全泄量时开闸分洪，降低下游河道水位，须按调度指令执行"),
    ("河道淤积影响", "河道淤积抬高同流量水位，降低行洪能力，同一流量下淤积后水位明显抬高"),
    ("汛期划分", "黄河汛期为6月至10月，主汛期7月至8月，秋汛9月至10月"),
    ("避险转移原则", "避险转移遵循先人员后财产、先危险区后安全区原则，转移路线避开低洼易涝路段"),
]

# 情景记忆：历史事件（对齐 agent_episodes 的 event_summary+resolution 拼接格式）
_EPISODES = [
    ("吴堡站超警戒水情研判", "吴堡站流量涨至4200立方米每秒超警戒，综合降雨和水位研判发布Ⅱ级预警，建议加强巡堤"),
    ("get_hydrology 超时降级", "实时水情接口连续超时，改用最近一次缓存数据并标注时效，用户接受降级方案"),
    ("用户纠正等级判定", "用户指出把Ⅲ级误报为Ⅱ级，复核发现是流量单位换算错误，已修正并沉淀教训"),
    ("龙门站洪峰过程", "龙门站出现洪峰流量5600立方米每秒，研判达Ⅰ级预警，生成了沿河企业撤离预案"),
    ("降雨数据缺失处理", "气象接口缺24小时降雨数据，仅凭流量和水位完成等级研判并注明数据缺口"),
    ("多轮复合查询", "用户先问吴堡水情再追问龙门对比，第二轮复用首轮工具结果完成对比回答"),
    ("预案生成任务", "府谷站达Ⅱ级预警，为乡镇干部生成应急处置预案，包含转移路线和物资清单"),
    ("GIS 地形叠加分析", "用户询问淹没范围，调用 GIS 地形工具叠加洪水水位完成影响范围分析"),
    ("法规条款检索", "用户询问蓄滞洪区启用权限，检索防洪法条款并给出准确引用"),
    ("流量单位混淆事件", "教师模型把立方米每秒和升每秒混淆导致等级误判，修正后加入了单位校验步骤"),
    ("夜间值班紧急研判", "凌晨吴堡站流量快速上涨，值班员要求紧急研判，两分钟内完成等级判定并推送预警"),
    ("洪峰预报偏差复盘", "洪峰预报偏小15%，复盘发现前期土壤含水量估计偏低，调整了参数取值"),
    ("多站连报处理", "用户一次询问三个站点水情，顺序调用工具汇总成对比表格回答"),
    ("预警升级事件", "降雨持续导致等级从Ⅲ级升到Ⅱ级，及时更新预警并通知相关责任人"),
    ("数据库连接失败", "MySQL 短暂不可用导致记忆检索失败，系统降级为无记忆回答，恢复后自动补检索"),
    ("重复提问识别", "用户重复询问相同站点水情，识别出重复并引用上次结果补充最新变化"),
    ("工具参数错误", "station 参数传错站点名导致返回空，捕获后用站名白名单校验重试成功"),
    ("秋汛长历时研判", "2021年秋汛类似情形，连续多日研判维持高等级预警，积累了长历时洪水经验"),
    ("图数不一致处理", "GIS 显示淹没范围与水文数据矛盾，以水文实测为准并在回答中说明差异原因"),
    ("夜间批量化查询", "值班员批量查询五个站点完成 nightly 报告，程序记忆复用了批量查询步骤"),
]

# 程序记忆：任务步骤（对齐 agent_procedures 的 applicability 格式）
_PROCEDURES = [
    ("单站实时水情查询流程", "适用于查询某水文站当前流量水位：先调 get_hydrology 获取实时数据，再按阈值判定预警等级，最后组织回答"),
    ("综合洪水研判流程", "适用于洪水风险综合研判：并行调用 get_weather 和 get_hydrology，再调 predict_runoff 推洪峰，最后综合三级数据定级"),
    ("应急预案生成流程", "适用于已达预警等级后的预案生成：先收集水情和降雨，再调 generate_plan 按等级模板生成对应角色的处置预案"),
    ("工具失败降级流程", "适用于工具连续失败：首次失败立即重试一次，再失败改用缓存数据并明确标注时效，不阻塞回答"),
    ("等级争议复核流程", "适用于用户对等级判定有异议：重新拉取最新工具数据，用规则引擎重算等级，与原判定对比后答复"),
    ("多站对比查询流程", "适用于多站水情对比：逐站调用 get_hydrology，汇总为对比表，突出最高等级站点"),
    ("淹没范围分析流程", "适用于淹没影响分析：先获取水位数据，再调 GIS 地形工具叠加分析，标注影响区和人口"),
    ("法规条款问答流程", "适用于防汛法规咨询：调 search_regulation 检索条款，回答须引用具体法规名称和条号"),
    ("批量巡检报告流程", "适用于夜间批量巡检：循环查询各站数据，超警站点标记突出，汇总成值守报告"),
    ("洪峰预报流程", "适用于洪峰预报：获取当前流量和降雨，调 predict_runoff 推算洪峰，与保证水位对比评估风险"),
    ("预警升级监测流程", "适用于水情持续发展场景：按时间间隔复查水情，等级变化立即重新研判并通知"),
    ("数据缺失研判流程", "适用于部分数据源缺失：评估可用判据是否足以定级，不足时明确说明并给出保守建议"),
]

_POOL = [
    (vector_index.SEMANTIC_COLLECTION,
     [f"{t}。{c}" for t, c in _SEMANTIC]),
    (vector_index.EPISODE_COLLECTION,
     [f"{s}。{r}" for s, r in _EPISODES]),
    (vector_index.PROCEDURE_COLLECTION,
     [a for a, _ in _PROCEDURES]),
]


def seed() -> int:
    from qdrant_client.http import models as qmodels

    from agent.rag.embedding import embed_texts

    client = get_qdrant_client()
    total = 0
    next_id = _ID_BASE
    for collection, texts in _POOL:
        if not vector_index._collection_ready(collection):
            print(f"  {collection}: Qdrant 不可用，跳过")
            continue
        vectors = embed_texts(texts)
        if vectors is None:
            print(f"  {collection}: embedding 失败，跳过")
            continue
        points = [
            qmodels.PointStruct(
                id=next_id + i, vector=v.tolist(),
                payload={"kind": _SEED_KIND, "text": text[:120]},
            )
            for i, (text, v) in enumerate(zip(texts, vectors, strict=False))
        ]
        client.upsert(collection_name=collection, points=points)
        next_id += len(points)
        total += len(points)
        print(f"  {collection}: +{len(points)} 条校准向量")
    return total


def clean() -> int:
    from qdrant_client.http import models as qmodels

    client = get_qdrant_client()
    removed = 0
    for collection, _ in _POOL:
        try:
            client.delete(
                collection_name=collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(must=[qmodels.FieldCondition(
                        key="kind", match=qmodels.MatchValue(value=_SEED_KIND),
                    )])
                ),
            )
            removed += 1
            print(f"  {collection}: 已清除校准向量")
        except Exception as e:
            print(f"  {collection}: 清除失败 {e}")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="校准用测试记忆向量播种/清除")
    parser.add_argument("--clean", action="store_true", help="清除本脚本播种的向量")
    args = parser.parse_args()
    if args.clean:
        clean()
    else:
        print(f"播种校准记忆（payload kind={_SEED_KIND}，id≥{_ID_BASE}）")
        seed()


if __name__ == "__main__":
    main()
