"""防汛领域 Skill 种子数据注册脚本。

把 Agent 的 4 项核心业务能力注册为真正的 Skill（backend/data/skills.json）：
1. realtime_hydrology_query   实时水情查询
2. rainfall_flood_forecast    降雨洪水预判
3. warning_level_interpretation 预警级别解读
4. emergency_response_advice  应急响应建议

设计依据（借鉴 Claude Skills 渐进式披露）：
- description 双用途：注入 system prompt 供枚举 + embedding 语义匹配（阈值 0.55），
  因此描述中包含用户典型问法，提高匹配命中率
- instructions 在匹配成功时按需注入 planner / synthesizer（agent/graph/nodes.py）
- tool_names 绑定技能所需工具子集，须在 Skill.BUILTIN_TOOLS 白名单内

幂等：重复执行时跳过已存在的同名 Skill（--overwrite 可覆盖）。

用法（项目根目录）：
    python backend/scripts/seed_skills.py [--overwrite]
"""
import sys
from pathlib import Path

# 双根结构：agent/ 在项目根，app/ 在 backend/ 下
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

# 加载 .env（MySQL 配置等）
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / "backend" / ".env")

from agent.skills import create_skill, list_skills  # noqa: E402
from agent.skills.matcher import invalidate_cache  # noqa: E402
from agent.skills.models import SkillCreate  # noqa: E402
from agent.skills.store import delete_skill  # noqa: E402

SKILLS = [
    SkillCreate(
        name="realtime_hydrology_query",
        description=(
            "实时水情查询：查询黄河吕梁段水文站（吴堡、龙门、府谷等）的实时水位、流量、"
            "距警戒水位差值和超警情况。当用户询问「吴堡站现在水情怎么样」「当前水位多少」"
            "「流量有多大」「现在水位超过警戒线了吗」「水文站实时数据」等问题时启用。"
        ),
        instructions=(
            "你是黄河吕梁段实时水情查询专家。工作流程：\n"
            "1. 从用户问题中识别水文站名称（吴堡、龙门、府谷等）；未指定站点时，默认查询吴堡站并在回答中说明\n"
            "2. 调用 get_hydrology 获取实时数据（metric 默认 both，水位+流量）\n"
            "3. 解读返回字段：water_level_m=当前水位，flow_m3_s=当前流量，"
            "warning_level_m=警戒水位，above_warning_m=距警戒水位差值（负数=低于警戒，未超警）\n"
            "4. 回答必须包含：当前水位、当前流量、距警戒水位差值、是否超警，以及数据来源和时间\n"
            "约束：\n"
            "- 严禁编造任何水情数值；工具未返回数据时明确告知「未能获取实时数据」，"
            "并建议用户稍后重试或查阅官方水情渠道\n"
            "- 数值保留工具返回的原始精度，不要四舍五入夸大超警程度"
        ),
        tool_names=["get_hydrology"],
    ),
    SkillCreate(
        name="rainfall_flood_forecast",
        description=(
            "降雨洪水预判：结合未来降雨预报与径流预测模型，研判未来时段是否可能形成洪水、"
            "洪峰流量量级和水位上涨趋势。当用户询问「未来24小时降雨会不会引发洪水」「明天洪峰"
            "流量预计多少」「未来6小时流量会涨吗」「接下来12小时水位变化」「径流预报」等预测类问题时启用。"
        ),
        instructions=(
            "你是降雨洪水预判专家。工作流程：\n"
            "1. 调用 get_weather 获取目标区域未来 N 小时降雨预报（location 如「吕梁市」，hours 按用户问的预见期）\n"
            "2. 调用 predict_runoff 获取径流预测（累计降雨量和逐小时降雨序列由系统自动从天气结果注入）\n"
            "3. 将预测洪峰流量与系统预警标准对比，判断是否可能成灾及可能达到的等级\n"
            "4. 回答结构：降雨预报要点 → 径流预测结果（洪峰流量、出现时间）→ 是否可能引发洪水 → 防御建议\n"
            "约束：\n"
            "- 明确说明这是模型预测结果，存在不确定性，实际汛情以实时水情为准\n"
            "- 若天气预报或径流预测任一环节未返回数据，如实说明缺失环节，不要用猜测补齐\n"
            "- 预见期超过 72 小时时提示预测置信度下降"
        ),
        tool_names=["get_weather", "predict_runoff"],
    ),
    SkillCreate(
        name="warning_level_interpretation",
        description=(
            "预警级别解读：解读防汛预警等级（Ⅰ级红色、Ⅱ级橙色、Ⅲ级黄色、Ⅳ级蓝色）的划分标准，"
            "并基于实时水情和径流预测研判当前应处的预警等级。当用户询问「当前预警等级是多少」"
            "「是否需要发布预警」「应该启动几级响应」「预警等级怎么划分」「Ⅱ级预警意味着什么」等问题时启用。"
        ),
        instructions=(
            "你是防汛预警级别解读专家。预警等级从低到高：Ⅳ级（蓝）< Ⅲ级（黄）< Ⅱ级（橙）< Ⅰ级（红）。\n"
            "工作流程：\n"
            "1. 用户问当前等级或是否发布预警时：调用 get_hydrology 获取实时水情；"
            "涉及未来研判时再调用 predict_runoff 获取径流预测\n"
            "2. 将水位、流量、降雨数据与系统内置预警标准（WARNING_THRESHOLDS）对比定级：\n"
            "   - Ⅰ级（红）：流量 ≥ 5000m³/s，或水位超保证水位，或24h降雨 > 100mm\n"
            "   - Ⅱ级（橙）：流量 3000-5000m³/s，或水位超警戒水位，或24h降雨 50-100mm\n"
            "   - Ⅲ级（黄）：流量 2000-3000m³/s，或水位接近警戒水位\n"
            "   - Ⅳ级（蓝）：流量 < 2000m³/s，水位正常\n"
            "3. 回答结构：当前等级 → 定级依据（具体数值与阈值的对比）→ 该等级的含义和严峻程度\n"
            "约束：\n"
            "- 定级必须基于工具返回的真实数据；无数据时不得凭经验猜测等级\n"
            "- 多个指标指向不同等级时，按就高原则定级并说明理由\n"
            "- 等级标准以系统当前 WARNING_THRESHOLDS 配置为准，不要引用过时阈值"
        ),
        tool_names=["get_hydrology", "predict_runoff"],
    ),
    SkillCreate(
        name="emergency_response_advice",
        description=(
            "应急响应建议：根据预警等级和防汛法规条款，给出应急响应措施、人员转移、巡堤查险、"
            "物资调度等预案建议。当用户询问「Ⅱ级响应应该做什么」「发生洪水怎么响应」「应急预案是什么」"
            "「人员转移方案」「巡堤查险要求」「应急物资调度」「防汛条例怎么规定」等问题时启用。"
        ),
        instructions=(
            "你是防汛应急响应建议专家。工作流程：\n"
            "1. 调用 search_regulation 检索相关法规条款（如《黄河防汛条例》《防洪法》应急响应章节）\n"
            "2. 用户已明确预警等级时直接采用；未明确时先结合实时水情研判等级再给出对应措施\n"
            "3. 需要生成结构化预案时调用 generate_plan（需明确 warning_level、affected_area、population_at_risk）\n"
            "4. 回答结构：响应等级 → 法规依据（准确引用条款编号）→ 关键措施清单"
            "（组织会商、人员转移、巡堤查险、物资队伍调度）\n"
            "约束：\n"
            "- 措施必须以检索到的法规条款为依据，准确引用条款编号，严禁编造法规条文\n"
            "- 涉及人员转移、工程调度等重大事项，提示用户以当地防汛指挥机构的正式指令为准\n"
            "- 法规检索无结果时如实说明，给出一般性防御建议并建议咨询当地防指"
        ),
        tool_names=["search_regulation", "generate_plan"],
    ),
]


def seed(overwrite: bool = False) -> None:
    existing = {s.name for s in list_skills()}
    created, skipped = [], []
    for skill in SKILLS:
        if skill.name in existing:
            if not overwrite:
                skipped.append(skill.name)
                continue
            delete_skill(skill.name)
        create_skill(skill)
        created.append(skill.name)
    invalidate_cache()

    print(f"注册成功: {created}")
    if skipped:
        print(f"已存在跳过（--overwrite 可覆盖）: {skipped}")
    print(f"当前全部 Skill: {[s.name for s in list_skills()]}")


if __name__ == "__main__":
    seed(overwrite="--overwrite" in sys.argv)
