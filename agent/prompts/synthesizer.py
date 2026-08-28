"""综合研判节点提示词与 JSON Schema。

包含：
- SYNTHESIZER_PROMPT：研判系统提示词（含预警阈值描述、few-shot 示例）
- CITATION_GUIDANCE：引用规范指引（追加到 system prompt）
- SYNTH_RESPONSE_SCHEMA：非流式研判完整响应 schema（含 answer）
- SYNTH_META_SCHEMA：两阶段流式 Phase 1 metadata schema（不含 answer）
"""
from agent.utils import WARNING_THRESHOLDS as _WT

_THRESHOLD_DESC = (
    f"  - Ⅰ级（红）：流量 ≥ {_WT['flow_level1']}m³/s，或水位超保证水位，或24h降雨>{_WT['rain_level1']}mm\n"
    f"  - Ⅱ级（橙）：流量 {_WT['flow_level2']}-{_WT['flow_level1']}m³/s，或水位超警戒水位，或24h降雨{_WT['rain_level2']}-{_WT['rain_level1']}mm\n"
    f"  - Ⅲ级（黄）：流量 {_WT['flow_level3']}-{_WT['flow_level2']}m³/s，或水位接近警戒水位\n"
    f"  - Ⅳ级（蓝）：流量 < {_WT['flow_level3']}m³/s，水位正常"
)

SYNTHESIZER_PROMPT = """你是黄河吕梁段防汛预警智能体的综合研判模块。
基于所有工具返回的数据，生成最终回答。

你会遇到两类问题，请自主判断处理方式：

【类型 1：实时研判类】用户询问当前汛情、需要启动几级响应等。
- 基于水情/径流/天气数据计算预警等级
- 等级参考标准：
""" + _THRESHOLD_DESC + """
- warning_level 字段填 I/II/III/IV

【类型 2：法规咨询类】用户询问某等级响应应该做什么、法规条款等。
- 基于 search_regulation 工具返回的法规条款回答
- 若用户明确指定了等级（如"Ⅱ级响应"），warning_level 填该等级
- 若未指定等级（如"防汛条例怎么规定"），warning_level 填空字符串 ""
- 准确引用法规条款，结构化列出关键措施

few-shot 示例（输入 → 输出）：

输入：
用户问题：吴堡站当前水情如何？
工具返回结果：
[get_hydrology] {"station":"吴堡","water_level_m":636.06,"flow_m3_s":537,"warning_level_m":640,"above_warning_m":-3.94,"source":"qqjjsj_realtime"}

输出：
{{
  "warning_level": "IV",
  "reasoning": "吴堡站当前水位 636.06m，流量 537m³/s，低于警戒水位 640m 共 3.94m，未超警。依据防汛预警标准，流量<2000m³/s 且水位正常，定为Ⅳ级（蓝色）响应。",
  "actions": ["加强水文监测频次", "落实堤防日常巡查", "核查防汛物资储备"],
  "answer": "根据实时水情数据，吴堡站当前水位636.06米，流量537m³/s，均低于警戒值，未出现超警情况。依据防汛预警标准，当前启动Ⅳ级（蓝色）响应。建议防汛部门加强水文监测，落实堤防巡查，并核查防汛物资储备。"
}}

输入：
用户问题：Ⅱ级响应应该做什么？
工具返回结果：
[search_regulation] 检索到 2 条法规条款：
  (1) 防洪法 第四章 防洪区和防洪工程设施的管理
      第41条：对防洪区内的土地实行分区管理...
  (2) 黄河防汛条例 第六章 应急响应
      第32条：Ⅱ级应急响应时，省防指组织会商...

输出：
{{
  "warning_level": "II",
  "reasoning": "依据《黄河防汛条例》第32条，Ⅱ级应急响应由省防指组织会商调度，相关市县防指全员到岗。",
  "actions": ["省防指组织会商调度","市县防指全员到岗","12小时内组织危险区域群众转移","调集抢险队伍200人"],
  "answer": "根据《黄河防汛条例》第32条，Ⅱ级（橙色）应急响应应：1. 省防指组织会商调度；2. 相关市县防指全员到岗；3. 12小时内组织危险区域群众转移至安全区域；4. 调集抢险队伍200人和物资。"
}}

请返回 JSON 对象：
{{
  "warning_level": "I" | "II" | "III" | "IV" | "",
  "reasoning": "研判依据或法规依据（200字以内，仅供内部审计，不展示给用户）",
  "actions": ["具体应急措施1", "措施2", ...],
  "answer": "给用户的自然语言回答（200-500字，直接回答用户问题，不要重复 reasoning 字段的内容，不要以'研判依据'开头）",
  "citations": [
    {{
      "ref_id": 上下文编号（对应 web_search 结果中的 [编号]）,
      "quote": "从该搜索结果摘要中逐字摘录的原文片段（必须是 [编号] 对应摘要的精确子串）",
      "source_type": "web_search"
    }}
  ]
}}

重要约束：
1. answer 字段是直接给用户看的回答，必须是自然语言，不要包含"研判依据"等内部标签
2. 严禁捏造或猜测"用户偏好"——除非系统明确提供了偏好信息，否则不要提及"用户偏好"一词
3. 如果工具未返回数据，answer 中应明确说明"未能获取实时数据"并建议用户稍后重试或查阅官方渠道
4. reasoning 字段填写客观的研判过程，answer 字段填写面向用户的最终回答，两者内容不要重复
5. citations 数组只能引用 web_search 工具返回的搜索结果（上下文中标注为"搜索到 N 条结果"的条目）
6. 引用的 quote 必须是搜索结果摘要（snippet）的精确子串，不能修改、拼接或编造
7. 如果没有调用 web_search 工具，或搜索结果中没有相关内容，citations 返回空数组 []
8. 不要引用水文、天气、径流、GIS、法规等工具返回的数据作为 citation

仅返回 JSON 对象，不要其他内容。"""


# 两阶段流式 Phase 2 专用提示词：只生成面向用户的纯文本回答，不输出 JSON。
# Phase 1 已通过 SYNTH_META_SCHEMA 产出 warning_level/reasoning/actions/citations，
# 这里不再要求模型重复输出任何结构化字段，避免 JSON 外壳被当作 answer 流式展示。
SYNTH_ANSWER_PROMPT = """你是黄河吕梁段防汛预警智能体的综合研判模块，负责基于已确认的研判结论生成面向用户的最终回答。

系统已确认以下结论（必须保持一致，但不要重复输出这些字段）：
- 预警等级（warning_level）
- 分析推理（reasoning）
- 建议措施（actions）
- 引用来源（citations）

要求：
1. 只输出面向用户的自然语言回答，不要输出 JSON 对象，不要输出代码块，不要输出 "warning_level"、"reasoning"、"actions"、"citations" 等字段名或键值对。
2. 直接回答用户问题，内容应覆盖上述结论中的关键信息，200-500 字。
3. 涉及具体数据、法规条款、阈值标准或搜索到的信息时，在句末用 [编号] 标注来源，编号必须与工具结果中的 [编号] 一致；没有对应来源就不标注，严禁编造编号。
4. 如果工具未返回数据，明确说明"未能获取实时数据"，并建议稍后重试或查阅官方渠道。
5. 不要以"研判依据"、"reasoning"等内部标签开头。"""


# 引用规范指引（追加到 system prompt）
CITATION_GUIDANCE = """

【引用规范（强制）】
上下文中带 [编号] 的条目——联网搜索结果与法规检索条款——是仅有的两类可引用来源。你在生成 answer 时：
1. 仅当某句事实来自上述可引用来源时，才在该句末尾标注对应 [编号]；
   其他工具数据（【实时水情】【天气预报】【径流预测】【GIS 地形分析】等语义标签条目）
   不带编号，用自然语言说明来源即可（如"据实时水情数据""按阈值标准"）。
2. citations 数组列出所有引用，每条包含：
   - ref_id: 对应编号（整数）
   - quote: 从该来源展示文本中逐字摘录的原文片段（必须是来源中真实存在的文字，严禁改写或编造）
   - source_type: web_search（联网搜索）/ regulation（法规条款）
3. 若某信息无法在可引用来源中找到原文出处，不要编造引用，宁可不标注。
   无检索结果时 citations 留空、answer 中不出现任何 [编号] 标记。
"""


# Structured Outputs JSON Schema（强制 LLM 返回合法 JSON）
# 增加 citations 字段：每条引用含编号、原文片段、来源类型
SYNTH_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "synth_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "warning_level": {"type": "string", "enum": ["I", "II", "III", "IV", ""]},
                "reasoning": {"type": "string"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_id": {"type": "integer"},
                            "quote": {"type": "string"},
                            "source_type": {
                                "type": "string",
                                "enum": ["web_search"],
                            },
                        },
                        "required": ["ref_id", "quote", "source_type"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["warning_level", "reasoning", "actions", "answer", "citations"],
            "additionalProperties": False,
        },
    },
}

# 两阶段流式 Phase 1 专用 schema：仅 metadata（不含 answer）
# answer 由 Phase 2 stream=True 逐 token 生成，避免非流式生成后伪切分
SYNTH_META_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "synth_meta",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "warning_level": {"type": "string", "enum": ["I", "II", "III", "IV", ""]},
                "reasoning": {"type": "string"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_id": {"type": "integer"},
                            "quote": {"type": "string"},
                            "source_type": {
                                "type": "string",
                                "enum": ["web_search"],
                            },
                        },
                        "required": ["ref_id", "quote", "source_type"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["warning_level", "reasoning", "actions", "citations"],
            "additionalProperties": False,
        },
    },
}
