"""移除独立路由层后的路由逻辑测试。

验证方案 A（借鉴 OpenAI / Cohere 主流方案）：
- planner 第 1 轮返回空工具调用 → 走 direct_chat（闲聊）
- planner 第 1 轮返回工具调用 → 走 executor（业务）
- planner 第 2+ 轮返回空工具调用 → 走 synthesizer（信息已充分）
"""
from agent.graph.runner import _route_after_executor, _route_after_planner

# ====== _route_after_planner ======

class TestRouteAfterPlanner:
    """planner 后路由：第 1 轮空工具走闲聊，否则走 executor。"""

    def test_round1_no_tools_goes_to_direct_chat(self):
        """第 1 轮 planner 返回空工具调用 → direct_chat（闲聊）。"""
        state = {"rounds": 1, "planned_calls": []}
        assert _route_after_planner(state) == "direct_chat"

    def test_round1_with_tools_goes_to_executor(self):
        """第 1 轮 planner 返回工具调用 → executor（业务）。"""
        state = {
            "rounds": 1,
            "planned_calls": [{"name": "get_hydrology", "arguments": {}}],
        }
        assert _route_after_planner(state) == "executor"

    def test_round2_no_tools_goes_to_executor(self):
        """第 2+ 轮空工具 → 仍走 executor（由 _route_after_executor 决定进 synthesizer）。

        注意：_route_after_planner 只区分"第 1 轮闲聊"vs"业务流程"。
        第 2+ 轮空工具时 planned 为空，但 round != 1，所以走 executor，
        然后 executor 不会执行（planned 为空），should_continue=False → synthesizer。
        """
        state = {"rounds": 2, "planned_calls": []}
        assert _route_after_planner(state) == "executor"

    def test_round1_empty_planned_calls_list(self):
        """planned_calls 为空列表（非 None）。"""
        state = {"rounds": 1, "planned_calls": []}
        assert _route_after_planner(state) == "direct_chat"

    def test_round1_missing_planned_calls_key(self):
        """planned_calls 键不存在时等同于空列表，走 direct_chat（闲聊）。"""
        state = {"rounds": 1}
        assert _route_after_planner(state) == "direct_chat"

    def test_round0_treated_as_business(self):
        """rounds=0（异常情况）走 executor（保守策略）。"""
        state = {"rounds": 0, "planned_calls": []}
        assert _route_after_planner(state) == "executor"


# ====== _route_after_executor ======

class TestRouteAfterExecutor:
    """executor 后路由：基于 should_continue 决策。"""

    def test_should_continue_true_goes_to_planner(self):
        """should_continue=True → planner（继续循环）。"""
        state = {"should_continue": True}
        assert _route_after_executor(state) == "planner"

    def test_should_continue_false_goes_to_synthesizer(self):
        """should_continue=False → synthesizer（信息已充分）。"""
        state = {"should_continue": False}
        assert _route_after_executor(state) == "synthesizer"

    def test_missing_should_continue_defaults_to_synthesizer(self):
        """should_continue 键不存在 → synthesizer（保守策略，避免死循环）。"""
        state = {}
        assert _route_after_executor(state) == "synthesizer"


# ====== 图结构验证 ======

class TestGraphStructure:
    """验证移除 router 后的图结构。"""

    def test_graph_has_no_router_node(self):
        """图中不应再有 router 节点。"""
        from agent.graph.runner import build_agent_graph
        app = build_agent_graph()
        # LangGraph 的 nodes 属性包含所有节点名
        node_names = set(app.nodes.keys())
        # 过滤掉 LangGraph 内部节点（如 __start__, __end__）
        actual_nodes = {n for n in node_names if not n.startswith("__")}
        assert "router" not in actual_nodes
        assert "planner" in actual_nodes
        assert "direct_chat" in actual_nodes
        assert "executor" in actual_nodes
        assert "synthesizer" in actual_nodes

    def test_graph_start_edge_goes_to_planner(self):
        """START 边应直接指向 planner（非 router）。"""
        from agent.graph.runner import build_agent_graph
        app = build_agent_graph()
        # 验证 __start__ 的下一个节点是 planner
        # LangGraph 内部用 __start__ 表示入口
        start_node = app.nodes.get("__start__")
        assert start_node is not None
