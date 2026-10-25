"""模块拆分后的依赖边界：图节点与记忆模块不得产生顶层导入环。"""
import ast
from pathlib import Path


def test_graph_and_memory_imports_are_acyclic():
    root = Path(__file__).resolve().parents[2]
    paths = [p for folder in ("graph", "memory")
             for p in (root / "agent" / folder).rglob("*.py")
             if p.name != "__init__.py"]
    modules = {".".join(p.relative_to(root).with_suffix("").parts): p for p in paths}
    edges = {}
    for module, path in modules.items():
        imports = set()
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        edges[module] = imports & modules.keys()

    visited = set()

    def visit(module, stack):
        assert module not in stack, "循环依赖：" + " -> ".join([*stack, module])
        if module in visited:
            return
        for dependency in edges[module]:
            visit(dependency, [*stack, module])
        visited.add(module)

    for module in edges:
        visit(module, [])
