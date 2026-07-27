"""骨架冒烟：train 包可导入，且能引用 agent/backend 现有模块。"""


def test_train_package_importable():
    import train  # noqa: F401


def test_existing_modules_importable():
    import agent.tools.schemas  # noqa: F401
    import app.core.config  # noqa: F401
