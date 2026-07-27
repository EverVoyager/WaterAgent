"""阶段 F 水文模型模块：基于降雨的径流预测。

子模块：
- scs_cn: SCS-CN 降雨-径流经验模型（美国农业部土壤保持局）
"""
from agent.hydrology.scs_cn import predict_runoff_scs

__all__ = ["predict_runoff_scs"]
