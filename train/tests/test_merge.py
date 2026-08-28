"""合并脚本：PEFT 模型 merge_and_unload 被调用，输出目录正确。"""
from unittest.mock import MagicMock, patch

from train.lora.merge import merge_adapter


def test_merge_calls_peft_api(tmp_path):
    fake_model = MagicMock()
    fake_model.merge_and_unload.return_value = fake_model
    with patch("train.lora.merge.AutoModelForCausalLM") as m_auto, \
         patch("train.lora.merge.PeftModel") as m_peft, \
         patch("train.lora.merge.torch"):
        m_peft.from_pretrained.return_value = fake_model
        out = merge_adapter("base-x", "adapter-y", str(tmp_path / "merged"))
    m_auto.from_pretrained.assert_called_once()
    m_peft.from_pretrained.assert_called_once()
    fake_model.merge_and_unload.assert_called_once()
    fake_model.save_pretrained.assert_called_once_with(str(tmp_path / "merged"))
    assert out.endswith("merged")
