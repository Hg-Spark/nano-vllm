"""Public API with lightweight package initialization."""

__all__ = ["LLM", "SamplingParams", "ImagePrompt"]


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm.engine.llm_engine import LLMEngine

        return LLMEngine
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    if name == "ImagePrompt":
        from nanovllm.multimodal import ImagePrompt

        return ImagePrompt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
