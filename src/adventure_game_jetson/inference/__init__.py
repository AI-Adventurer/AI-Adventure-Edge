from importlib import import_module

__all__ = [
    "MediaPipePoseExtractor",
    "CTRGCNRunner",
    "InferenceResult",
    "ActionRecognizer",
    "ActionPrediction",
    "RecognizerTimings",
    "FrameTimings",
    "RollingProfiler",
]


def __getattr__(name):
    if name == "MediaPipePoseExtractor":
        return import_module(".pose_extractor", __name__).MediaPipePoseExtractor
    if name in {"CTRGCNRunner", "InferenceResult"}:
        module = import_module(".ctrgcn_runner", __name__)
        return getattr(module, name)
    if name in {"ActionRecognizer", "ActionPrediction"}:
        module = import_module(".runtime", __name__)
        return getattr(module, name)
    if name in {"RecognizerTimings", "FrameTimings", "RollingProfiler"}:
        module = import_module(".profiling", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
