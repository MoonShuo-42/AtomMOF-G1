"""Apply runtime-only options after all modules have been initialized."""
from collections.abc import Mapping


def configure_execution(model, options: Mapping | None = None):
    from src.models.g1.readback import AtomReadback
    from src.models.simple.attention import AttentionPairBias
    from src.models.simple.transformers import DiT, DiTBlock

    cfg = dict(options or {})
    defaults = dict(
        pair_norm_reuse=False,
        fused_qkvg=False,
        native_sdpa=False,
        sdpa_backend="auto",
        fold_readback=False,
        mixed_precision=False,
        compile_blocks=False,
        compile_pair_features=False,
        compile_backend="inductor",
        compile_dynamic=True,
        geometry_layout_reuse=False,
        vectorized_metrics=False,
        avoid_repeat_copies=False,
    )
    unknown = cfg.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown execution options: {sorted(unknown)}")
    defaults.update(cfg)
    for name, value in defaults.items():
        if name not in ("sdpa_backend", "compile_backend") and type(value) is not bool:
            raise TypeError(f"execution.{name} must be bool")
    if defaults["sdpa_backend"] not in ("auto", "math", "efficient", "flash", "cudnn"):
        raise ValueError("Unsupported execution.sdpa_backend")
    if not isinstance(defaults["compile_backend"], str):
        raise TypeError("execution.compile_backend must be a string")

    # Plain attributes add neither parameters nor state_dict keys.
    for module in model.modules():
        module.avoid_repeat_copies = defaults["avoid_repeat_copies"]
        if isinstance(module, AttentionPairBias):
            module.fused_qkvg = defaults["fused_qkvg"]
            module.native_sdpa = defaults["native_sdpa"]
            module.sdpa_backend = defaults["sdpa_backend"]
        if isinstance(module, DiT):
            module.pair_norm_reuse = defaults["pair_norm_reuse"]
        if isinstance(module, AtomReadback):
            module.fold_values = defaults["fold_readback"]
        if module.__class__.__module__.startswith("src.models.g1."):
            module.mixed_precision = defaults["mixed_precision"]
            module.geometry_layout_reuse = defaults["geometry_layout_reuse"]
    if defaults["compile_blocks"]:
        for module in model.modules():
            if isinstance(module, DiTBlock):
                # Module.compile keeps the original state_dict, unlike wrapping in
                # an OptimizedModule with an _orig_mod prefix. Search stays eager.
                module.compile(
                    backend=defaults["compile_backend"], dynamic=defaults["compile_dynamic"]
                )
    if defaults["compile_pair_features"]:
        import torch

        from src.models.g1.conditioning.features import AtomConditionFeatures
        from src.models.simple.layers import AtomAttentionEncoder

        for module in model.modules():
            if isinstance(module, AtomAttentionEncoder):
                module.condition_features = torch.compile(
                    module.condition_features,
                    backend=defaults["compile_backend"],
                    dynamic=defaults["compile_dynamic"],
                )
            elif isinstance(module, AtomConditionFeatures):
                module.compile(
                    backend=defaults["compile_backend"], dynamic=defaults["compile_dynamic"]
                )
    model.execution_options = defaults
    return defaults
