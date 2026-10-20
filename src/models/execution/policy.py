"""Apply runtime-only options after all modules have been initialized."""
from collections.abc import Mapping


def configure_execution(model, options: Mapping | None = None):
    from src.models.g1.readback import AtomReadback
    from src.models.simple.attention import AttentionPairBias
    from src.models.simple.transformers import DiT, DiTBlock

    cfg = dict(options or {})
    defaults = dict(
        reuse_pair_buffer=False,
        cache_time_features=False,
        aggregation_layout_reuse=False,
        packed_block_attention=False,
        compile_geometry_moments=False,
        compile_readback=False,
        compile_token_pairs=False,
        pair_norm_reuse=False,
        pair_bias_chunk_size=1,
        fused_pair_features=False,
        packed_geometry=False,
        bucket_geometry_images=False,
        joint_time_metrics=False,
        broadcast_masks=False,
        defer_validation=False,
        prepared_layouts=False,
        cache_sampling_conditions=False,
        fused_qkvg=False,
        sdpa_backend="auto",
        fold_readback=False,
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
        if (
            name not in ("sdpa_backend", "compile_backend", "pair_bias_chunk_size")
            and type(value) is not bool
        ):
            raise TypeError(f"execution.{name} must be bool")
    if (
        type(defaults["pair_bias_chunk_size"]) is not int
        or defaults["pair_bias_chunk_size"] < 1
    ):
        raise ValueError("execution.pair_bias_chunk_size must be a positive integer")
    if defaults["pair_bias_chunk_size"] > 1 and not defaults["pair_norm_reuse"]:
        raise ValueError("Grouped pair projections require pair_norm_reuse")
    if defaults["sdpa_backend"] not in ("auto", "math", "efficient", "flash", "cudnn"):
        raise ValueError("Unsupported execution.sdpa_backend")
    if not isinstance(defaults["compile_backend"], str):
        raise TypeError("execution.compile_backend must be a string")

    # Plain attributes add neither parameters nor state_dict keys.
    for module in model.modules():
        for key in (
            "reuse_pair_buffer",
            "cache_time_features",
            "aggregation_layout_reuse",
            "packed_block_attention",
        ):
            setattr(module, key, defaults[key])
        module._time_frequencies = None
        module.avoid_repeat_copies = defaults["avoid_repeat_copies"]
        module.prepared_layouts = defaults["prepared_layouts"]
        module.fused_pair_features = defaults["fused_pair_features"]
        if isinstance(module, AttentionPairBias):
            module.fused_qkvg = defaults["fused_qkvg"]
            module.broadcast_masks = defaults["broadcast_masks"]
            module.sdpa_backend = defaults["sdpa_backend"]
        if isinstance(module, DiT):
            module.pair_norm_reuse = defaults["pair_norm_reuse"]
            module.pair_bias_chunk_size = defaults["pair_bias_chunk_size"]
        if isinstance(module, AtomReadback):
            module.fold_values = defaults["fold_readback"]
        if module.__class__.__module__.startswith("src.models.g1."):
            module.geometry_layout_reuse = defaults["geometry_layout_reuse"]
            module.packed_geometry = defaults["packed_geometry"]
            module.bucket_geometry_images = defaults["bucket_geometry_images"]
    from src.models.g1.conditioning.features import AtomConditionFeatures
    from src.models.simple.layers import AtomAttentionEncoder

    for module in model.modules():
        if isinstance(module, DiTBlock) and not defaults["compile_blocks"]:
            module._compiled_call_impl = None
        if (
            isinstance(module, AtomConditionFeatures)
            and not defaults["compile_pair_features"]
        ):
            module._compiled_call_impl = None
        if isinstance(module, AtomAttentionEncoder):
            original = getattr(
                module.condition_features, "_torchdynamo_orig_callable", None
            )
            if original is not None:
                module.condition_features = original

    if defaults["compile_blocks"]:
        for module in model.modules():
            if isinstance(module, DiTBlock):
                # Module.compile keeps the original state_dict, unlike wrapping in
                # an OptimizedModule with an _orig_mod prefix. Search stays eager.
                module.compile(
                    backend=defaults["compile_backend"],
                    dynamic=defaults["compile_dynamic"],
                )
    if defaults["compile_pair_features"]:
        import torch

        for module in model.modules():
            if isinstance(module, AtomAttentionEncoder):
                module.condition_features = torch.compile(
                    module.condition_features,
                    backend=defaults["compile_backend"],
                    dynamic=defaults["compile_dynamic"],
                )
            elif isinstance(module, AtomConditionFeatures):
                module.compile(
                    backend=defaults["compile_backend"],
                    dynamic=defaults["compile_dynamic"],
                )
    # Compile only continuous cores. Cutoff decisions and validation stay eager.
    import torch
    from src.models.g1.geometry.moments import PeriodicPairMoments
    from src.models.g1.readback.values import RelationConditionedValue
    from src.models.simple.layers import TokenTransformer

    cores = (
        (PeriodicPairMoments, "_reduce", "compile_geometry_moments"),
        (RelationConditionedValue, "_pool_hidden", "compile_readback"),
        (TokenTransformer, "_pair_features", "compile_token_pairs"),
    )
    for module in model.modules():
        for cls, method, option in cores:
            if isinstance(module, cls):
                original = getattr(
                    getattr(module, method),
                    "_torchdynamo_orig_callable",
                    getattr(module, method),
                )
                setattr(
                    module,
                    method,
                    torch.compile(
                        original,
                        backend=defaults["compile_backend"],
                        dynamic=defaults["compile_dynamic"],
                    )
                    if defaults[option]
                    else original,
                )
    model.execution_options = defaults
    return defaults
