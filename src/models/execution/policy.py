"""Apply runtime-only options after all modules have been initialized."""
from collections.abc import Mapping


def configure_execution(model, options: Mapping | None = None):
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

    # Components declare their runtime flags and compile targets locally. The
    # dispatcher imports neither the backbone nor an optional extension.
    import torch

    for module in model.modules():
        for key in (
            "reuse_pair_buffer", "cache_time_features", "avoid_repeat_copies",
            "prepared_layouts", "fused_pair_features",
        ):
            setattr(module, key, defaults[key])
        module._time_frequencies = None
        for attribute, option in getattr(module, "_execution_flags", {}).items():
            setattr(module, attribute, defaults[option])
        for method, option in getattr(module, "_execution_compile_targets", {}).items():
            if method is None:
                # Module.compile preserves parameter registration/checkpoint keys.
                module._compiled_call_impl = None
                if defaults[option]:
                    module.compile(
                        backend=defaults["compile_backend"],
                        dynamic=defaults["compile_dynamic"],
                    )
            else:
                current = getattr(module, method)
                original = getattr(current, "_torchdynamo_orig_callable", current)
                setattr(
                    module,
                    method,
                    torch.compile(
                        original,
                        backend=defaults["compile_backend"],
                        dynamic=defaults["compile_dynamic"],
                    ) if defaults[option] else original,
                )
    model.execution_options = defaults
    return defaults


def configure_data_execution(datamodule, flow_model):
    """Bind optional CPU preparation to the effective model, before workers start.

    In prediction, the checkpoint and ablation switch determine the architecture;
    the unrelated default training config must not decide which layouts to build.
    Custom datamodules without this optional interface remain supported.
    """
    configure = getattr(datamodule, "configure_model_execution", None)
    if configure is not None:
        configure(g1_enabled=flow_model.g1_enabled, execution=flow_model.execution_options)
