"""One G1 atom update composed from all six independently configurable modules."""

from typing import Literal, Mapping

from torch import Tensor, nn

from .aggregation import GeometryAggregator
from .conditioning import ConditionEncoder
from .geometry import PeriodicGeometry
from .geometry.groups import prepare_group_internal
from .group_attention import GroupInteraction
from .outputs import G1Context, G1Intermediates, G1Output
from .readback import AtomReadback
from .routing import SoftRouter
from .validation import (
    require_module,
    validate_components,
    validate_condition,
    validate_context,
    validate_mask,
)


class G1Module(nn.Module):
    """Condition + geometry -> routing -> U/Z + S_plus -> atom readback/update.

    H is provided by an atom backbone; this module does not initialize H,
    predict coordinate/lattice velocities or select insertion points in Flow.
    Child modules remain separately registered and replaceable. Fresh defaults
    have independent parameters; explicitly supplied instances retain theirs.
    """

    def __init__(
        self,
        condition_encoder: ConditionEncoder | None = None,
        geometry: PeriodicGeometry | None = None,
        router: SoftRouter | None = None,
        aggregation: GeometryAggregator | None = None,
        group_interaction: GroupInteraction | None = None,
        readback: AtomReadback | None = None,
    ):
        super().__init__()
        for name, component in (
            ("condition_encoder", condition_encoder),
            ("geometry", geometry),
            ("router", router),
            ("aggregation", aggregation),
            ("group_interaction", group_interaction),
            ("readback", readback),
        ):
            require_module(name, component)
        self.condition_encoder = (
            ConditionEncoder() if condition_encoder is None else condition_encoder
        )
        self.geometry = PeriodicGeometry() if geometry is None else geometry
        self.router = (
            SoftRouter(c_dim=self.condition_encoder.output_dim) if router is None else router
        )
        self.aggregation = GeometryAggregator() if aggregation is None else aggregation
        self.group_interaction = (
            GroupInteraction(h_dim=self.router.query_encoder.h_dim)
            if group_interaction is None
            else group_interaction
        )
        self.readback = (
            AtomReadback(
                h_dim=self.router.query_encoder.h_dim,
                group_dim=self.group_interaction.content_pool.value_projection.out_features,
                pool_eps=self.router.pool_eps,
            )
            if readback is None
            else readback
        )
        validate_components(
            self.condition_encoder, self.router, self.group_interaction, self.readback
        )
        self.h_dim = self.router.query_encoder.h_dim
        self.c_dim = self.router.query_encoder.c_dim

    def encode_condition(self, feats: Mapping[str, Tensor]) -> Tensor:
        """Encode static chemical conditions only; explicit reuse is caller-managed."""
        mask = validate_mask(feats["atom_pad_mask"])
        c_cond = self.condition_encoder(feats)
        validate_condition(c_cond, mask, self.c_dim)
        return c_cond

    def prepare(
        self,
        feats: Mapping[str, Tensor],
        x_t: Tensor,
        l_t: Tensor,
        *,
        c_cond: Tensor | None = None,
        lattice_format: Literal["parameters", "cell"] = "parameters",
    ) -> G1Context:
        """Prepare static conditions and CURRENT geometry, without routing.

        An explicit c_cond skips condition encoding (useful in inference).
        Every call recomputes geometry. A resulting context can be reused only
        while conditions, atom ordering/mask, x_t and l_t remain unchanged.
        No detach, hidden cache or dtype/unit conversion is added here.
        """
        if lattice_format not in ("parameters", "cell"):
            raise ValueError("lattice_format must be 'parameters' or 'cell'")
        mask = validate_mask(feats["atom_pad_mask"])
        if c_cond is None:
            c_cond = self.encode_condition(feats)
        else:
            validate_condition(c_cond, mask, self.c_dim)
        if lattice_format == "parameters":
            geo = self.geometry(x_t, l_t, mask)
        else:
            geo = self.geometry.from_cell(x_t, l_t, mask)
        prepared = (
            self.geometry.aggregation_layout(geo)
            if type(self.geometry) is PeriodicGeometry
            and getattr(self.geometry, "geometry_storage_reuse", False)
            else None
        )
        return G1Context(
            c_cond, geo, mask.clone(), self.router.group_selector.prepare(mask), prepared
        )

    def forward_with_context(
        self,
        H: Tensor,
        t: Tensor,
        context: G1Context,
        *,
        return_intermediates: bool = False,
    ) -> G1Output:
        """Reuse prepared conditions/geometry, but recompute routes for the current H/t."""
        validate_context(H, t, context, self.h_dim, self.c_dim)
        c_cond, geo, mask = context.c_cond, context.geometry, context.atom_mask
        routes = self.router(
            H,
            c_cond,
            geo.rho,
            t,
            geo.geometry_ok,
            mask,
            group_layout=context.group_layout,
        )
        if context.prepared_geometry is None:
            relations = self.aggregation(geo.E, routes.Pbar, mask, routes.group_mask, geo.geometry_ok)
        else:
            relations = self.aggregation(
                geo.E, routes.Pbar, mask, routes.group_mask, geo.geometry_ok,
                prepared_geometry=context.prepared_geometry,
            )
        internal = prepare_group_internal(relations.Z, routes.group_mask, geo.geometry_ok)
        groups = self.group_interaction(
            H,
            routes.Pbar,
            relations.Z,
            mask,
            routes.group_mask,
            geo.geometry_ok,
            internal_geometry=internal,
        )
        result = self.readback(
            H,
            groups.S_plus,
            relations.U,
            relations.Z,
            routes.P,
            mask,
            routes.group_mask,
            geo.geometry_ok,
            internal_geometry=internal,
        )
        intermediates = (
            G1Intermediates(context, routes, relations, groups, result)
            if return_intermediates
            else None
        )
        # AtomReadback already applies H + delta_H exactly once.
        return G1Output(
            result.H_updated,
            result.delta_H,
            result.readback,
            context.group_layout.counts,
            geo.geometry_ok,
            intermediates,
        )

    def forward(
        self,
        H: Tensor,
        x_t: Tensor,
        l_t: Tensor,
        t: Tensor,
        feats: Mapping[str, Tensor],
        *,
        return_intermediates: bool = False,
        lattice_format: Literal["parameters", "cell"] = "parameters",
    ) -> G1Output:
        """Run all six modules once, returning the atom update and optional diagnostics."""
        context = self.prepare(feats, x_t, l_t, lattice_format=lattice_format)
        return self.forward_with_context(H, t, context, return_intermediates=return_intermediates)
