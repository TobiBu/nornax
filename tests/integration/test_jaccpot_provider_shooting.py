"""Shooting nodes on jaccpot's fused lane (S10, track N).

:func:`~nornax.solvers.leapfrog_kdk.shooting_node` with
:class:`~nornax.adapters.jaccpot.FusedProviderForce` (a
:class:`~nornax.forces.base.RebuildingForceModel`): the node carries the evaluation's
aux, so a :func:`~nornax.solvers.leapfrog_kdk.leapfrog_kdk_rollout` from it runs and
is the rollout from ``initialize_block_state`` bit for bit. On CPU, N = 512, float64,
order 2, every Pallas kernel of the lane in interpret mode
(``test_jaccpot_provider_gradients.py``'s set), ``jax.default_backend`` reporting
``"gpu"``, ``mac_type`` ``dehnen`` and ``dehnen_error`` (seeded with the prepare's
force scale). Skipped when jaccpot is not importable or has no
``differentiable_step``.

* node + rollout == ``initialize_block_state`` + rollout: state and aux, bitwise, for
  the default force and ``differentiable=True``;
* two segments chained through a node at the seam (``differentiable=True``) are the
  four-step rollout bitwise, the seam's ``shooting_defect`` exactly zero; under
  ``dehnen_error`` the node is seeded with the force scale the long rollout's step 3
  rebuilt with (recorded).

Each configuration is two jitted programs. The module took 11.5 min on CPU
(2026-11-01, jaccpot s9/session2 342d75f9, the host shared with the full suite):
120-125 s for each default-force case, 171 s (dehnen) and 271 s (dehnen_error)
for the differentiable ones with the chain; every comparison 0 differing bits.
"""

from __future__ import annotations

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jaccpot = pytest.importorskip("jaccpot", reason="jaccpot is not importable")
if not hasattr(jaccpot.FusedForceProvider, "differentiable_step"):
    pytest.skip(
        "jaccpot has no FusedForceProvider.differentiable_step", allow_module_level=True
    )

from nornax.adapters import FusedProviderForce  # noqa: E402
from nornax.solvers.leapfrog_kdk import (  # noqa: E402
    initialize_block_state,
    leapfrog_kdk_rollout,
    shooting_defect,
    shooting_node,
)

N, LEAF, ORDER, THETA, SOFTENING = 512, 16, 2, 0.6, 1e-2
DT = 1e-3

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_WALK_PALLAS_INTERPRET",
    "JACCPOT_STRICT_SEGMENT_RETRY",
)
_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(scope="module", autouse=True)
def _lane() -> Iterator[None]:
    jax.config.update("jax_enable_x64", True)
    with pytest.MonkeyPatch.context() as mp:
        for key in _ENV_UNSET:
            mp.delenv(key, raising=False)
        for key, value in _ENV.items():
            mp.setenv(key, value)
        mp.setattr(jax, "default_backend", lambda: "gpu")
        yield


def _solver(mac_type: str) -> Any:
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    criterion = (
        dict(
            adaptive_eps=1e-5,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac_type == "dehnen_error"
        else {}
    )
    return jaccpot.FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        softening_kernel="plummer",
        working_dtype=jnp.float64,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=mac_type,
        ),
        fixed_order=ORDER,
        **criterion,
    )


_PREPARED: dict = {}


def _prepared(mac_type: str) -> dict:
    """The blobs, the provider prepared on them and the seed (once per MAC)."""
    if mac_type in _PREPARED:
        return _PREPARED[mac_type]
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    x = jnp.asarray(
        np.clip(centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1)
    )
    v = jnp.asarray(0.1 * rng.normal(size=(N, 3)))
    m = jnp.asarray(np.abs(rng.normal(size=N)) + 0.5) / N
    provider, topology = jaccpot.FusedForceProvider.prepare(
        _solver(mac_type),
        x,
        m,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        differentiable=True,
    )
    f_b = None
    if provider.scale_carry:
        _, f_b = provider.accelerations(topology, force_scale=True)
    del topology
    _PREPARED[mac_type] = dict(provider=provider, x=x, v=v, m=m, f_b=f_b)
    return _PREPARED[mac_type]


@pytest.fixture(scope="module", autouse=True)
def _drop() -> Iterator[None]:
    yield
    _PREPARED.clear()


def _leaves_equal(a: Any, b: Any) -> None:
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for p, q in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert p.dtype == q.dtype and np.array_equal(np.asarray(p), np.asarray(q))


@pytest.mark.parametrize("differentiable", [False, True], ids=["step", "diff"])
@pytest.mark.parametrize("mac_type", ["dehnen", "dehnen_error"])
def test_a_node_on_the_lane_is_its_start_state_and_chains(mac_type, differentiable):
    """Node + 4 steps == ``initialize_block_state`` + 4 steps, state and aux bitwise;
    with ``differentiable=True`` also 2 + node + 2 == 4, the seam's defect zero."""
    lane = _prepared(mac_type)
    force = FusedProviderForce(lane["provider"], differentiable=differentiable)
    seed = force.seed(lane["f_b"])

    @jax.jit
    def via_start(x, v, m, s):
        start = initialize_block_state(x, v, m, force, topology=s)
        return start, leapfrog_kdk_rollout(start, DT, force, n_steps=4)

    @jax.jit
    def via_node(x, v, m, s):
        node0 = shooting_node(x, v, m, force, k_max=0, dt_max=DT, topology=s)
        long = leapfrog_kdk_rollout(node0, DT, force, n_steps=4)
        if not differentiable:
            return node0, long
        mid, scales = leapfrog_kdk_rollout(
            node0, DT, force, n_steps=2, record_fn=lambda st: st.topology[0]
        )
        node1 = shooting_node(
            mid.positions,
            mid.velocities,
            m,
            force,
            k_max=0,
            dt_max=DT,
            base_index=mid.base_index,
            topology=force.seed(None if scales is None else scales[0]),
        )
        end = leapfrog_kdk_rollout(node1, DT, force, n_steps=2)
        return node0, long, mid, node1, end

    args = (lane["x"], lane["v"], lane["m"], seed)
    start, want = via_start(*args)
    out = via_node(*args)
    node0, long = out[:2]
    _leaves_equal(node0, start)
    _leaves_equal(long, want)
    assert force.verdict(long)[0], "the caps do not fit"
    assert (node0.topology[0] is not None) == (mac_type == "dehnen_error")
    if not differentiable:
        assert len(node0.topology) == 3  # the provider's own aux: the S4 carry
        return
    mid, node1, end = out[2:]
    assert FusedProviderForce.done(long) == 4
    for name in ("positions", "velocities", "acc"):
        got, ref = np.asarray(getattr(end, name)), np.asarray(getattr(long, name))
        assert np.array_equal(got, ref), name
    assert np.array_equal(np.asarray(node1.acc), np.asarray(mid.acc))
    for d in shooting_defect(mid, node1):
        assert not np.any(np.asarray(d))
    assert FusedProviderForce.done(end) == 2 and force.verdict(end)[0]
