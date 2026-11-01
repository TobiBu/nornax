"""Tests for the analytic galaxy potentials (``nornax.potentials``).

The Odisseo references are COPIED here (``odisseo/potentials.py``: ``MyamotoNagai``
and ``NFW``, as numpy, the formulas operand for operand) rather than imported, so
the tests run without Odisseo; Odisseo has no Hernquist, whose reference is the
textbook one (Hernquist 1990, eqs. 2-3). Everything in float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nornax.forces.mutual_direct import MutualDirectSumGravity
from nornax.potentials import (
    G_KPC_KMS_MSUN,
    NFW,
    CompositePotential,
    Hernquist,
    HernquistParams,
    MiyamotoNagai,
    MiyamotoNagaiParams,
    NFWParams,
    nfw_mass_from_virial,
)
from nornax.solvers.leapfrog_kdk import initialize_block_state, leapfrog_kdk_rollout

G = 1.3  # not 1, so a missing G shows

CASES = {
    "miyamoto_nagai": (MiyamotoNagai(G=G), MiyamotoNagaiParams(M=2.0, a=1.5, b=0.3)),
    "nfw": (NFW(G=G), NFWParams(M=5.0, r_s=2.0)),
    "hernquist": (Hernquist(G=G), HernquistParams(M=0.7, a=0.4)),
}


def _galaxy():
    field = CompositePotential(
        disc=CASES["miyamoto_nagai"][0],
        halo=CASES["nfw"][0],
        bulge=CASES["hernquist"][0],
    )
    args = {
        "disc": CASES["miyamoto_nagai"][1],
        "halo": CASES["nfw"][1],
        "bulge": CASES["hernquist"][1],
    }
    return field, args


ALL = dict(CASES, composite=_galaxy())


def _points(n=64, seed=0, scale=3.0):
    """Random points over several decades of radius, a few near the plane."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) * scale * 10.0 ** rng.uniform(-2, 1, (n, 1))
    x[:4, 2] = rng.normal(size=4) * 1e-3  # near the disc plane
    return jnp.asarray(x)


# --- Odisseo's formulas (odisseo/potentials.py, copied), and the textbook Hernquist --


def _odisseo_mn(pos, G, M, a, b):
    z2 = pos[:, 2] ** 2
    Dz = a + (z2 + b**2) ** 0.5
    D = np.linalg.norm(pos[:, :2], axis=1) ** 2 + Dz**2
    K = -G * M / D ** (3 / 2)
    acc = np.stack(
        [K * pos[:, 0], K * pos[:, 1], K * pos[:, 2] * Dz / (z2 + b**2) ** 0.5], 1
    )
    return acc, -G * M / np.sqrt(D)


def _odisseo_nfw(pos, G, M, r_s):
    r = np.linalg.norm(pos, axis=1)
    x = r / r_s
    ratio = (np.log1p(x) - x / (1.0 + x)) / (x * x)  # r > 0 here: the direct branch
    acc = (-G * M / (r_s * r_s) * ratio)[:, None] * pos / r[:, None]
    return acc, -G * M / r_s * np.log1p(x) / x


def _hernquist(pos, G, M, a):
    r = np.linalg.norm(pos, axis=1)
    return (-G * M / (r * (r + a) ** 2))[:, None] * pos, -G * M / (r + a)


REFERENCE = {
    "miyamoto_nagai": _odisseo_mn,
    "nfw": _odisseo_nfw,
    "hernquist": _hernquist,
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_the_reference_formulas(name) -> None:
    """Odisseo's MN and NFW (copied), the textbook Hernquist: to round-off."""
    field, params = CASES[name]
    x = _points(seed=1)
    acc, phi = REFERENCE[name](np.asarray(x), G, *params)
    np.testing.assert_allclose(field(x, None, params), acc, rtol=1e-13, atol=0)
    np.testing.assert_allclose(field.potential(x, None, params), phi, rtol=1e-14)


@pytest.mark.parametrize("name", list(ALL))
def test_acceleration_is_minus_the_potentials_gradient(name) -> None:
    """``acc == -grad Phi`` by ``jax.grad``, float64, to round-off."""
    field, params = ALL[name]
    x = _points(seed=2)
    grad = jax.grad(lambda p: jnp.sum(field.potential(p, None, params)))(x)
    acc = field(x, None, params)
    scale = jnp.linalg.norm(acc, axis=-1, keepdims=True)
    assert float(jnp.max(jnp.abs(acc + grad) / scale)) < 1e-12


def test_the_enclosed_masses_fix_the_mass_conventions() -> None:
    """Radial force ``G M(<r) / r^2``: NFW ``M [ln(1 + x) - x / (1 + x)]`` (``M`` the
    characteristic mass, Odisseo's), Hernquist ``M r^2 / (r + a)^2`` (``M`` the
    total); and :func:`nfw_mass_from_virial` puts ``m_vir`` inside ``c r_s``."""
    x = _points(seed=3)
    r = jnp.linalg.norm(x, axis=-1)
    for name, enclosed in (
        ("nfw", lambda p, r: p.M * (jnp.log1p(r / p.r_s) - r / (r + p.r_s))),
        ("hernquist", lambda p, r: p.M * r**2 / (r + p.a) ** 2),
    ):
        field, p = CASES[name]
        radial = -jnp.sum(field(x, None, p) * x, axis=-1) / r
        np.testing.assert_allclose(radial, G * enclosed(p, r) / r**2, rtol=1e-12)
    m_vir, c, r_s = 3.0, 12.0, 1.7
    m = nfw_mass_from_virial(m_vir, c)
    r_vir = c * r_s
    assert float(m * (jnp.log1p(c) - r_vir / (r_vir + r_s))) == pytest.approx(m_vir)


def test_the_nfw_series_is_the_closed_form_in_extended_precision() -> None:
    """Below the switch-over the force ratio is a series: against the closed form in
    ``long double`` (x86: 64-bit mantissa; its own cancellation is ~2e-19 / x, so
    from ``1e-3``) to past the switch, and below that against the series' first
    terms (truncation ~x^3)."""
    from nornax.potentials import _nfw_force_ratio

    x = np.geomspace(1e-3, 0.2, 200)
    xl = x.astype(np.longdouble)
    want = (np.log1p(xl) - xl / (1 + xl)) / (xl * xl)
    got = np.asarray(_nfw_force_ratio(jnp.asarray(x))).astype(np.longdouble)
    assert float(np.max(np.abs(got / want - 1))) < 2e-14
    tiny = np.geomspace(1e-9, 1e-5, 20)
    np.testing.assert_allclose(
        _nfw_force_ratio(jnp.asarray(tiny)),
        0.5 - 2 * tiny / 3 + 0.75 * tiny**2,
        rtol=4e-15,  # the dropped 4 x^3 / 5: 1.6e-15 at 1e-5
    )
    assert float(_nfw_force_ratio(jnp.zeros(()))) == 0.5
    # float32 switches later (0.3) and stays near its own round-off
    x32 = np.geomspace(1e-4, 1.0, 100).astype(np.float32)
    got32 = np.asarray(_nfw_force_ratio(jnp.asarray(x32))).astype(np.longdouble)
    want32 = (
        np.log1p(x32.astype(np.longdouble)) - x32 / (1 + x32.astype(np.longdouble))
    ) / (x32.astype(np.longdouble) ** 2)
    assert float(np.max(np.abs(got32 / want32 - 1))) < 5e-6


def test_miyamoto_nagai_limits() -> None:
    """``a = 0``: a Plummer sphere of scale ``b``. ``b -> 0``: the Kuzmin disc off the
    plane, and on it no vertical force; at ``b = 0`` the plane itself is ``NaN``
    (documented: the vertical force jumps there)."""
    field = MiyamotoNagai(G=G)
    x = _points(seed=4)
    r2 = jnp.sum(x * x, axis=-1)
    M, b = 2.0, 0.3
    plummer = MiyamotoNagaiParams(M=M, a=0.0, b=b)
    np.testing.assert_allclose(
        field(x, None, plummer),
        -G * M * x / ((r2 + b * b) ** 1.5)[:, None],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        field.potential(x, None, plummer), -G * M / jnp.sqrt(r2 + b * b), rtol=1e-14
    )

    a = 1.5
    thin = MiyamotoNagaiParams(M=M, a=a, b=1e-9)
    off = x[jnp.abs(x[:, 2]) > 1e-2]
    z = off[:, 2]
    d2 = jnp.sum(off[:, :2] ** 2, axis=-1) + (a + jnp.abs(z)) ** 2
    kuzmin_z = -G * M * jnp.sign(z) * (a + jnp.abs(z)) / d2**1.5
    np.testing.assert_allclose(field(off, None, thin)[:, 2], kuzmin_z, rtol=1e-12)
    on_plane = x.at[:, 2].set(0.0)
    assert jnp.all(field(on_plane, None, thin)[:, 2] == 0.0)
    razor = field(on_plane, None, MiyamotoNagaiParams(M=M, a=a, b=0.0))
    assert jnp.all(jnp.isnan(razor[:, 2])) and jnp.all(jnp.isfinite(razor[:, :2]))


@pytest.mark.parametrize("name", list(ALL))
def test_parameter_gradients_match_central_differences(name) -> None:
    """d/d(every parameter) of a loss over the field and the potential, float64."""
    field, params = ALL[name]
    x = _points(seed=5)
    rng = np.random.default_rng(6)
    w = jnp.asarray(rng.normal(size=x.shape))
    u = jnp.asarray(rng.normal(size=x.shape[0]))

    def loss(p):
        return jnp.sum(w * field(x, None, p)) + jnp.sum(u * field.potential(x, None, p))

    grads = jax.grad(loss)(jax.tree.map(jnp.asarray, params))
    leaves, treedef = jax.tree.flatten(params)
    for i, (value, g) in enumerate(zip(leaves, jax.tree.leaves(grads))):
        h = 1e-6 * abs(value)

        def at(v):
            return loss(jax.tree.unflatten(treedef, leaves[:i] + [v] + leaves[i + 1 :]))

        fd = (at(value + h) - at(value - h)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-7), (name, i)


@pytest.mark.parametrize("name", ["nfw", "hernquist", "miyamoto_nagai"])
def test_finite_at_the_centre(name) -> None:
    """At ``r = 0``: zero force, the finite central potential, and finite gradients in
    the positions and every parameter (no ``NaN`` from a particle on the centre)."""
    field, params = CASES[name]
    x = jnp.zeros((2, 3)).at[1].set(jnp.asarray([0.3, -0.2, 0.1]))
    acc = field(x, None, params)
    phi = field.potential(x, None, params)
    assert jnp.array_equal(acc[0], jnp.zeros(3))
    centre = {
        "nfw": lambda p: -G * p.M / p.r_s,
        "hernquist": lambda p: -G * p.M / p.a,
        "miyamoto_nagai": lambda p: -G * p.M / (p.a + p.b),
    }[name](params)
    assert float(phi[0]) == pytest.approx(centre, rel=1e-15)

    def loss(p, prm):
        return jnp.sum(field(p, None, prm) ** 2) + jnp.sum(
            field.potential(p, None, prm)
        )

    gx, gp = jax.grad(loss, argnums=(0, 1))(x, jax.tree.map(jnp.asarray, params))
    assert jnp.all(jnp.isfinite(gx))
    assert all(bool(jnp.isfinite(g)) for g in jax.tree.leaves(gp))


def test_the_cusps_force_tends_to_its_limit() -> None:
    """Approaching the centre the force magnitude tends to ``G M / (2 r_s^2)`` (NFW)
    and ``G M / a^2`` (Hernquist) -- the series branch carries NFW there."""
    for name, limit in (
        ("nfw", lambda p: G * p.M / (2 * p.r_s**2)),
        ("hernquist", lambda p: G * p.M / p.a**2),
    ):
        field, p = CASES[name]
        x = jnp.asarray([[1e-9, 0.0, 0.0], [0.0, -1e-9, 0.0]])
        mag = jnp.linalg.norm(field(x, None, p), axis=-1)
        np.testing.assert_allclose(mag, limit(p), rtol=1e-8)


@pytest.mark.parametrize("name", list(ALL))
def test_jit_vmap_and_mapping_args(name) -> None:
    """``jit`` is eager to round-off; ``vmap`` over a batch of parameters is the loop;
    a plain mapping with the same keys is the ``NamedTuple``."""
    field, params = ALL[name]
    x = _points(seed=7)
    eager = field(x, None, params)
    np.testing.assert_allclose(
        jax.jit(field)(x, None, params), eager, rtol=1e-14, atol=0
    )
    batch = jax.tree.map(
        lambda v: jnp.asarray(v) * jnp.asarray([1.0, 1.1, 0.9]), params
    )
    batched = jax.vmap(lambda p: field(x, None, p))(batch)
    for k in range(3):
        one = jax.tree.map(lambda v: v[k], batch)
        np.testing.assert_allclose(batched[k], field(x, None, one), rtol=1e-14)
    as_dict = (
        {k: v._asdict() for k, v in params.items()}
        if isinstance(params, dict)
        else params._asdict()
    )
    assert jnp.array_equal(field(x, None, as_dict), eager)


def test_the_composite_sums_its_components_in_order() -> None:
    """Sum of the components, bit for bit in the order given; the gradient has the
    args' structure; a mapping and keywords combine; no component or a repeat raises."""
    field, args = _galaxy()
    x = _points(seed=8)
    want = (
        CASES["miyamoto_nagai"][0](x, None, args["disc"])
        + CASES["nfw"][0](x, None, args["halo"])
        + CASES["hernquist"][0](x, None, args["bulge"])
    )
    assert jnp.array_equal(field(x, None, args), want)
    grads = jax.grad(lambda a: jnp.sum(field.potential(x, None, a)))(
        jax.tree.map(jnp.asarray, args)
    )
    assert jax.tree.structure(grads) == jax.tree.structure(args)
    mixed = CompositePotential({"disc": MiyamotoNagai(G=G)}, halo=NFW(G=G))
    assert [n for n, _ in mixed.components] == ["disc", "halo"]
    nested = CompositePotential(inner=mixed, bulge=Hernquist(G=G))
    nested_args = {
        "inner": {"disc": args["disc"], "halo": args["halo"]},
        "bulge": args["bulge"],
    }
    np.testing.assert_allclose(
        field(x, None, args), nested(x, None, nested_args), rtol=1e-15
    )
    hash(field)  # static configuration: usable as a jit static argument
    with pytest.raises(ValueError, match="at least one"):
        CompositePotential()
    with pytest.raises(ValueError, match="twice"):
        CompositePotential({"disc": MiyamotoNagai()}, disc=NFW())


def test_dtype_follows_the_positions_and_time_is_ignored() -> None:
    """float32 positions give float32 fields; ``time`` does not enter."""
    field, args = _galaxy()
    x = _points(seed=9).astype(jnp.float32)
    assert field(x, None, args).dtype == jnp.float32
    assert field.potential(x, None, args).dtype == jnp.float32
    assert jnp.array_equal(field(x, 3.0, args), field(x, None, args))
    assert G_KPC_KMS_MSUN == pytest.approx(4.30091727e-6)


def test_drives_a_rollout_with_differentiable_parameters() -> None:
    """As ``leapfrog_kdk_rollout``'s external field: the gradient in every parameter
    matches central differences, and a test particle's energy in the static field is
    conserved to O(dt^2)."""
    field, args = _galaxy()
    rng = np.random.default_rng(10)
    x = jnp.asarray(rng.normal(size=(8, 3)))
    v = jnp.asarray(0.3 * rng.normal(size=(8, 3)))
    m = jnp.full((8,), 1e-3)
    force = MutualDirectSumGravity(softening=0.1)

    def final(theta, dt=0.01, n_steps=20):
        state = initialize_block_state(
            x, v, m, force, external_fn=field, external_args=theta
        )
        return leapfrog_kdk_rollout(
            state, dt, force, n_steps=n_steps, external_fn=field, external_args=theta
        )

    def loss(theta):
        out = final(theta)
        return jnp.sum(out.positions**2) + jnp.sum(out.velocities**2)

    theta = jax.tree.map(jnp.asarray, args)
    grads = jax.jit(jax.grad(loss))(theta)
    leaves, treedef = jax.tree.flatten(theta)
    for i, g in enumerate(jax.tree.leaves(grads)):
        h = 1e-6 * abs(float(leaves[i]))

        def at(d):
            moved = leaves[:i] + [leaves[i] + d] + leaves[i + 1 :]
            return loss(jax.tree.unflatten(treedef, moved))

        fd = (at(h) - at(-h)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-6), i

    # one test particle (zero self-gravity): energy drift ~ dt^2
    def drift(dt):
        state = initialize_block_state(
            x[:1], v[:1], m[:1], force, external_fn=field, external_args=args
        )
        out = leapfrog_kdk_rollout(
            state,
            dt,
            force,
            n_steps=int(round(2.0 / dt)),
            external_fn=field,
            external_args=args,
        )

        def energy(s):
            return (
                0.5 * jnp.sum(s.velocities**2)
                + field.potential(s.positions, None, args)[0]
            )

        return abs(float(energy(out) - energy(state)))

    coarse, fine = drift(0.02), drift(0.01)
    assert fine < 1e-3 and coarse / fine > 3.0
