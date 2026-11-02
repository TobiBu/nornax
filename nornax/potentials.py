"""Analytic galaxy potentials as external fields with traced, differentiable parameters.

Each potential is a small frozen dataclass whose only configuration is the
gravitational constant ``G`` (static, a Python float); its *parameters* -- masses
and scale lengths -- come in as the call's ``args``, so they are traced and
differentiable like any argument. A potential is an
:data:`~nornax.solvers.leapfrog_kdk.ExternalField`:

* ``field(positions, time, args) -> (N, 3)`` is the acceleration ``-grad Phi``;
* ``field.potential(positions, time, args) -> (N,)`` is ``Phi`` per particle (for
  energies: a particle's external potential energy is ``m_i Phi(x_i)``).

``time`` is accepted and ignored (every profile here is static), so a state
without a time leaf (``time=None``) works.

**Parameters and units.** ``args`` is the component's parameter ``NamedTuple``
(:class:`MiyamotoNagaiParams`, :class:`NFWParams`, :class:`HernquistParams`) or
any mapping with the same keys. Everything is in code units with ``G = 1`` by
default -- nornax's and jaccpot's convention (``MutualDirectSumGravity.G``,
``FastMultipoleMethod(G=1.0)``) and Odisseo's (``SimulationParams.G = 1.0``,
``CodeUnits(..., G=1)``). Pass ``G=`` for other units, e.g. ``4.30091727e-6``
for kpc, km/s and solar masses.

* Miyamoto-Nagai disc (``M``, ``a``, ``b``):
  ``Phi = -G M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)``. ``a = 0`` is a Plummer
  sphere of scale ``b``; ``b -> 0`` the razor-thin Kuzmin disc. Finite
  everywhere for ``b > 0``; at ``b = 0`` the vertical force jumps across the
  plane and is ``NaN`` on it.
* NFW halo (``M``, ``r_s``): ``Phi = -G M ln(1 + r/r_s) / r``. ``M`` is the
  *characteristic* mass ``4 pi rho_0 r_s^3`` (the mass inside ``r`` is
  ``M [ln(1 + x) - x / (1 + x)]``, ``x = r / r_s``), which is what Odisseo's NFW
  computes with the field it calls ``NFWParams.Mvir`` (``odisseo/potentials.py``;
  no ``ln(1 + c) - c / (1 + c)`` normalisation). Convert a virial mass with
  :func:`nfw_mass_from_virial`.
* Hernquist bulge (``M``, ``a``): ``Phi = -G M / (r + a)``, the total mass ``M``.
* :class:`CompositePotential`: the sum of named components; its ``args`` is a
  dict of the components' parameters under the same names.

**At the centre.** NFW and Hernquist have a cusp: the force magnitude tends to
``G M / (2 r_s^2)`` (NFW) and ``G M / a^2`` (Hernquist) as ``r -> 0``, but its
direction is undefined there. At exactly ``r = 0`` both return a zero
acceleration (Odisseo's NFW does the same) and the finite central potential
(``-G M / r_s`` and ``-G M / a``); their gradients there are finite (no ``NaN``
leaks into a ``jax.grad`` from a particle sitting on the centre), but not
meaningful. NFW's force uses a series for ``r < 0.05 r_s`` (``0.3 r_s`` in
float32), where the closed form cancels.

**With the integrators.** Pass a potential and its parameters as the external
field; the parameters are then differentiable through the rollout::

    from nornax.potentials import (
        CompositePotential, HernquistParams, Hernquist, MiyamotoNagai,
        MiyamotoNagaiParams, NFW, NFWParams,
    )

    galaxy = CompositePotential(
        disc=MiyamotoNagai(), halo=NFW(), bulge=Hernquist()
    )
    theta = {
        "disc": MiyamotoNagaiParams(M=1.0, a=3.0, b=0.3),
        "halo": NFWParams(M=10.0, r_s=15.0),
        "bulge": HernquistParams(M=0.2, a=0.5),
    }
    state = initialize_block_state(
        x, v, m, force, external_fn=galaxy, external_args=theta
    )

    def loss(theta):
        final = leapfrog_kdk_rollout(
            state, dt, force, n_steps=100,
            external_fn=galaxy, external_args=theta,
        )
        return objective(final)

    grads = jax.grad(loss)(theta)  # a dict of NamedTuples, like theta

(the start state's ``acc`` was built at the starting ``theta``; a loss over
``theta`` that is to see its effect on the first kick rebuilds the state inside
the loss.)

**With jaccpot's rollouts** (``f(state[N, 2, 3], args)``) slice the position row:
``lambda state, args: galaxy(state[:, 0], None, args)``. A field that reads only
the positions qualifies for jaccpot's zero velocity rows (nothing extra is stored
for its backward). This module depends on neither jaccpot nor mimirax.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp

from nornax._typing import PerParticle, Vec3


class MiyamotoNagaiParams(NamedTuple):
    """Parameters of :class:`MiyamotoNagai`.

    Attributes
    ----------
    M : Any
        The disc's total mass.
    a : Any
        The radial scale length (``a = 0``: a Plummer sphere of scale ``b``).
    b : Any
        The vertical scale height (``b > 0`` for a field finite on the plane).
    """

    M: Any
    a: Any
    b: Any


class NFWParams(NamedTuple):
    """Parameters of :class:`NFW`.

    Attributes
    ----------
    M : Any
        The characteristic mass ``4 pi rho_0 r_s^3`` (Odisseo's
        ``NFWParams.Mvir`` as its NFW uses it); see :func:`nfw_mass_from_virial`.
    r_s : Any
        The scale radius.
    """

    M: Any
    r_s: Any


class HernquistParams(NamedTuple):
    """Parameters of :class:`Hernquist`.

    Attributes
    ----------
    M : Any
        The total mass.
    a : Any
        The scale radius.
    """

    M: Any
    a: Any


def _get(args: Any, name: str) -> Any:
    """Field ``name`` of a parameter ``NamedTuple`` or mapping."""
    if isinstance(args, Mapping):
        return args[name]
    return getattr(args, name)


def _as(value: Any, positions: Vec3) -> Any:
    """``value`` as an array in the positions' dtype."""
    return jnp.asarray(value, dtype=positions.dtype)


def _safe_radius(positions: Vec3) -> tuple[Any, Any, Any]:
    """``(r, r_safe, at_centre)``: the radius, and one safe to divide by.

    ``r_safe`` is ``1`` where ``r = 0``, so ``positions / r_safe`` and every
    derivative of it stay finite at the centre (the double-``where`` idiom: a
    ``sqrt`` at zero has an infinite derivative, which a later ``where`` would
    turn into ``NaN`` in the backward).
    """
    r2 = jnp.sum(positions * positions, axis=-1)
    at_centre = r2 == 0
    r_safe = jnp.sqrt(jnp.where(at_centre, jnp.ones_like(r2), r2))
    r = jnp.where(at_centre, jnp.zeros_like(r2), r_safe)
    return r, r_safe, at_centre


@dataclass(frozen=True)
class MiyamotoNagai:
    """Miyamoto & Nagai (1975) disc: ``Phi = -G M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)``.

    The disc lies in the ``z = 0`` plane. Called as an external field with
    :class:`MiyamotoNagaiParams` (or a mapping with keys ``M``, ``a``, ``b``)
    as ``args``.

    Attributes
    ----------
    G : float
        The gravitational constant (code units: ``1``).
    """

    G: float = 1.0

    def __call__(self, positions: Vec3, time: Any, args: Any) -> Vec3:
        """The acceleration ``-grad Phi`` at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Ignored (a static profile).
        args : Any
            :class:`MiyamotoNagaiParams` or a mapping with its keys.

        Returns
        -------
        Vec3
            The acceleration, ``(N, 3)``, in the positions' dtype.
        """
        del time
        gm, a, b = self._params(positions, args)
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        zeta = jnp.sqrt(z * z + b * b)
        dz = a + zeta
        d2 = x * x + y * y + dz * dz
        k = -gm / (d2 * jnp.sqrt(d2))
        return jnp.stack([k * x, k * y, k * z * dz / zeta], axis=-1)

    def potential(self, positions: Vec3, time: Any, args: Any) -> PerParticle:
        """The potential ``Phi`` at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Ignored (a static profile).
        args : Any
            :class:`MiyamotoNagaiParams` or a mapping with its keys.

        Returns
        -------
        PerParticle
            ``Phi``, ``(N,)``, in the positions' dtype.
        """
        del time
        gm, a, b = self._params(positions, args)
        z = positions[:, 2]
        dz = a + jnp.sqrt(z * z + b * b)
        return -gm / jnp.sqrt(jnp.sum(positions[:, :2] ** 2, axis=-1) + dz * dz)

    def _params(self, positions: Vec3, args: Any) -> tuple[Any, Any, Any]:
        """``(G M, a, b)`` in the positions' dtype."""
        gm = _as(self.G, positions) * _as(_get(args, "M"), positions)
        return gm, _as(_get(args, "a"), positions), _as(_get(args, "b"), positions)


#: the NFW force's series ``(ln(1 + x) - x / (1 + x)) / x^2 = sum_k c_k x^k``,
#: ``c_k = (-1)^k (k + 1) / (k + 2)``, through ``x^11``; with the switch-over
#: below its truncation is below the closed form's cancellation error
_NFW_SERIES = tuple((-1) ** k * (k + 1) / (k + 2) for k in range(12))
#: where the series takes over: ``x^12`` against the closed form's ``~2 eps / x``
_NFW_SWITCH = {8: 0.05, 4: 0.3}


def _nfw_force_ratio(x: Any) -> Any:
    """``(ln(1 + x) - x / (1 + x)) / x^2``, stable down to ``x = 0`` (``1/2``)."""
    small = x < _NFW_SWITCH.get(jnp.dtype(x.dtype).itemsize, 0.3)
    xs = jnp.where(small, jnp.ones_like(x), x)  # the closed form off its NaNs
    closed = (jnp.log1p(xs) - xs / (1 + xs)) / (xs * xs)
    xt = jnp.where(small, x, jnp.zeros_like(x))
    series = jnp.zeros_like(x)
    for c in reversed(_NFW_SERIES):
        series = series * xt + c
    return jnp.where(small, series, closed)


def nfw_mass_from_virial(m_vir: Any, concentration: Any) -> Any:
    """The NFW characteristic mass ``M`` of a halo of virial mass ``m_vir``.

    ``M = m_vir / (ln(1 + c) - c / (1 + c))``, ``c = r_vir / r_s``: the mass
    inside ``r_vir`` is then ``m_vir``. Traceable.

    Parameters
    ----------
    m_vir : Any
        The mass inside the virial radius.
    concentration : Any
        ``r_vir / r_s``.

    Returns
    -------
    Any
        :attr:`NFWParams.M`.
    """
    c = jnp.asarray(concentration)
    return m_vir / (jnp.log1p(c) - c / (1 + c))


@dataclass(frozen=True)
class NFW:
    """Navarro, Frenk & White (1996) halo: ``Phi = -G M ln(1 + r/r_s) / r``.

    ``M`` is the characteristic mass ``4 pi rho_0 r_s^3``, Odisseo's convention
    (module docstring). Called as an external field with :class:`NFWParams` (or a
    mapping with keys ``M``, ``r_s``) as ``args``. At ``r = 0`` the acceleration
    is zero and the potential ``-G M / r_s``.

    Attributes
    ----------
    G : float
        The gravitational constant (code units: ``1``).
    """

    G: float = 1.0

    def __call__(self, positions: Vec3, time: Any, args: Any) -> Vec3:
        """The acceleration ``-grad Phi`` at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Ignored (a static profile).
        args : Any
            :class:`NFWParams` or a mapping with its keys.

        Returns
        -------
        Vec3
            The acceleration, ``(N, 3)``, in the positions' dtype.
        """
        del time
        gm, r_s = self._params(positions, args)
        r, r_safe, at_centre = _safe_radius(positions)
        magnitude = gm / (r_s * r_s) * _nfw_force_ratio(r / r_s)
        scale = jnp.where(at_centre, jnp.zeros_like(r), magnitude / r_safe)
        return -scale[:, None] * positions

    def potential(self, positions: Vec3, time: Any, args: Any) -> PerParticle:
        """The potential ``Phi`` at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Ignored (a static profile).
        args : Any
            :class:`NFWParams` or a mapping with its keys.

        Returns
        -------
        PerParticle
            ``Phi``, ``(N,)``, in the positions' dtype.
        """
        del time
        gm, r_s = self._params(positions, args)
        r, _, at_centre = _safe_radius(positions)
        x = r / r_s
        xs = jnp.where(at_centre, jnp.ones_like(x), x)
        log_ratio = jnp.where(at_centre, jnp.ones_like(x), jnp.log1p(xs) / xs)
        return -gm / r_s * log_ratio

    def _params(self, positions: Vec3, args: Any) -> tuple[Any, Any]:
        """``(G M, r_s)`` in the positions' dtype."""
        gm = _as(self.G, positions) * _as(_get(args, "M"), positions)
        return gm, _as(_get(args, "r_s"), positions)


@dataclass(frozen=True)
class Hernquist:
    """Hernquist (1990) sphere: ``Phi = -G M / (r + a)``.

    Called as an external field with :class:`HernquistParams` (or a mapping with
    keys ``M``, ``a``) as ``args``. At ``r = 0`` the acceleration is zero and the
    potential ``-G M / a``.

    Attributes
    ----------
    G : float
        The gravitational constant (code units: ``1``).
    """

    G: float = 1.0

    def __call__(self, positions: Vec3, time: Any, args: Any) -> Vec3:
        """The acceleration ``-grad Phi`` at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Ignored (a static profile).
        args : Any
            :class:`HernquistParams` or a mapping with its keys.

        Returns
        -------
        Vec3
            The acceleration, ``(N, 3)``, in the positions' dtype.
        """
        del time
        gm, a = self._params(positions, args)
        r, r_safe, at_centre = _safe_radius(positions)
        ra = r + a
        scale = jnp.where(at_centre, jnp.zeros_like(r), gm / (r_safe * ra * ra))
        return -scale[:, None] * positions

    def potential(self, positions: Vec3, time: Any, args: Any) -> PerParticle:
        """The potential ``Phi`` at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Ignored (a static profile).
        args : Any
            :class:`HernquistParams` or a mapping with its keys.

        Returns
        -------
        PerParticle
            ``Phi``, ``(N,)``, in the positions' dtype.
        """
        del time
        gm, a = self._params(positions, args)
        r, _, _ = _safe_radius(positions)
        return -gm / (r + a)

    def _params(self, positions: Vec3, args: Any) -> tuple[Any, Any]:
        """``(G M, a)`` in the positions' dtype."""
        gm = _as(self.G, positions) * _as(_get(args, "M"), positions)
        return gm, _as(_get(args, "a"), positions)


@dataclass(frozen=True, init=False)
class CompositePotential:
    """The sum of named potentials; ``args`` maps each name to its parameters.

    ``CompositePotential(disc=MiyamotoNagai(), halo=NFW())`` (or a mapping as
    the one positional argument) is called with ``args = {"disc":
    MiyamotoNagaiParams(...), "halo": NFWParams(...)}``: a pytree whose
    gradient has the same structure. Components are summed in the order given.
    Any object with ``__call__(positions, time, args)`` and ``potential(...)``
    is a component, a composite included.

    Parameters
    ----------
    components : Mapping[str, Any] | None
        Named components (first, in their order).
    **named : Any
        More named components, after them.

    Attributes
    ----------
    components : tuple[tuple[str, Any], ...]
        ``(name, potential)`` pairs, in summation order.

    Raises
    ------
    ValueError
        With no component, or a name given twice.
    """

    components: tuple[tuple[str, Any], ...]

    def __init__(
        self, components: Mapping[str, Any] | None = None, **named: Any
    ) -> None:
        pairs = list((components or {}).items())
        for name, value in named.items():
            if any(name == n for n, _ in pairs):
                raise ValueError(f"component {name!r} given twice")
            pairs.append((name, value))
        if not pairs:
            raise ValueError("a CompositePotential needs at least one component")
        object.__setattr__(self, "components", tuple(pairs))

    def __call__(self, positions: Vec3, time: Any, args: Any) -> Vec3:
        """The summed acceleration at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Passed to every component.
        args : Any
            A mapping from each component's name to its parameters.

        Returns
        -------
        Vec3
            The acceleration, ``(N, 3)``.
        """
        names = iter(self.components)
        name, field = next(names)
        acc = field(positions, time, args[name])
        for name, field in names:
            acc = acc + field(positions, time, args[name])
        return acc

    def potential(self, positions: Vec3, time: Any, args: Any) -> PerParticle:
        """The summed potential at ``positions``.

        Parameters
        ----------
        positions : Vec3
            Where to evaluate, ``(N, 3)``.
        time : Any
            Passed to every component.
        args : Any
            A mapping from each component's name to its parameters.

        Returns
        -------
        PerParticle
            ``Phi``, ``(N,)``.
        """
        names = iter(self.components)
        name, field = next(names)
        phi = field.potential(positions, time, args[name])
        for name, field in names:
            phi = phi + field.potential(positions, time, args[name])
        return phi


#: kpc (km/s)^2 / M_sun: ``G`` for lengths in kpc, velocities in km/s, masses in
#: solar masses (CODATA 2018 G and IAU 2015 nominal solar mass; astropy's value)
G_KPC_KMS_MSUN = 4.300917270036279e-06

__all__ = [
    "CompositePotential",
    "G_KPC_KMS_MSUN",
    "Hernquist",
    "HernquistParams",
    "MiyamotoNagai",
    "MiyamotoNagaiParams",
    "NFW",
    "NFWParams",
    "nfw_mass_from_virial",
]
