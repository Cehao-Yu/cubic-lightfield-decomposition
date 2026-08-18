"""Checks of the cubic decomposition against cases with a known answer.

Run: python test_lightfield.py
"""

import numpy as np

from lightfield import axis_components, decompose, vector_direction, rate_of_change

TOL = 1e-12


def check(name, got, want, tol=TOL):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    ok = np.allclose(got, want, atol=tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got  {got}\n      want {want}")
    return ok


def main():
    nwl = 4
    results = []

    # 1. Purely isotropic light: no directional component, diffuse equals the
    #    common level.
    E = np.full((6, nwl), 2.0)
    directional, diffuse = decompose(E)
    results.append(check("isotropic: directional is zero", directional, np.zeros(nwl)))
    results.append(check("isotropic: diffuse equals the level", diffuse, np.full(nwl, 2.0)))

    # 2. Isotropic pedestal plus a collimated beam arriving from due east.
    ambient, beam = 2.0, 3.0
    E = np.full((6, nwl), ambient)
    E[0] += beam                       # x+ faces east
    directional, diffuse = decompose(E)
    altitude, azimuth = vector_direction(E)
    results.append(check("beam: directional equals the beam", directional, np.full(nwl, beam)))
    results.append(check("beam: diffuse equals the pedestal", diffuse, np.full(nwl, ambient)))
    results.append(check("beam: altitude is zero", altitude, 0.0))
    results.append(check("beam: azimuth is 90 deg (east)", azimuth, 90.0))

    # 3. Beam from directly overhead.
    E = np.full((6, nwl), ambient)
    E[4] += beam                       # z+ faces up
    altitude, _ = vector_direction(E)
    results.append(check("zenith beam: altitude is 90 deg", altitude, 90.0))

    # 4. The symmetric term is the minimum of each opposing pair.
    rng = np.random.default_rng(0)
    E = rng.uniform(0.1, 5.0, size=(6, nwl))
    _, symmetric = axis_components(E)
    expected = np.minimum(E[0::2], E[1::2])
    results.append(check("symmetric term equals the pairwise minimum", symmetric, expected))

    # 5. Directional magnitude is invariant to swapping a pair, which only
    #    flips the sign of that axis component.
    E_swapped = E.copy()
    E_swapped[[0, 1]] = E_swapped[[1, 0]]
    results.append(check("directional magnitude is sign-invariant",
                         decompose(E)[0], decompose(E_swapped)[0]))

    # 6. Rates use the true elapsed interval, not a nominal constant.
    values = np.array([0.0, 10.0, 10.0, 25.0])
    times = np.array([0.0, 5.0, 15.0, 20.0])
    results.append(check("rate of change", rate_of_change(values, times),
                         [2.0, 0.0, 3.0]))

    # 7. A series of samples decomposes elementwise.
    E_series = rng.uniform(0.1, 5.0, size=(7, 6, nwl))
    directional, diffuse = decompose(E_series)
    one_by_one = np.array([decompose(E_series[i])[0] for i in range(7)])
    results.append(check("vectorised over samples", directional, one_by_one))

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
