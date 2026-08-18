"""
Cubic decomposition of a measured light field into its diffuse and directional
components.

The cubic illumination method records spectral irradiance on the six faces of a
cube, that is along the three Cartesian axes in both directions. This gives a
first-order spherical-harmonic description of the local light field, which
separates into

    - a DIRECTIONAL component, the illumination vector: the first-order SH term,
      describing the dominant direction from which light arrives; and
    - a DIFFUSE component, the symmetric or ambient term: equivalent to the
      zeroth-order SH light field up to a normalisation constant.

Notation follows the associated paper.

    E(l, x+), E(l, x-)    spectral irradiance measured facing +x and -x
    E(l, y+), E(l, y-)    ... and likewise for the y and z axes
    E(l, z+), E(l, z-)

    E(l, x)               axis component of the illumination vector
                          = E(l, x+) - E(l, x-)
    E(l, vector)          = [E(l, x), E(l, y), E(l, z)]
    |E(l, vector)|        magnitude of the directional component
                          = sqrt(E(l, x)^2 + E(l, y)^2 + E(l, z)^2)

    Et(l, x)              axis component of the symmetric (diffuse) term
                          = [E(l, x+) + E(l, x-) - |E(l, x)|] / 2
    Et(l)                 magnitude of the diffuse component
                          = mean of Et(l, x), Et(l, y), Et(l, z)

    theta                 altitude of the illumination vector
                          = arctan( E(l, z) / sqrt(E(l, x)^2 + E(l, y)^2) )
    phi                   azimuth of the illumination vector
                          = arctan2( E(l, y), E(l, x) )

Here l denotes wavelength; every quantity above is computed wavelength by
wavelength, so the two components are full spectra, not scalars.

Axis convention, matching the paper: +x points east, +y points north and +z
points up. Only the three opposing pairs enter the magnitude calculations, so
the magnitudes are unaffected by how the horizontal axes are labelled - but the
azimuth is not, so keep the pairing and the labelling consistent.

Note that Et(l, x) = [E(l, x+) + E(l, x-) - |E(l, x+) - E(l, x-)|] / 2 is
algebraically identical to min(E(l, x+), E(l, x-)); the code below uses the
explicit form so that it reads as written in the paper.

Requires: numpy.
"""

import numpy as np

# Order of the six faces in the input array.
FACES = ("x+", "x-", "y+", "y-", "z+", "z-")


def axis_components(E):
    """Directional and symmetric components along each axis, per wavelength.

    Parameters
    ----------
    E : array, shape (..., 6, n_wavelengths)
        Spectral irradiance on the six cube faces, in the order given by FACES,
        in W m-2 nm-1. Leading axes (for example a series of timepoints) are
        preserved.

    Returns
    -------
    vector : array, shape (..., 3, n_wavelengths)
        E(l, x), E(l, y), E(l, z) - the signed axis components of the
        illumination vector.
    symmetric : array, shape (..., 3, n_wavelengths)
        Et(l, x), Et(l, y), Et(l, z) - the axis-wise symmetric terms.
    """
    E = np.asarray(E, dtype=float)
    if E.shape[-2] != 6:
        raise ValueError("expected six faces on the second-to-last axis, "
                         f"got {E.shape[-2]}")

    plus = E[..., 0::2, :]    # x+, y+, z+
    minus = E[..., 1::2, :]   # x-, y-, z-

    vector = plus - minus                                   # E(l, axis)
    symmetric = (plus + minus - np.abs(vector)) / 2.0       # Et(l, axis)
    return vector, symmetric


def decompose(E):
    """Split a light-field measurement into directional and diffuse components.

    Parameters
    ----------
    E : array, shape (..., 6, n_wavelengths)
        As in axis_components.

    Returns
    -------
    directional : array, shape (..., n_wavelengths)
        |E(l, vector)|, the magnitude of the illumination vector, in
        W m-2 nm-1.
    diffuse : array, shape (..., n_wavelengths)
        Et(l), the magnitude of the symmetric component, in W m-2 nm-1.
    """
    vector, symmetric = axis_components(E)
    directional = np.sqrt(np.sum(vector ** 2, axis=-2))
    diffuse = np.mean(symmetric, axis=-2)
    return directional, diffuse


def vector_direction(E, weights=None):
    """Altitude and azimuth of the illumination vector, in degrees.

    A direction is a property of the light field as a whole rather than of a
    single wavelength, so the per-wavelength axis components are first collapsed
    into one vector by integrating over wavelength. Pass `weights` to integrate
    against a spectral sensitivity - the photopic luminous efficiency function
    V(l), or one of the CIE S 026 alpha-opic action spectra - to obtain the
    direction as seen by that channel.

    Parameters
    ----------
    E : array, shape (..., 6, n_wavelengths)
        As in axis_components.
    weights : array, shape (n_wavelengths,), optional
        Spectral weighting. Defaults to unweighted (radiometric) integration.

    Returns
    -------
    altitude : array, shape (...)
        Elevation above the horizontal plane, in degrees; +90 is straight up.
    azimuth : array, shape (...)
        Compass bearing of the vector, in degrees clockwise from north (+y),
        wrapped to [0, 360).
    """
    vector, _ = axis_components(E)

    if weights is None:
        vx, vy, vz = np.sum(vector, axis=-1).T
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[-1] != vector.shape[-1]:
            raise ValueError("weights must match the wavelength axis")
        vx, vy, vz = np.sum(vector * weights, axis=-1).T

    horizontal = np.hypot(vx, vy)
    altitude = np.degrees(np.arctan2(vz, horizontal))
    azimuth = np.degrees(np.arctan2(vx, vy)) % 360.0   # from north, towards east
    return altitude, azimuth


def rate_of_change(values, times):
    """Rate of change per second between successive samples.

    Sampling is not necessarily uniform: integration times are lengthened at low
    light levels, so intervals must be taken from the timestamps rather than
    assumed constant.

    Parameters
    ----------
    values : array, shape (n_samples, ...)
        Any per-sample quantity.
    times : array, shape (n_samples,)
        Sample times in seconds. Where each light-field sample consists of six
        sequentially acquired faces, use the mean of the six acquisition times.

    Returns
    -------
    array, shape (n_samples - 1, ...)
        Difference between successive samples divided by the true elapsed
        interval.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")
    shape = (-1,) + (1,) * (values.ndim - 1)
    return np.diff(values, axis=0) / dt.reshape(shape)
