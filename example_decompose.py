"""
Worked example: decompose one day of cubic light-field measurements.

Reads a cube-face spectral irradiance file in the format distributed with the
Newcastle light-field dataset, decomposes every light-field sample into its
diffuse and directional components, and writes the result alongside a short
summary.

Usage
-----
    python example_decompose.py cube_face_spectral_irradiance.csv

Input format
------------
One row per spectrum, six rows per light-field sample:

    sample, face, time_local, time_utc, E_360nm, E_361nm, ..., E_780nm

where `face` is one of up, down, north, south, east, west, and the E_ columns
are spectral irradiance in W m-2 nm-1.

Requires: numpy, pandas.
"""

import sys
import numpy as np
import pandas as pd

from lightfield import decompose, vector_direction, FACES

# Map the compass names used in the data files onto the Cartesian faces used by
# lightfield.py. The paper's convention is +x east, +y north, +z up.
FACE_FROM_COMPASS = {"east": "x+", "west": "x-",
                     "north": "y+", "south": "y-",
                     "up": "z+", "down": "z-"}


def load_day(path):
    """Return (E, times, wavelengths) for one measurement day.

    E has shape (n_samples, 6, n_wavelengths), with the six faces ordered as in
    lightfield.FACES. `times` holds the mean of each sample's six acquisition
    times, in seconds - the timestamp convention used for sampling intervals and
    rates of change.
    """
    df = pd.read_csv(path)
    wl_cols = [c for c in df.columns if c.startswith("E_") and c.endswith("nm")]
    wavelengths = np.array([int(c[2:-2]) for c in wl_cols])

    df["cartesian"] = df["face"].map(FACE_FROM_COMPASS)
    if df["cartesian"].isna().any():
        raise ValueError("unrecognised face name in the input file")
    df["seconds"] = pd.to_datetime(df["time_local"]).astype("int64") / 1e9

    samples = sorted(df["sample"].unique())
    E = np.empty((len(samples), 6, len(wl_cols)))
    times = np.empty(len(samples))

    for i, s in enumerate(samples):
        block = df[df["sample"] == s].set_index("cartesian")
        if len(block) != 6:
            raise ValueError(f"sample {s} does not have six faces")
        E[i] = block.loc[list(FACES), wl_cols].to_numpy(dtype=float)
        times[i] = block["seconds"].mean()

    return E, times, wavelengths


def main(path):
    E, times, wavelengths = load_day(path)
    directional, diffuse = decompose(E)
    altitude, azimuth = vector_direction(E)

    # Integrate each component over wavelength to get a single irradiance per
    # sample, for the summary below. Wavelengths are on a 1 nm grid, so the
    # integral is a plain sum.
    step = int(np.median(np.diff(wavelengths)))
    directional_total = directional.sum(axis=1) * step
    diffuse_total = diffuse.sum(axis=1) * step

    print(f"{len(times)} light-field samples, "
          f"{wavelengths.min()}-{wavelengths.max()} nm at {step} nm steps")
    print(f"directional irradiance : {directional_total.min():.4g} to "
          f"{directional_total.max():.4g} W m-2")
    print(f"diffuse irradiance     : {diffuse_total.min():.4g} to "
          f"{diffuse_total.max():.4g} W m-2")
    print(f"ratio directional/diffuse, median : "
          f"{np.median(directional_total / diffuse_total):.2f}")
    print(f"illumination-vector altitude : {altitude.min():.1f} to "
          f"{altitude.max():.1f} deg")

    out = pd.DataFrame(
        {"sample": np.arange(len(times)),
         "directional_irradiance_W_m2": directional_total,
         "diffuse_irradiance_W_m2": diffuse_total,
         "vector_altitude_deg": altitude,
         "vector_azimuth_deg": azimuth})
    out.to_csv("decomposition_summary.csv", index=False)
    print("wrote decomposition_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
