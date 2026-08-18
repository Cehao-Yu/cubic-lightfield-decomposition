CUBIC LIGHT-FIELD DECOMPOSITION - SAMPLE CODE
================================================================================

Minimal, self-contained code for the core calculation behind

  Yu C, Pont SC, Pastilha R, Hurlbert A.
  Capturing all the light we can and cannot see. (submitted)

Given spectral irradiance measured on the six faces of a cube, it recovers the
two components of the first-order light field: the directional component (the
illumination vector) and the diffuse component (the symmetric or ambient term),
together with the direction the illumination vector points in.


FILES
-----

  lightfield.py            The decomposition itself. Notation and formulae
                           follow section 4.4 of the paper. Four functions:
                             axis_components()  per-axis vector and symmetric
                                                terms, per wavelength
                             decompose()        directional and diffuse spectra
                             vector_direction() altitude and azimuth of the
                                                illumination vector
                             rate_of_change()   per-second differences using
                                                true elapsed intervals
                           Requires numpy only.

  example_decompose.py     Worked example: reads one day of measurements in the
                           format distributed with the Newcastle light-field
                           dataset, decomposes every sample, and writes a
                           summary. Requires numpy and pandas.

  test_lightfield.py       Checks the decomposition against cases with a known
                           answer. Run: python test_lightfield.py


QUICK START
-----------

  python example_decompose.py cube_face_spectral_irradiance.csv

Or from your own code:

  import numpy as np
  from lightfield import decompose, vector_direction

  # E has shape (n_samples, 6, n_wavelengths), faces ordered
  # x+, x-, y+, y-, z+, z-
  directional, diffuse = decompose(E)
  altitude, azimuth = vector_direction(E)


THE CALCULATION
---------------

For each wavelength, from the six face irradiances:

  axis component of the illumination vector
      E(x) = E(x+) - E(x-)                          and likewise for y and z

  directional component
      |E(vector)| = sqrt( E(x)^2 + E(y)^2 + E(z)^2 )

  axis-wise symmetric term
      Et(x) = [ E(x+) + E(x-) - |E(x)| ] / 2        and likewise for y and z

  diffuse component
      Et = mean( Et(x), Et(y), Et(z) )

  direction of the illumination vector
      altitude = arctan( E(z) / sqrt(E(x)^2 + E(y)^2) )
      azimuth  = arctan2( E(x), E(y) )              clockwise from north

Both components are full spectra: every quantity above is computed wavelength by
wavelength. Note that Et(x) is algebraically identical to min(E(x+), E(x-)); the
code uses the explicit form so that it reads as written in the paper.


CONVENTIONS THAT MATTER
-----------------------

Axes. +x east, +y north, +z up. Only the three opposing face pairs enter the
magnitude calculations, so |E(vector)| and Et are unaffected by how the
horizontal axes are labelled - but the azimuth is not. Keep the pairing and the
labelling consistent.

Timestamps. Where the six faces of a sample are acquired sequentially rather
than simultaneously, each face carries its own acquisition time. Timestamp each
light-field sample by the mean of its six acquisition times, and compute
sampling intervals and rates of change from those timestamps rather than from a
nominal constant interval - integration times are lengthened at low light
levels, so intervals are not uniform. rate_of_change() enforces this by taking
the timestamps as an argument.

Units. Spectral irradiance in W m-2 nm-1 throughout. Integrating over wavelength
gives irradiance in W m-2; weighting by V(l) and multiplying by 683 lm W-1 gives
illuminance in lx; weighting by the CIE S 026 action spectra gives the alpha-opic
quantities.


LICENCE
-------

Creative Commons Attribution 4.0 International (CC BY 4.0).
