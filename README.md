# Exoplanet hunting with deep learning

***The Search for New Earths***
-------------------------

The data describe the change in flux (light intensity) of several thousand stars. Each star has a binary label of `2` or `1`. 2 indicated that that the star is confirmed to have at least one exoplanet in orbit; some observations are in fact multi-planet systems.

As you can imagine, planets themselves do not emit light, but the stars that they orbit do. If said star is watched over several months or years, there may be a regular 'dimming' of the flux (the light intensity). This is evidence that there may be an orbiting body around the star; such a star could be considered to be a 'candidate' system. Further study of our candidate system, for example by a satellite that captures light at a different wavelength, could solidify the belief that the candidate can in fact be 'confirmed'.



In the above diagram, a star is orbited by a blue planet. At t = 1, the starlight intensity drops because it is partially obscured by the planet, given our position. The starlight rises back to its original value at t = 2. The graph in each box shows the measured flux (light intensity) at each time interval.

----------

# Description

Trainset:

 * 5087 rows or observations.
 * 3198 columns or features.
 * Column 1 is the label vector. Columns 2 - 3198 are the flux values over time.
 * **37** confirmed exoplanet-stars and 5050 non-exoplanet-stars.

Testset:

 * 570 rows or observations.
 * 3198 columns or features.
 * Column 1 is the label vector. Columns 2 - 3198 are the flux values over time.
 * **5** confirmed exoplanet-stars and 565 non-exoplanet-stars.
