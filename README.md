# Tracer particle analysis tools
### Forming the basis of the research into: the evolution of the multi-phase Circumgalactic Medium (CGM) of low-redshift Milky Way-mass Galaxies
### Using cosmological-zoom in Auriga galaxy formation and evolution simulations using the massively-parallelised magneto-hydrodynamical simulation code Arepo
The code presented in this repository forms the basis upon which the software in my other PhD research repositories CGM_Cosmic-Rays and CGM_Hybrid-Refinement were developed. The analysis tools in this repository are used by the aforementioned other two repositories, but can be used standalone.

---

## Affiliations
*Author*: Andrew Tomos Hannington

*Affiliation*: Cardiff University

*Supervisor*: Dr Freeke van de Voort (*Email*: vandevoortF@cardiff.ac.uk)

> [!IMPORTANT]
> The code in this repository builds upon proprietary software from the Auriga Project (https://wwwmpa.mpa-garching.mpg.de/auriga/) collaboration that is not available to the general public (there is, however, a publicly available version of Arepo here: https://arepo-code.org/wp-content/userguide/index.html). As such, **the code of this repository cannot be run without membership of the Auriga Project collaboration and access to Arepo Snap Utils**. Thus, for all other persons, the software of this repository is for illustrative purposes only.

## The introduction in brief
Given a set of selection criteria, we want to track the evolution of tracer particles through our simulation as they follow the flow of gas.

> [!NOTE]
> Throughout my PhD I have found a metaphor that likens tracer particles to rubber ducks to be particularly helpful in aiding understanding. In this metaphor, tracer particles can be thought of as individually numbered rubber ducks that are floating downstream in the flow of gas in our simulations, much like rubber ducks on a stream. The individual numbers of each rubber duck do not change, and therefore we can follow each from beginning to end of their journey downstream without confusion between which duck is which.

## The introduction in full
The code presented in this repository form the basis of the analysis of multi-phase gas in the Circumgalactic Medium of Milky Way-mass galaxies, the first research project of my PhD. The key scientific question of this work was to explore:
> How does the evolution of multi-phase gas in the CGM of low-redshift (z < 0.15) Milky Way-mass galaxies vary with temperature, and with galactocentric radial distance from the central galaxy?

> [!NOTE]
> Throughout the documentation I may refer to tracking the evolution of gas, and to the gas cells within which the tracer particles reside. This is a simplification. In reality, _the tracer particles can be within gas cells or a particle of any type_ (particles are used to model dark matter, stars, stellar winds, and black holes).

To investigate this question in simulations performed with Arepo is particularly challenging owing to the moving mesh used to discretise space in Arepo. There are, of course, advantages and disadvantages of using the moving mesh method as implemented in Arepo, compared to other hydrodynamical simulation techniques such as Smoothed Particle Hydrodynamics (SPH), fixed grid, and adaptive mesh-refinement hydrodynamics codes. However, the gas cells in Arepo move with the flow of gas, and are merged, split and otherwise deform during the simulation. Therefore, due to the finite number of simulation outputs (created at time intervals that are significantly larger than the time steps used by Arepo to evolve the simulated gas and other simulation elements) without additional methods it is impossible to map the identifier of each gas cell from one simulation output to the next. 

Thus, to track the evolution of individual parcels of gas mass in Arepo we use additional Monte Carlo Tracer Particles that move with gas flow and have a constant, unique identifier throughout the entire simulation. To track the gas evolution using tracer particles, we use the unique, constant tracer ID ('trid') of each tracer particle, and the current ID of the parent gas cell ('prid') within which the tracer particle resides. The tracer particles themselves have no physical properties as they are purely a numerical tool that allows us to track the evolution of the simulated gas over time. As such, to obtain the physical properties of each indidividual parcel of gas we map between the parent ID of the gas cell the tracer particle is situated within and the cell ID ('id') of the gas cells in the main simulation data. 

## Problems/numerical challenges solved by the code in this repository 
Each gas cell within the simulation may contain zero, one, or more tracer particles at any simulation output. Thus, the mapping between tracer particles and gas cells is often many-to-one, rather than simply one-to-one. This represents the first problem that the code in this repository solves bin order to make use of these tracer particles:

> 1. How can we efficiently perform a many-to-one mapping that returns a copy of the gas cell data for every tracer particle within the cell, and that returns this data in a sensible, consistent ordering that matches the ordering of the tracer particle ID's?

The method used to solve 1. is implemented in `get_copy_of_cell_for_every_tracer()` in `Tracers_Subroutines.py`. The current version of this method make use of pandas to perform SQL-style dataframe joins.
> [!WARNING]
> The current method uses pandas, which may not be suited to the analysis of very large numbers of tracer particles (for my own research, the method was applied to 17 sets of ~560,000 tracer particles) or simulations with an extremely large number of gas cells. Whilst chunking the process into loops over smaller sets of tracer particles may help, alternative python packages may be needed in order to handle cases with large number of simulation cells.

There are different physical properties asssociated with the gas cells and each of the different particle types. For example, the gas cells each have densities associated with them, however it is meaningless to associate density to the point-mass approximations used for the particles representing stars, dark matter, and black holes. As such, the lengths of each physical property vary accordingly. Thus, the second problem solved by the code in this repository is:

> 2. How can we sort and store the simulation data such that the evolution tracked by each tracer particle can be analysed simply and quickly, even when a tracer particle is exchanged between gas and different particle types?

The method used to solve 2. is implemented in `pad_non_entries()` in `Tracers_Subroutines.py`. This method modifies the data such that the data from gas and star particles (the only particle type analysed in the original work using these tools) have the same shape. The gas data is followed by a NaN value for every star particle, and vice versa, such that both gas and star particle data arrays have a length `N == number of gas cells + number of star particles`.
> [!WARNING]
> The current method is only configured for gas cells and star particles. The tracking of additional particle types will require modification of the code as presented here. Please contact me should you need assistance with this.

> [!WARNING]
> The current method is only configured for gas cells and star particles. The load order of gas (type zero), and then star particles (type 4) must not be altered!

## Software outline
My analysis is split into two core functions in the main analysis as performed in `Tracers.py`.
1. Select our Tracers of interest in `tracer_selection_snap_analysis()`
2. Iterate over time, selecting only the tracers that were of interest in 1,
   and saving their data (usually in a hdf5 `.h5` file). This is performed in
   `snap_analysis()`

These two work almost identically, save for 1. is going from "Cells that have
the right temperature, radius etc, to the tracers within these cells" and
therefore uses `get_tracers_from_cells()`. 2., however, is now going from those
tracers we selected at our time and properties of interest to the cells they
are now within - therefore using `get_cells_from_tracers()`.

The postprocessing step uses a function called `flatten_wrt_time()` to convert the raw data obtained in the main analysis to a 'per tracer particle' format. The raw data is in the format where data is given seperatley for each simulation output and in terms of cells/particles rather than tracer particles. The final format of the data is in a format where the full evolution of each tracer particle is given by a single column with `NaN` entries for data not applicable to the current particle type or gas cell that contains the tracer particle. Similarly, all data will be `NaN` for a tracer particle that has left our region of interest.

> [!WARNING]
> The "original" format data is not time flattened and is given in terms of gas cells and star particles rather than per tracer particle. As such, data in this format cannot be used to track time evolution using tracer particles without further analysis equivalent to the `flatten_wrt_time()` function given above. Relevant user warnings are provided when data in this format is loaded. 

---

## Final notes
Wherever code is commented out in the versions presented here they have been left in for my own future reference, or for ease of debugging and development. For example, it saves significant time to be able to uncomment out a few dummy function calls needed to obtain the necessary data to debug sebsequent functions, rather than to write these dummy function calls from scratch each time (they are frequently unique, subtle modifications to similar function calls made elsewhere in the code, and remembering exactly which variant is needed at any given line of code is quite difficult). There are also entire functions and old versions commented out too. Again, these are for my ease of reference only.