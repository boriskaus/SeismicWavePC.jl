# Models

## The model

A model is a GMG `CartData` (coordinates in km, z up; constant spacing in x and y, see below for z) with the fields
- `rho` [kg/m³] (or g/cm³ with `density_unit = :g_cm3`, as in OpenSWPC),
- `vp`, `vs` [km/s],
- optionally `Qp`, `Qs` (without them the model is elastic). Q is constant between `fq_min` and `fq_max`, and the velocities
  apply at `fq_ref` (keywords of `simulate`; defaults 0.05, 5 and 1 Hz),
- optionally `epsilon`, `delta`, `gamma`: VTI anisotropy, and on curvilinear grids `tilt`, `azimuth` [°]: a tilted
  symmetry axis (TTI; see below).

Cells with `vp = vs = 0` are air: the free surface and topography follow from them. Cells with `vs = 0` and `vp > 0` are
water (oceans, lakes). A `CartData` of size `(nx, 1, nz)` is a 2D model (P-SV by default, `wave = :sh` for SH waves).
Any GMG tool can be used to build the model: layers, volcanoes, spheres, slabs, geographic data and projections.

Topography is a staircase of cells. As in OpenSWPC, the shear modulus at a shear-stress point is the harmonic mean of the
four cells around it, so it is zero next to air, also at the inner corner at the foot of every step. A top cell one cell wide
in x (or y) then has no shear coupling to the ground in that direction and does not move in it. With
`simulate(...; free_surface = :corners)` these inner corners (one cell of the four with μ = 0) take the harmonic mean of the
three solid cells; flat surfaces and outer corners are unchanged. In 2D tests on stair-stepped slopes of 10–30° against
the exact flat surface, it makes the surface seismograms 2–4 times more accurate (misfit 17 instead of 34–52 % at 20 cells
per S wavelength, 3–4 instead of 6–12 % at 80). See [Validation](@ref) for a benchmark with tilted planes and a hill.

## Topography on a curvilinear grid

The staircase of cells is accurate away from the surface, but close to a sloping surface its errors reach 10–30 % at
16 points per S wavelength (see [Validation](@ref)). A 2D or 3D model can instead be a *deformed* `CartData`: a terrain-following
mesh whose top row of nodes is the topography. It is then simulated on a curvilinear grid with summation-by-parts (SBP)
finite differences: the free surface lies exactly on the topography, and the scheme is provably stable (the discrete
energy is conserved, or dissipated at the absorbing boundaries). The SBP operators are the boundary-optimized dual-pairing
operators of Mattsson, Niemelä & Winters (2027, *J. Comput. Phys.* 568, 115399; order 7), from [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
(Ranocha, 2021, *J. Open Source Softw.* 6(64), 3454). The method and its relation to published SBP-SAT schemes for
seismic waves: [Curvilinear solver: method and related work](@ref).

```julia
topo(x) = 3.0 * exp(-x^2 / 50)                                   # [km]
moho(x) = -7.0 + 1.5 * sin(2π * x / 40)
mesh = curvilinear_mesh(range(-72, 52, length=249), [topo, moho]; dz=0.5, bottom=-80)
X, Z = mesh.x.val, mesh.z.val
Phase = ifelse.(Z .< moho.(X), 2, 1)                              # layers
add_ellipsoid!(Phase, zeros(size(Phase)), mesh; cen=(0, 0, -3), axes=(3, 3, 1.5), phase=ConstantPhase(3))   # a magma chamber
vp_, vs_, rho_ = (5.5, 6.5, 3.8), (3.2, 3.75, 2.0), (2600.0, 2900.0, 2400.0)
model = addfield(mesh, (; rho=getindex.(Ref(rho_), Phase), vp=getindex.(Ref(vp_), Phase), vs=getindex.(Ref(vs_), Phase)))
result = simulate(model; tmax=20.0, sources, stations)           # dt from the stability limit
```

- **The mesh.** [`curvilinear_mesh`](@ref) makes the mesh from surfaces (functions or surface `CartData`): a grid line
  follows the topography and every interface with a large jump in the material (e.g. a Moho). Columns are vertical. Fill
  the material at the nodes with the usual GMG tools. GMG ≥ 0.10.2 makes the same mesh, with all its model tools working
  on it (`add_layer!`, `add_box!`, `add_ellipsoid!`, interpolation of data onto it):
  `curvilinear_grid(sbp_nodes(-72, 52, 249), 0, (topo, moho, -80.0); dz=0.5, spacing=:sbp)`.
- **Blocks.** The mesh is split into blocks at grid rows where the material jumps along the full width. Their interfaces
  are grid lines, so the jump stays sharp. Other heterogeneity (a magma chamber, gradients) is sampled at the nodes.
  Give the nodes on an interface the material above it (`z >= interface`): where the node spacing does not change at the
  interface (e.g. a flat mesh), the interface is taken at the upper row of the jump, so the material below would shift it
  up by one row (a first-order error).
- **Oceans** (fluids, `vs = 0`): a fluid layer is a block of its own, between two surfaces of the mesh (e.g. sea level
  and the bathymetry), and must span the whole width (no coastlines). At the seafloor the normal velocity and the
  normal traction are continuous and the tangential traction vanishes: the fluid slips along the solid (SAT terms on the
  normal components only), stable as all interfaces. Against the staggered grid with a flat seafloor (exact there) below 1 %
  at 100 m; with a sloping seafloor the staggered grid converges towards it. SH waves see the seafloor as a free surface.
  Strains at stations in the fluid: the volumetric strain, split equally between the normal components.
- **Time step and boundaries.** The time step follows from the spectral radius of the operator: `dt` is not needed, give
  the duration `tmax`. The sides and the bottom absorb with impedance boundary conditions; `absorber = :pml` adds a
  perfectly matched layer of `na = 20` nodes (at least 6; the CFS-PML of OpenSWPC on the ξ and η parts of the SBP operators: about
  20 times fewer reflections, ~1.5× the cost per step with layers on all sides), `:mpml` the multiaxial PML (as on the
  staggered grid, `mpml_ratio = 0.1`: stable under soft surface layers, where the PML can grow exponentially; 2–3 times
  the reflections of the PML), `:cerjan` a sponge.
- **Stations** with `surface=true` record exactly on the topography. Attenuation, moment tensors and forces work as on
  the staggered grid.
- **3D:** `curvilinear_mesh(x, y, surfaces; dz, bottom)` with surfaces z(x, y) (functions or surface `CartData`); columns
  are vertical in 3D too.
- **MPI and GPUs** as on the staggered grid: `nproc` processes along x (2D) or `(npx, npy)` (3D); the columns are not
  split. Results are identical to a run on one process.
- **Gradients and Born modelling:** [`misfit_gradient`](@ref) (2D and 3D, attenuation, MPI, GPUs) uses the exact
  discrete adjoint (`checkpoint_steps` as on the staggered grid); the gradient is per model node. Source gradients
  (location, moment tensor, force, timing, source time function) as on the staggered grid. [`born`](@ref) works as on the staggered
  grid (also with attenuation, on GPUs and with MPI).
- **Anisotropy (VTI)** as on the staggered grid (fields `epsilon`, `delta`, `gamma`; the symmetry axis stays vertical on the
  deformed mesh), also in gradients and Born modelling. **Tilted axes (TTI)** with the fields `tilt` and `azimuth` (see
  [Tilted symmetry axis (TTI)](@ref); forward simulations only).
- **Snapshots:** ParaView files and callbacks of the whole wavefield, and netCDF planes ([`Snapshots`](@ref): the free
  surface `fs` on the topography, `xy` planes and sections resampled at regular heights; `:v`, `:u`, `:ps`, also in pieces).
- **Checkpoints** (`checkpoint`, `checkpoint_every`, `restart`) as on the staggered grid.
- **SH waves** in 2D (`wave = :sh`), with everything above (attenuation, VTI with γ, PML, gradients, Born, MPI, GPUs).
- **Teleseismic waves** ([`TeleseismicWave`](@ref)) as on the staggered grid: total-field / scattered-field injection
  along a box (the box reaches up to the topography; the model must be horizontally layered and elastic along its sides
  and bottom, where the topography must be flat), with the exact gradient. `PlaneWave` (OpenSWPC's initial plane wave)
  is not available on curvilinear grids: use `TeleseismicWave`.

![A hill, a Moho-like interface and a magma chamber on a curvilinear grid](assets/2D_curvilinear.png)

*[`examples/2D_curvilinear.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/2D_curvilinear.jl): the
model on its deformed mesh and the vertical velocity at stations on the topography.*
- **Cost.** A step costs about 1.3–2.5× one on the staggered grid with as many cells (2D and 3D, CPU and GPU). Near topography, however, the
  curvilinear grid reaches the same accuracy with about 8× coarser spacing (in 2D, 64× fewer cells, 8× fewer steps).

## Anisotropy (VTI)

With any of the fields `epsilon`, `delta`, `gamma` (Thomsen parameters; missing ones are 0) the model is transversely
isotropic with a vertical symmetry axis (VTI: layered sediments, horizontally aligned cracks or melt lenses). `vp` and `vs`
are then the vertical velocities:

| | |
|---|---|
| qP | `vp` vertically, `vp √(1 + 2ε)` horizontally; `δ` shapes the wavefront near the vertical |
| SH | `vs` vertically, `vs √(1 + 2γ)` horizontally |
| qSV | `vs` vertically and horizontally, its anisotropy follows from ε − δ |

```julia
model = CartData(X, Y, Z, (; rho, vp, vs, epsilon=fill(0.1, size(X)), delta=fill(0.05, size(X)), gamma=fill(0.08, size(X))))
```

- The symmetry axis is the z axis of the grid: every stress needs only the strain rates at its own point, as in isotropic
  media, so the accuracy and the stability (time step: the fastest qP direction) are those of the isotropic code. Tilted
  axes (TTI): curvilinear grids only, see below.
- **Attenuation:** the normal stiffness is split as in the isotropic scheme into a P part (`Qp`) and the rest (`Qs`), and
  stays symmetric: qP waves along the axes attenuate with `Qp`, S waves with `Qs`; isotropic media give exactly the
  isotropic scheme.
- Everything else works as for isotropic models: free surface and topography, oceans, PML, sources and stations between
  grid points, GPUs, MPI (also a [`DistributedModel`](@ref) created with `anisotropy = true`), gradients (`misfit_gradient` also returns `epsilon`, `delta`, `gamma`; exact, as checked against
  automatic differentiation) and Born modelling. Not yet: plane waves and teleseismic waves (their incoming field assumes
  isotropic layers).
- Models without these fields run the isotropic kernels: anisotropy costs nothing when it is not used.

## Tilted symmetry axis (TTI)

On curvilinear grids (`grid = :curvilinear`, also for a flat surface) the symmetry axis may tilt, node by node: dipping
layers, foliated or fractured rock, aligned cracks in a volcano. The fields
- `tilt` [°]: the angle θ of the axis from the vertical,
- `azimuth` [°] (3D): the direction φ of its horizontal part, from x towards y (counter-clockwise from x seen from above),

give the axis n = (sin θ cos φ, sin θ sin φ, cos θ); in 2D n = (sin θ, cos θ) in the x-z plane (a signed `tilt`, positive
towards +x, no `azimuth`). `vp`, `vs` and the Thomsen parameters then hold along and across this axis instead of the
vertical: qP travels with `vp` along n and with `vp √(1 + 2ε)` perpendicular to it.

```julia
# 2D: layers dipping by 30° towards +x (axis normal to the layers)
model = addfield(mesh, (; rho, vp, vs, epsilon=fill(0.2, size(X)), delta=fill(0.05, size(X)), gamma=fill(0.1, size(X)),
                        tilt=fill(30.0, size(X))))
```

- **The scheme.** The grid is collocated (all strain rates at every node), so the stress update rotates the strain rate
  into the frame of the axis, applies the VTI update and rotates the stress back: σ = Aᵀ C'(A ε Aᵀ) A. The stiffness stays
  symmetric and positive definite, so the free surface, interfaces and the energy estimate of the scheme hold as for
  VTI. The grid stores one or two angles per node instead of 21 moduli; a 3D step costs ~10 % more than VTI (Metal).
- **Attenuation:** the split into a `Qp` part and a `Qs` part rotates with the axis (qP along the axis attenuates with
  `Qp`); the memory variables live in the frame of the axis.
- **Checks** (`test_curvilinear.jl`): `tilt = 0` gives the VTI result bit for bit; an axis along x (or y) gives the VTI
  result with the axes swapped (2D P-SV, SH and 3D, to rounding); for tilt 30° (2D) and tilt 40°, azimuth 60° (3D) the
  rotated VTI solution is reproduced to the discretization error, which falls with the grid spacing as for VTI (2D: 1.6 %
  → 0.3 % when the source period grows by 1.5×; 3D: 5 % → 1.4 % from 0.2 to 0.13 km). Long runs with topography, an
  interface, attenuation and an axis varying in space decay (impedance absorber and PML).
- **Not yet:** the staggered solver (its normal and shear stresses live at different points: TTI would need
  interpolation), gradients and Born modelling, teleseismic waves. The impedance absorber uses the speeds of the untilted
  axis (it absorbs a little less); CFS-PMLs can become unstable in strongly anisotropic media (a known property of
  PMLs), so check long runs or use `absorber = :impedance`.

## Finer cells near the surface

The spacing in z may vary: give the `CartData` the depths you want, for example fine cells in slow near-surface layers
(sediments, volcanic deposits) and coarser ones at depth. The solver then works on a grid stretched in depth (the depth
derivatives are taken along the cell index and divided by the local spacing, with 4th-order consistent factors), so that
the waves see the same physics with far fewer cells. The spacing must change smoothly, by at most 25 % from cell to cell (a
few percent is best). The spacing in x and y stays constant.

```julia
depths, h = [0.0025], 0.005                         # cell centres [km]: 5 m cells down to 250 m,
while depths[end] < 3.0                              # then 3 % larger per cell up to 40 m
    hn = depths[end] < 0.25 ? 0.005 : min(1.03h, 0.04)
    push!(depths, depths[end] + (h + hn) / 2)
    h = hn
end
z = [reverse(-depths); 0.0025; 0.0075]              # z up, with 2 cells of air on top
X, Y, Z = xyz_grid(range(0, 4, step=0.005), 0, z)
```

Everything else works as on a uniform grid: sources and stations (also between grid points), the PML (its profile follows
the actual depths), MPI, GPUs, and the exact adjoint gradients. The time step is limited by the smallest cells, as before:
dt < 6/7 / (vp √(1/dx² + 1/dz²)) layer by layer, so fine cells in slow rock cost little extra. Plane waves need a uniform
grid.

Example: a 250 m thick layer with vs = 0.4 km/s over faster rock (4 Hz Ricker, P-SV, stations 10 m deep), compared with a
uniform 5 m grid:

| Grid in depth | Cells in depth | Time | Difference (vx, vz) |
|---|---:|---:|---:|
| uniform 5 m (reference) | 602 | 12.0 s | — |
| stretched, 5 m to 250 m, then +3 % per cell to 40 m | 163 | 1.7 s | 1.2 %, 1.8 % |
| uniform 10 m | 302 | 1.7 s | 10 %, 5.3 % |
| uniform 20 m | 152 | 1.1 s | 67 %, 29 % |

## Models built per MPI process

A `CartData` holds the whole model, and in a run on several MPI processes every process sets up the whole medium before it
keeps the part of its subdomain. A model that does not fit into the memory of one process is given as a
[`DistributedModel`](@ref) instead: the coordinates of the whole grid and a function that builds the model of a range of
columns. Every process calls it once, for its own subdomain:

```julia
x, y, z = range(0, 100, 2001), range(0, 80, 1601), range(-30, 2, 641)     # cell centres [km], z up
model = DistributedModel(x, y, z) do ix, iy
    X, Y, Z = xyz_grid(x[ix], y[iy], z)                                   # this part of the grid: use any GMG tool on it
    air = Z .> 0
    (rho=ifelse.(air, 1.0, 2700.0), vp=ifelse.(air, 0.0, 5.0), vs=ifelse.(air, 0.0, 2.9), Qp=fill(200.0, size(X)), Qs=fill(100.0, size(X)))
end
result = simulate(model; nproc=(2, 2), dt, nt, sources, stations)
```

The function returns a `NamedTuple` of arrays of size `(length(ix), length(iy), length(z))` or a `CartData`, with the fields of
a `CartData` model (with `DistributedModel(...; anisotropy = true)` also `epsilon`, `delta`, `gamma`). The results are those of the same model given as a `CartData`. The memory a process needs for the setup
then grows with its subdomain, not with the model. Moment tensor and force sources at their cells (`interpolation = :cell`),
stations, snapshots, checkpoints, GPUs and stretched grids work as usual; plane and teleseismic waves, sources between grid
points, misfit gradients and Born modelling need a `CartData`. The wavefield at the end is not collected (`wavefield = true` does).

A curvilinear model is built per process in the same way, column by column: `DistributedModel(f, x, y, nz)` with the SBP
nodes `x`, `y` of the columns and `nz` nodes per column; `f(ix, iy)` returns the deformed grid of these columns with its
material, for example from GMG with a fixed number of cells per layer (so that all pieces have the same rows):

```julia
xs, ys = sbp_nodes(-100, 100, 2001), sbp_nodes(-80, 80, 1601)
model = DistributedModel(xs, ys, 1 + 60 + 140) do ix, iy
    g = curvilinear_grid(xs[ix], ys[iy], (topo, moho, -60.0), (60, 140); spacing=:sbp)
    low = g.z.val .< moho.(g.x.val, g.y.val)
    addfield(g, (; rho=ifelse.(low, 3300.0, 2700.0), vp=ifelse.(low, 8.0, 6.0), vs=ifelse.(low, 4.5, 3.5)))
end
result = simulate(model; nproc=(4, 4), tmax, sources, stations, absorber=:pml)
```
Every process then holds the mesh and the material of its own columns only (and a few more around them).

## Geographic coordinates

Models built from geographic data with a GMG `ProjectionPoint` keep their projection: sources and stations can be given in
longitude and latitude with the same `ProjectionPoint` ([Sources and stations](@ref)).

## Models from thermomechanical and petrological models

The temperature of a [MagmaThermoKinematics.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl) model, or any other
temperature field, can be turned into density, seismic velocities and melt fraction with
[MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin_C.jl) phase diagrams, including the reduction of the
velocities by partial melt and a low Q in partially molten rock. See the Unzen examples in [Examples](examples.md).

## Random media

```julia
xi = random_medium(x, y, z; spectrum=:vonkarman, a=0.5, epsilon=0.05, kappa=0.3)
model = perturb(model, xi)
```
makes fluctuations with a Gaussian, exponential or von Kármán spectrum (anisotropic with `a=(ax, ay, az)`) and applies them as
OpenSWPC does (vp, vs × (1 + ξ), ρ × (1 + 0.8 ξ) in solid rock). `read_random_medium(file)` reads OpenSWPC's random media.

## Layered models and OpenSWPC files

`read_lhm`, `read_lgm` and `read_grd` build models from layered descriptions (with OpenSWPC's options `vcut`, `munk` and
`earth_flattening`), and `read_netcdf` reads OpenSWPC's netCDF velocity models; see [Coming from OpenSWPC](@ref).
`flatten_earth(model)` applies the Earth-flattening transformation to any model.
