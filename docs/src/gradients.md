# Misfit gradients

```julia
using SeismicWavePC, GeophysicalModelGenerator
observed = simulate(true_model; dt, nt, sources=source, stations)      # or recorded data, as Seismograms
J, grad, sgrad = misfit_gradient(start_model, observed; dt, nt, sources=source)
grad.fields.vs                                                          # ∂J/∂vs on the grid of the model (per km/s)
sgrad[1].M, sgrad[1].t0                                                 # ∂J/∂(moment tensor), ∂J/∂(origin time)
```

- **Misfit:** `misfit=L2Misfit()` (default) is `J = ½ Σ (v - v_observed)²` over all samples, components and stations of
  `observed`, whose stations are used.
  - `L2Misfit(weights=w, window=(t1, t2))` weights stations, components or samples and selects time windows (one for all, or
    one per station); `L2Misfit(invcov=C⁻¹)` uses an inverse covariance matrix of the samples of a trace (correlated noise);
    `L2Misfit(normalize=true)` divides the misfit of each trace by the energy of the observed trace.
  - `TravelTimeMisfit(; weights, window)` uses the cross-correlation time shifts of the traces.
  - `EnvelopeMisfit()` compares the envelopes (insensitive to the phase, robust against cycle skipping), `PhaseMisfit()` the
    instantaneous phases (insensitive to amplitudes), `TFPhaseMisfit(width=T)` the phase differences in time and frequency
    (Fichtner et al., 2008).
  - `Filtered(misfit, (f1, f2))` applies a misfit to band-pass filtered synthetic and observed seismograms (zero-phase
    Butterworth, [`bandpass`](@ref)), for inversions from low to high frequencies.
  - [`geographic_weights`](@ref)`(stations, L)` gives station weights that balance clusters of stations.
  - Any function `(syn, obs, t) -> (J, ∂J/∂syn)` of the synthetic and observed velocities `[it, component, station]` works too.
- **Several shots:** `misfit_gradient(model, [obs1, obs2, …]; sources=[src1, src2, …], …)` sums the misfits and model
  gradients of the shots; each shot can have several sources. `misfit` is one misfit for all shots, or a vector with one per
  shot (e.g. event weights).
- **Model gradient:** `grad` is a `CartData` with the fields `rho`, `vp`, `vs`, `Qp`, `Qs`. With
  `mute=(sources=r, stations=r)` it is tapered to zero within the distance `r` [km] of the sources and stations, where it is
  large and of little use.
- **Teleseismic waves:** with a `TeleseismicWave` in `sources` the gradient is exact as well (the injection along the box acts
  as sources that do not depend on the model inside it; its dependence on the material at the box boundary is included).
- **Illumination:** with `illumination=true`, `grad` also has the field `illumination`: Σₜ |v|² of the source wavefield
  [(m/s)²] per cell (summed over the shots, not muted). Dividing the gradient by it (clipped near the sources, plus a small
  constant) is a simple preconditioner (the diagonal of a pseudo-Hessian) for inversions and compensates the illumination in
  reverse-time migration ([Examples](examples.md)).
- **Source gradients:** `sgrad` has one entry per source (per shot: a vector):
  - moment tensors: `M` (∂J/∂ of the six components `M0 .* M`) and `M0`;
  - forces: `F` (∂J/∂ of `fx, fy, fz`);
  - both: the location `x, y, z`, `t0`, `trise`, and `stf`, the gradient with respect to the source time function at the
    times `t`. With a sampled source time function (`stf = SampledSTF(t, values)`), the gradient with respect to its samples
    follows from `stf` by the chain rule of the linear interpolation.

  With `interpolation = :sinc` (sources between grid points, see [Sources and stations](@ref)) all source gradients are exact,
  including the location. With the default `interpolation = :cell` the source sits at a cell, and the location gradient
  comes from the spatial derivative of the adjoint field (an approximation).
- All keywords of `simulate` work: GPUs (`backend`), MPI (`nproc`; every process gets all results), `T`, `absorber`,
  `record_every`, `stf`, ….
- **Method:** adjoint-state method with checkpointing. The forward wavefield is saved every few steps (balancing host and device
  memory) and the steps in between are recomputed; the adjoint simulation runs backwards.
- **Exactness:** the adjoint kernels are the exact transposes of the forward kernels. They include the 2nd-order cells at free
  surfaces and the PML or M-PML (with adjoint PML memory variables), and adjoint memory variables of attenuation give the `Qp`, `Qs`
  gradients. The result agrees with reverse-mode automatic differentiation to round-off (10⁻¹⁵ with `T = Float64`) in 2D and 3D.
  It is approximate only for the Cerjan sponge, the side extrapolation of plane waves, and the density in the source cell of
  P-SV body forces.
- **Cost:** about 5 forward simulations. On Marmousi2 with 641 receivers a gradient takes 80 s on 12 CPU threads and 30 s on
  an Apple GPU, 5× and 13× faster than [SeismicWaves.jl](https://github.com/GinvLab/SeismicWaves.jl) ([Performance](@ref)).
- `method=:ad` (with `using Enzyme`) differentiates the whole simulation automatically instead: exact in all cases, but 20–50
  forward simulations, on the CPU and one process, with the default misfit and without source gradients.


See [`misfit_gradient`](@ref) and the misfits ([`L2Misfit`](@ref), [`TravelTimeMisfit`](@ref), [`EnvelopeMisfit`](@ref),
[`PhaseMisfit`](@ref), [`TFPhaseMisfit`](@ref), [`Filtered`](@ref)) for all options.

## Born modelling (the Jacobian)

[`born`](@ref) computes the seismograms of a model perturbation `δm` to first order, J δm (J: the Jacobian of the
seismograms with respect to the model), together with the seismograms of the model itself, in one simulation:

```julia
δm = CartData(model.x.val, model.y.val, model.z.val, (; vp=0.1 .* blob))   # any of rho, vp, vs, Qp, Qs
background, scattered = born(model, δm; dt, nt, sources, stations)
```

- **Exactness:** it is the exact linearization of the discrete simulation (free surface, PML, attenuation), and the
  transpose of the gradient of `misfit_gradient`: ⟨J δm, r⟩ = ⟨δm, Jᵀ r⟩ to rounding errors (10⁻¹⁵ with `T = Float64`),
  for P-SV, SH and 3D, moment tensors and forces, at cells or between them.
- `δm` may also perturb the Thomsen parameters `epsilon`, `delta`, `gamma` of VTI models, and `tilt`, `azimuth` of TTI
  models (curvilinear grids).
- **Use:** with `misfit_gradient` (Jᵀ r) it gives Gauss-Newton Hessian-vector products JᵀJ δm (Newton-type inversions,
  uncertainties of the Laplace approximation) and least-squares migration; alone, it shows how a model change shows up in
  the data.
- **How:** the simulation runs with numbers that carry a tangent (forward-mode differentiation through the kernels): the
  medium holds the perturbation, and the tangent of the wavefield is the scattered wavefield.
- **Cost:** about 2.5 forward simulations on GPUs, 4.5 on CPU threads. All keywords of `simulate` work, including MPI
  (`nproc`), except teleseismic waves (for now).
