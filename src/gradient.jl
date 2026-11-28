# Misfit gradients: the adjoint-state method (adjoint.jl) or automatic differentiation with Enzyme (SeismicWavePCEnzymeExt);
# here the documentation and the chain rule from the internal moduli back to the fields of the model.

"""
    J, gradient, source_gradients = misfit_gradient(model, observed::Seismograms; sources, misfit=L2Misfit(), kwargs...)
    J, gradient, source_gradients = misfit_gradient(model, observed::Vector{Seismograms}; sources::Vector, kwargs...)

Misfit `J` between the velocity seismograms of a simulation and `observed` (whose stations are used), and its gradient with
respect to the model and the sources. With a vector of `observed` seismograms and a vector of `sources` (one entry per shot:
a source or a vector of sources) the misfits and model gradients of all shots are summed.

- `misfit`: [`L2Misfit`](@ref) (default: `J = ½ Σ (v - v_observed)²`, optionally with weights, time windows, an inverse
  covariance matrix and normalization per trace), [`TravelTimeMisfit`](@ref) (cross-correlation time shifts),
  [`EnvelopeMisfit`](@ref), [`PhaseMisfit`](@ref) (instantaneous phase), [`TFPhaseMisfit`](@ref) (time-frequency phase),
  any of them on band-pass filtered seismograms with [`Filtered`](@ref), or any function `(syn, obs, t) -> (J, ∂J/∂syn)`
  of the synthetic and observed velocities `[it, component, station]` [m/s] and the sample times `t` [s]. With several
  shots: one misfit for all, or a vector with one per shot (event weights, e.g. from [`geographic_weights`](@ref)).
- `mute = (sources=r, stations=r)`: taper the model gradient to 0 at the sources and stations within the distance `r` [km]
  (the gradient is large and of little use next to them).
- `illumination = true`: also return the illumination of the model by the sources, Σₜ |v|² of the forward wavefield
  [(m/s)²] per cell, as the field `illumination` of `gradient` (summed over shots). Dividing the gradient by it is a simple
  preconditioner (diagonal pseudo-Hessian) for inversions, and gives illumination-compensated images in reverse-time
  migration (see the Marmousi2 RTM example).
- `kwargs` are those of [`simulate`](@ref) (`dt`, `nt`, `stf`, `record_every`, `absorber`, `T`, `backend`, `nproc`, …) and
  must describe the simulation that `observed` is compared with.
- `gradient`: a `CartData` on the grid of `model` with the fields `rho`, `vp`, `vs`, `Qp`, `Qs` (VTI models: also
  `epsilon`, `delta`, `gamma`; TTI models: also `tilt` (and `azimuth` in 3D), per degree): ∂J/∂ of the model field in its units (per kg/m³ (or g/cm³ with `density_unit = :g_cm3`), per km/s, per unit Q). Cells in the absorbing layers are
  homogenized in the simulation, so their sensitivity is added to the cells next to the layers.
- `source_gradients`: one entry per source (per shot: a vector of them), `nothing` for plane waves and ignored sources:
  - moment tensors: `M` (∂J/∂ of the six components `M0 .* M` [per N m]), `M0` (at fixed `M`);
  - forces: `F` (∂J/∂ of `fx, fy, fz` [per N]);
  - both: `x, y, z` (location [per km]; exact with `interpolation = :sinc`, otherwise from the spatial derivative of the
    adjoint field, as the source is injected at a single cell), `t0`, `trise` [per s], and `stf`: ∂J/∂ of the moment rate
    (source time function) at the times `t`.

`method`
- `:adjoint` (default): adjoint-state method with adjoint kernels (transposed stencils, including the free-surface cells
  and the PML with adjoint PML memory variables, and adjoint memory variables of attenuation), on all backends (CPU
  threads, GPUs); about 5 forward simulations. It is the exact gradient of the discrete simulation (to rounding errors),
  except for the Cerjan sponge, the side extrapolation of plane waves and the density in the source cell of P-SV forces.
- `:ad`: reverse-mode automatic differentiation of the whole simulation (needs `using Enzyme`): exact in all cases, but
  ~20-50 forward simulations, CPU and one process only, the default misfit only and no source gradients. Use
  `T = Float64` with it.

The wavefields are saved every `checkpoint_steps` steps (default: balances the memory of the checkpoints, kept in host
memory, and of the velocity fields of the steps in between, kept on the device), and the steps in between are recomputed in
the reverse pass. With `nproc` (MPI, as in `simulate`) every process gets all results. Plane waves enter as a fixed initial
wavefield (their dependence on the medium is not differentiated).
"""
function misfit_gradient(model::CartData, observed::Seismograms; method=:adjoint, checkpoint_steps=nothing, misfit=L2Misfit(),
                         mute=nothing, illumination=false, kwargs...)
    bad = intersect(keys(kwargs), (:stations, :snapshots, :callback, :checkpoint, :restart, :driver))
    isempty(bad) || error("misfit_gradient does not take the keywords $(join(bad, ", "))")
    method in (:adjoint, :ad) || error("method must be :adjoint or :ad, got :$method")
    K = checkpoint_steps === nothing ? nothing : Int(checkpoint_steps)
    illumination && method === :ad && error("illumination = true needs method = :adjoint")
    if is_curvilinear(model) || get(kwargs, :grid, :auto) === :curvilinear       # curvilinear grids: the SBP adjoint
        method === :adjoint || error("curvilinear grids: method = :adjoint")
        res = simulate(model; kwargs..., grid=:curvilinear, stations=observed.stations, verbose=false,
                       driver=CurvilinearGradient(observed, misfit, K, illumination))
        mute === nothing || mute_gradient!(res[2], mute, kwargs[:sources], observed.stations)
        return res
    end
    driver = method === :adjoint ? AdjointDriver(observed, K, misfit, illumination) : ad_driver(observed, K, misfit)
    res = simulate(model; kwargs..., stations=observed.stations, verbose=false, driver)
    mute === nothing || mute_gradient!(res[2], mute, kwargs[:sources], observed.stations)
    return res
end

"""
Multiply the model gradient `g` by a taper that is 0 at the sources and stations and rises as (1 - cos(π d/r))/2 to 1 at the
distance `r` [km] (`mute = (sources=r, stations=r)`; either may be left out).
"""
function mute_gradient!(g::CartData, mute, sources, stations)
    keys(mute) ⊆ (:sources, :stations) || error("mute = (sources=r, stations=r) [km], got $mute")
    srcs = [s for s in (sources isa AbstractVector ? sources : [sources]) if s isa Union{MomentTensorSource,ForceSource}]
    points = [[(s.x, s.y, s.z, mute.sources) for s in srcs if haskey(mute, :sources)];
              [(s.x, s.y, s.z, mute.stations) for s in stations if haskey(mute, :stations)]]
    is3d = size(g)[2] > 1
    X, Y, Z = g.x.val, g.y.val, g.z.val
    taper = ones(size(X))
    for (px, py, pz, r) in points, I in eachindex(X)
        d = sqrt((X[I] - px)^2 + (is3d ? (Y[I] - py)^2 : 0.0) + (Z[I] - pz)^2)
        d < r && (taper[I] *= (1 - cos(π * d / r)) / 2)
    end
    foreach(f -> g.fields[f] .*= taper, filter(!=(:illumination), keys(g.fields)))
    return g
end

function misfit_gradient(model::CartData, observed::AbstractVector{Seismograms}; sources::AbstractVector, misfit=L2Misfit(), kwargs...)
    length(sources) == length(observed) || error("one entry of `sources` per shot: $(length(observed)) shots, $(length(sources)) sources")
    shots = [misfit_gradient(model, o; kwargs..., sources=src, misfit=mi) for (o, src, mi) in zip(observed, sources, shot_misfits(misfit, length(observed)))]
    g = first(shots)[2]
    fields = map(keys(g.fields)) do f
        sum(r -> r[2].fields[f], shots)
    end
    return sum(first, shots), CartData(g.x.val, g.y.val, g.z.val, NamedTuple{keys(g.fields)}(fields)), [r[3] for r in shots]
end

"The misfits of `n` shots: one for all, or a vector with one per shot (e.g. with event weights)."
shot_misfits(misfit, n) = fill(misfit, n)
shot_misfits(misfit::AbstractVector, n) = (length(misfit) == n || error("misfit: one per shot ($n), got $(length(misfit))"); misfit)

"Driver of `simulate` for `method = :ad` (Enzyme extension)."
ad_driver(args...) = error("misfit_gradient(...; method = :ad) needs the optional package Enzyme: add it and load it with `using Enzyme`")

"d chi / d tau of `visco_chi(ts, tau, f)`."
function visco_dchi(ts, tau, f)
    ω = 2π * f
    cc = sum(t -> (1 - im * ω * t * (1 + tau)) / (1 - im * ω * t), ts) / NM
    dcc = sum(t -> (-im * ω * t) / (1 - im * ω * t), ts) / NM
    return -visco_chi(ts, tau, f)^2 * real(-0.5 * cc^(-1.5) * dcc)
end

"""
Gradient with respect to the model fields from the gradient `dM` with respect to the internal (padded) arrays `rho`, `lam`,
`mu`, `taup`, `taus` (chain rule of `setup_medium`, the absorber padding and the units). `s` is the setup of `simulate`.
"""
function model_gradient(s, dM)
    haskey(dM, :wx) && return model_gradient_vti(s, dM)                    # VTI (anisotropy.jl)
    m, g, N = s.m, s.g, length(s.g.n)
    ts = s.med.ts
    pad(A) = pad_absorber(internal(A, Val(N)), g.n, s.na)
    rho_in = pad(m.rho)
    rho, vp, vs, qp, qs = max.(rho_in, 0.001), pad(m.vp), pad(m.vs), pad(m.Qp), pad(m.Qs)
    taup, taus = s.med.taup, s.med.taus
    gr, gl, gm, gtp, gts = (Float64.(Array(getfield(dM, k))) for k in (:rho, :lam, :mu, :taup, :taus))
    G = map(CartesianIndices(rho)) do I
        χp, χs = visco_chi(ts, taup[I], s.fq_ref), visco_chi(ts, taus[I], s.fq_ref)
        gmt = gm[I] - 2gl[I]                                           # lam = rho vp²/χp² - 2 mu
        drho = (gr[I] + gl[I] * vp[I]^2 / χp^2 + gmt * vs[I]^2 / χs^2) * (rho_in[I] >= 0.001)
        dvp = gl[I] * 2rho[I] * vp[I] / χp^2
        dvs = gmt * 2rho[I] * vs[I] / χs^2
        dtp = gtp[I] - gl[I] * 2rho[I] * vp[I]^2 / χp^3 * visco_dchi(ts, taup[I], s.fq_ref)
        dts = gts[I] - gmt * 2rho[I] * vs[I]^2 / χs^3 * visco_dchi(ts, taus[I], s.fq_ref)
        dqp = isfinite(qp[I]) ? -dtp * taup[I] / qp[I] : 0.0                # tau = NM zeta / Q
        dqs = isfinite(qs[I]) ? -dts * taus[I] / qs[I] : 0.0
        (drho / s.rho_unit, dvp, dvs, dqp, dqs)
    end
    return unpad_gradient(s, G, (:rho, :vp, :vs, :Qp, :Qs))
end

"""
The gradient as a `CartData` of the model with the fields `names`, from the per-cell tuples `G` on the padded grid: the
adjoint of the padding (absorber zones and ghost cells copy cells of the model) and of `internal`.
"""
function unpad_gradient(s, G, names)
    n, N = s.g.n, length(s.g.n)
    fields = map(1:length(names)) do q
        A = zeros(n)
        for I in CartesianIndices(G)
            J = ntuple(a -> a == N ? clamp(I[a] - H, 1, n[a] - s.na) : clamp(I[a] - H, s.na + 1, n[a] - s.na), N)
            A[J...] += G[I][q]
        end
        reverse(reshape(A, size(s.m.rho)), dims=3)
    end
    x, y, z = s.model.x.val, s.model.y.val, s.model.z.val
    return CartData(x, y, z, NamedTuple{names}(Tuple(fields)))
end
