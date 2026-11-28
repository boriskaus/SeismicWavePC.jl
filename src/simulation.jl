# Driver: domain decomposition (ImplicitGlobalGrid), sources, stations, time loop and output.

"""
    Station(name, x, y, z; surface=false)
    Station(name, lon, lat, z, proj::ProjectionPoint; surface=false)

Receiver at model coordinates `x, y, z` [km] (z up, as in the `CartData`), or at a geographic location that is projected
with the same `ProjectionPoint` that was used to create the model (see `GeophysicalModelGenerator.convert2CartData`).
With `surface=true` the station is placed in the first solid cell below the free surface of its column and `z` is ignored.
In 2D models `y` is ignored.
"""
struct Station
    name::String
    x::Float64
    y::Float64
    z::Float64
    surface::Bool
end
Station(name, x, y, z; surface=false) = Station(name, x, y, z, surface)
Station(name, lon, lat, z, proj::ProjectionPoint; surface=false) = Station(name, project(lon, lat, z, proj)...; surface)

function Base.show(io::IO, s::Station)
    print(io, "Station(\"", s.name, "\", x=", s.x, ", y=", s.y, s.surface ? ", at surface)" : ", z=$(s.z))")
end

"""
    Seismograms

Result of [`simulate`](@ref). Fields:
- `t`: times [s] of the samples
- `v`: velocities `v[it, component, station]` [m/s], components along `components` (z positive up)
- `u`: displacements [m], like `v`
- `stress` [Pa] and `strain`: `stress[it, component, station]`, components along `tensor_components`
- `stations`: the `Station`s
- `components`: `(:x, :z)` (P-SV), `(:y,)` (SH) or `(:x, :y, :z)` (3D)
- `tensor_components`: `(:xx, :zz, :xz)` (P-SV), `(:yz, :xy)` (SH) or `(:xx, :yy, :zz, :yz, :xz, :xy)` (3D)
- `wavefield`: velocity field `V` [m/s] at the end of the simulation as `CartData` (on MPI rank 0; `nothing` elsewhere)
- `div`, `curl`: with `simulate(...; derivatives = true)`, the divergence `div[it, station]` and the curl
  `curl[it, component, station]` (components x, y, z) of the velocity [1/s]; otherwise `nothing`

`r["name"]` returns the `(nt, ncomponents)` matrix of the station called `"name"`.
"""
struct Seismograms
    t::Vector{Float64}
    v::Array{Float64,3}
    u::Array{Float64,3}
    stress::Array{Float64,3}
    strain::Array{Float64,3}
    stations::Vector{Station}
    components::Tuple{Vararg{Symbol}}
    tensor_components::Tuple{Vararg{Symbol}}
    wavefield::Union{CartData,Nothing}
    div::Union{Matrix{Float64},Nothing}
    curl::Union{Array{Float64,3},Nothing}
end
Seismograms(t, v, u, stress, strain, stations, components, tensor_components, wavefield) =
    Seismograms(t, v, u, stress, strain, stations, components, tensor_components, wavefield, nothing, nothing)

function Base.getindex(r::Seismograms, name::AbstractString)
    n = findfirst(s -> s.name == name, r.stations)
    n === nothing && error("no station \"$name\"")
    return r.v[:, :, n]
end

function Base.show(io::IO, ::MIME"text/plain", r::Seismograms)
    println(io, "Seismograms: ", length(r.stations), " station(s), ", length(r.t), " samples, t = ",
            first(r.t), " … ", round(last(r.t), sigdigits=6), " s, components ", join(r.components, ", "))
    for (n, s) in enumerate(r.stations)
        println(io, "  ", rpad(s.name, 10), " max |v| = ", join((@sprintf("%.3e", maximum(abs, view(r.v, :, c, n))) for c in eachindex(r.components)), ", "), " m/s")
    end
    r.wavefield === nothing || print(io, "  final wavefield: CartData of size ", size(r.wavefield))
end

function igg_device(backend)
    name = string(nameof(typeof(backend)))
    name == "CUDABackend" && return "CUDA"
    name == "ROCBackend" && return "AMDGPU"
    backend isa CPU && return "none"
    error("MPI runs are supported on CPU, CUDA and AMDGPU backends, not $name")
end

"""
    result = simulate(model; dt, nt, sources, kwargs...)

Run a visco-elastic wave propagation simulation (OpenSWPC algorithm) on the velocity model `model` and return
[`Seismograms`](@ref).

`model` is a `CartData` (coordinates in km, z up; constant spacing in x and y, constant or smoothly varying in z) with
fields `rho` [kg/m³], `vp`, `vs` [km/s] and optionally `Qp`, `Qs` (default: no attenuation) and the Thomsen parameters
`epsilon`, `delta`, `gamma` (VTI anisotropy; `vp`, `vs` are then vertical velocities; on curvilinear grids also `tilt`,
`azimuth` [°]: a tilted symmetry axis, TTI). Cells with `vp = vs = 0` are
air: the free surface and topography follow from that. Cells with `vs = 0` and `vp > 0` are water. A `CartData` of size
`(nx, 1, nz)` gives a 2D simulation in the x-z plane. A model too large to hold as a whole is given as a
[`DistributedModel`](@ref), of which every MPI process builds only its own subdomain.

Keyword arguments
- `dt`, `nt`: time step [s] and number of steps; `tbeg = 0.0` is the start time.
- `sources`: `MomentTensorSource`, `ForceSource`, `PlaneWave` or a vector of them (see `DoubleCoupleSource`); `stf = :kupper` source time function: a name or any function (see `moment_rate`, `SampledSTF`).
- `stations = Station[]`; seismograms (velocity, displacement, stress, strain) are recorded every `record_every = 1` steps.
- `interpolation = :cell`: sources and stations at the cell that contains them (as OpenSWPC), or `:sinc` at their exact
  position, between grid points (Kaiser-windowed sinc interpolation, Hicks 2002; strain is then still taken at the cell).
- `wave = :psv`: `:psv` or `:sh`, for 2D models.
- `grid = :auto`: a deformed 2D model (a terrain-following mesh, see [`curvilinear_mesh`](@ref)) runs on a curvilinear grid
  ([`SeismicWavePC.simulate_curvilinear`](@ref): SBP finite differences, the free surface at the true topography),
  any other on the staggered grid; `:curvilinear` or `:staggered` to choose.
- `free_surface = :openswpc`: the shear modulus at the shear-stress points is the harmonic mean of the four cells around them,
  zero next to air or water (as OpenSWPC). On a stair-stepped topography this also zeroes the shear stress at the inner
  corners of the steps, so a top cell one cell wide in x (or y) has no shear coupling to the ground in that direction and
  its horizontal velocity stays exactly zero there. `:corners` takes the harmonic mean of the three solid cells at these
  inner corners (one cell of the four with μ = 0); flat surfaces and outer corners are unchanged (identical results on a
  flat free surface). P-SV and 3D (SH has no such points).
- `absorber = :pml`: absorbing boundary at the sides and the bottom, `:pml` (perfectly matched layer), `:mpml` (multiaxial
  PML) or `:cerjan` (sponge); `na = 20` is its thickness in cells. The PML can grow exponentially where a soft surface layer
  (high Poisson ratio, e.g. water-saturated sediments) reaches into the side layers. The M-PML is stable there: each layer also
  damps the derivatives parallel to it, with the fraction `mpml_ratio = 0.1` of the layer's damping. It reflects more than the
  PML (less than `:cerjan`), mostly as a slowly decaying long-period residual. (`:pml` damps the top `h` of the side layers in
  z too: `h` = `na` cells, or on a stretched z axis the thickness of the bottom layer; with less air than `h` this reaches into
  the ground. `:mpml` does not damp at the free surface.) Plane waves need `:pml`.
- `fq_min = 0.05`, `fq_max = 5.0`, `fq_ref = 1.0`: frequency band [Hz] of constant Q, and reference frequency of the velocities.
- `density_unit = :kg_m3`: unit of `rho`, or `:g_cm3` (as in OpenSWPC).
- `snapshot_every = 0`: if > 0, write the velocity field every so many steps to ParaView files in `snapshot_dir = "snapshots"`.
- `snapshots = nothing`: [`Snapshots`](@ref) of planes as netCDF files (OpenSWPC format; needs `using NCDatasets`), written
  every `snapshot_every` steps into `snapshot_dir` instead of the ParaView files.
- `callback = nothing`: a function `callback(wavefield::CartData, t)` that is called every `snapshot_every` steps (on rank 0)
  instead of writing ParaView files, e.g. to plot the wavefield while the simulation runs. `wavefield` has the field `V` [m/s].
- `checkpoint = nothing`: file name (without extension) for checkpoints, written every `checkpoint_every` steps (one file per
  MPI rank); with `restart = true` the simulation continues from the last checkpoint. Snapshots before it are not rewritten.
- `backend = CPU()`: KernelAbstractions backend, e.g. `CUDABackend()`, `ROCBackend()`, `MetalBackend()`.
- `T = Float32`: floating point type of the wavefield.
- `nproc = 1`: MPI processes in x (2D), or `(npx, npy)` (3D); start julia with `mpiexec -n npx*npy`.
- `wavefield = true`: return the velocity field at the end (collected on rank 0) in the result; `false` by default for a
  `DistributedModel`.
- `verbose = true`: print a summary and the progress (on rank 0).
- `derivatives = false`: also the divergence and curl of the velocity at the stations (`div`, `curl` of the result), from
  4th-order centred differences over `dh` [km] (default: the grid spacing in x) between auxiliary stations around each
  station; at stations on the surface the vertical derivatives are one-sided from below.
"""
function simulate(model::Union{CartData,DistributedModel}; wave=:psv, interpolation=:cell, free_surface=:openswpc, grid=:auto,
                  derivatives=false, dh=nothing, kwargs...)
    derivatives && return simulate_derivatives(model, dh; wave, interpolation, free_surface, grid, kwargs...)
    grid in (:auto, :staggered, :curvilinear) || error("grid must be :auto, :staggered or :curvilinear, got :$grid")
    if (model isa CartData && (grid === :curvilinear || (grid === :auto && is_curvilinear(model)))) || (model isa DistributedModel && is_curvilinear(model))
        return size(model)[2] == 1 ? simulate_curvilinear(model; wave, kwargs...) : simulate_curvilinear3d(model; kwargs...)
    end
    has_tilt(model) && error("a tilted symmetry axis (`tilt`, `azimuth`: TTI) needs grid = :curvilinear (also for flat surfaces)")
    mode = size(model)[2] == 1 ? wave : :p3d
    mode in (:psv, :sh, :p3d) || error("wave must be :psv or :sh, got :$wave")
    interpolation in (:cell, :sinc) || error("interpolation must be :cell or :sinc, got :$interpolation")
    free_surface in (:openswpc, :corners) || error("free_surface must be :openswpc or :corners, got :$free_surface")
    return simulate(model, Val(mode), Val(interpolation), Val(has_attenuation(model)), Val(has_anisotropy(model)), Val(free_surface); kwargs...)
end

"Does the model have attenuation (finite `Qp` or `Qs` anywhere)? Without, the memory variables are left out."
has_attenuation(model::CartData) = any(f -> haskey(model.fields, f) && any(isfinite, model.fields[f]), (:Qp, :Qs))
has_attenuation(model::DistributedModel) = model.attenuation

# Function barrier on the solver type: from here on the dimension N is known to the compiler.
function simulate(model::Union{CartData,DistributedModel}, ::Val{mode}, ::Val{interp}=Val(:cell), ::Val{attenuation}=Val(true), ::Val{aniso}=Val(false),
                  ::Val{fsurf}=Val(:openswpc); dt, nt, sources, stations=Station[], stf=:kupper, tbeg=0.0,
                  absorber=:pml, mpml_ratio=0.1, na=20, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                  density_unit=:kg_m3, snapshot_every=0, snapshot_dir="snapshots", callback=nothing,
                  snapshots=nothing, checkpoint=nothing, checkpoint_every=0, restart=false, driver=nothing, backend=CPU(), T::Type{FT}=Float32, nproc=1, verbose=true,
                  wavefield=model isa CartData, periodic=false) where {mode,interp,attenuation,aniso,fsurf,FT}
    N = mode === :p3d ? 3 : 2
    # periodic side boundaries per horizontal axis (x, y): the wavefield leaves on one side and enters on the other
    per = periodic isa Bool ? ntuple(_ -> periodic, Val(N - 1)) : ntuple(a -> Bool(periodic[a]), Val(N - 1))
    sources = sources isa Union{MomentTensorSource,ForceSource,PlaneWave,TeleseismicWave} ? [sources] : collect(sources)
    all(s -> s isa Union{MomentTensorSource,ForceSource,PlaneWave,TeleseismicWave}, sources) ||
        error("sources must be MomentTensorSource, ForceSource, PlaneWave or TeleseismicWave")
    stations = collect(Station, stations)
    isempty(sources) && error("at least one source is required")
    distributed = model isa DistributedModel        # every rank has the medium of its own subdomain only
    if any(per)
        distributed && error("periodic boundaries need a CartData model (not a DistributedModel)")
        driver === nothing || error("periodic boundaries: simulations only (no misfit gradients or Born modelling yet)")
        any(s -> s isa TeleseismicWave, sources) && error("periodic boundaries: no TeleseismicWave (it is injected along a box)")
    end
    distributed && !(interp === :cell && driver === nothing && all(s -> s isa Union{MomentTensorSource,ForceSource}, sources)) &&
        error("a DistributedModel takes MomentTensorSource and ForceSource with interpolation = :cell; plane and teleseismic waves, interpolation = :sinc, misfit gradients and Born modelling need a CartData")
    m = whole_model(model, density_unit, Val(aniso))
    axes_ = model_axes(model)
    g = Grid(axes_, Val(N))
    all(ntuple(a -> g.n[a] > (a < N ? (per[a] ? 0 : 2na) : na), N)) || error("the model ($(join(g.n, "×")) cells) must be larger than the absorbing layers (na = $na cells)")
    decomposition, med0, courant = decomposed_medium(model, m, g, nproc, backend, density_unit, na, fq_min, fq_max, fq_ref, dt, Val(aniso); periodic=per)
    (; dims, nl, parallel, me, nb, off, R) = decomposition
    rule = fsurf === :openswpc ? (;) : (; free_surface=Val(fsurf))     # averaging of the shear moduli (`shear_rule`)
    med = merge(med0, rule)
    dev(A) = (B = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)

    # medium on this rank
    loc(A) = dev(FT.(A[R...]))
    col(A) = dev(A[Base.front(R)...])
    absorber in (:pml, :mpml, :cerjan) || error("absorber must be :pml, :mpml or :cerjan, got :$absorber")
    absorber === :mpml && !(0 <= mpml_ratio <= 1) && error("mpml_ratio must be between 0 and 1, got $mpml_ratio")
    absorber === :mpml && any(s -> s isa PlaneWave, sources) &&
        error("plane waves need absorber = :pml: they pass its side layers, which the M-PML damps along them")
    mpml = absorber === :mpml ? mpml_ratio : nothing             # (the PML zones of the M-PML: `pml_regions`)
    sponge(a) = map(v -> dev(FT.(v[R[a]])), absorber === :cerjan && !(a < N && per[a]) ? cerjan(g.n[a], g.d[a], na; top=a < N) : (ones(g.n[a] + 2H), ones(g.n[a] + 2H)))
    (gx_c, gx_b), (gz_c, gz_b) = sponge(1), sponge(N)
    G = N == 3 ? (; gx_c, gx_b, gy_c=sponge(2)[1], gy_b=sponge(2)[2], gz_c, gz_b) : (; gx_c, gx_b, gz_c, gz_b)
    M = merge((; rho=loc(med.rho), lam=loc(med.lam), mu=loc(med.mu), taup=loc(med.taup), taus=loc(med.taus),
                 fs_top=col(med.fs_top), fs_bot=col(med.fs_bot), ob_top=col(med.ob_top), ob_bot=col(med.ob_bot)), G,
              aniso ? (; wx=loc(med.wx), wz=loc(med.wz), s13=loc(med.s13), mu66=loc(med.mu66)) : (;), rule)
    # coefficients of the differences: constant along x (and y); per layer in depth, for fields at cell centres and at faces
    c = (ntuple(a -> (FT(9 / 8 / g.d[a]), FT(1 / 24 / g.d[a]), FT(1 / g.d[a])), N - 1)..., map(t -> map(dev, t), depth_coefficients(g, FT))...)
    attenuation || !med.attenuated || error("the model has attenuation (finite Qp or Qs): use Val(true)")
    vc = attenuation ? memory_coefficients(med.ts, dt, FT) : nothing         # nothing: elastic, no memory variables
    ijk = N == 3 ? (:i, :j, :k) : (:i, :k)

    # sources: moment tensors as stress glut at the cell of the source, forces on the velocities around it, normalized by
    # the total moment and force (forces carry a factor 10³ for the units, as in OpenSWPC)
    position(x, y, z) = cell_position(g, x, y, z)
    inside(I) = all(H + 1 .<= I .<= g.n .+ H)
    active = active_sources(sources, g, med, distributed && parallel)                             # as OpenSWPC, ignore sources
    moments = MomentTensorSource[s for (s, a) in zip(sources, active) if a && s isa MomentTensorSource]   # outside the model
    forces = ForceSource[s for (s, a) in zip(sources, active) if a && s isa ForceSource]
    planewaves = PlaneWave[s for s in sources if s isa PlaneWave]
    teleseismic = TeleseismicWave[s for s in sources if s isa TeleseismicWave]
    length(teleseismic) <= 1 || error("one TeleseismicWave per simulation")
    aniso && !(isempty(planewaves) && isempty(teleseismic)) && error("plane waves and teleseismic waves need an isotropic model (for now)")
    nused = length(moments) + length(forces) + length(planewaves) + length(teleseismic)
    nused == 0 && error("all sources are outside the model or in the air (vp = vs = 0)")
    nused < length(sources) && me == 0 &&
        @warn "$(length(sources) - nused) of $(length(sources)) sources are outside the model or in the air (vp = vs = 0) and are ignored"
    M0_ = sum((s.M0 for s in moments); init=0.0) + 1e3 * sum((sqrt(s.F[1]^2 + s.F[2]^2 + s.F[3]^2) for s in forces); init=0.0)
    M0 = M0_ > 0 ? M0_ : N == 3 ? 1e15 : 1e12     # plane waves only: output scale 1 (OpenSWPC)
    sinc = interp === :sinc                       # compile-time constant
    ipos(x, y, z) = internal_position(g, x, y, z)
    spec_v = (cells=stagger(Val(mode)).velocity, shear=ntuple(_ -> false, length(stagger(Val(mode)).velocity)))
    spec_s = (cells=stagger(Val(mode)).stress, shear=stagger(Val(mode)).shear)
    here(s) = sinc ? sinc_on_rank(g, med, ipos(s.x, s.y, s.z), s isa ForceSource ? spec_v : spec_s, off, nl) :
              all(2 .<= position(s.x, s.y, s.z) .- off .<= nl)    # the stencil reaches i-1, k-1; ghost copies are overwritten by the halo update
    iS, iF = findall(here, moments), findall(here, forces)
    # volume of the source: of its cell, or nominal between grid points (the weights carry the local spacing in depth)
    volume(s) = sinc ? prod(g.d) : prod(Base.front(g.d)) * thickness(g, position(s.x, s.y, s.z)[N])
    S, FS = moments[iS], forces[iF]
    # the injected amplitudes per source and step, on the device in windows of steps (source_window.jl)
    amp_m(n) = (s = S[n]; amp_row(source_stf(s, stf), s.M0 / M0, dt, volume(s), tbeg, 0.5, dt, s.t0, s.trise))
    amp_f(n) = (s = FS[n]; amp_row(source_stf(s, stf), 1e3 / M0, dt, volume(s), tbeg, 0, dt, s.t0, s.trise))
    Wm, Wf = source_window(amp_m, length(S), nt, dev, FT, backend), source_window(amp_f, length(FS), nt, dev, FT, backend)
    Src0 = merge(source_cells(S, position, off, dev, Val(N)), (; M=dev(FT[s.M[m] * (m in (4, 5) ? -1 : 1) for s in S, m in 1:6]),     # z up -> z down
                              s=Wm.s))
    FSrc0 = merge(source_cells(FS, position, off, dev, Val(N)), (; F=dev(FT[s.F[m] * (m == 3 ? -1 : 1) for s in FS, m in 1:3]),
                                s=Wf.s))
    # between grid points: the weights of the points around the sources (Kaiser-windowed sinc)
    Src = sinc ? merge(Src0, first(sinc_tables(g, med, [ipos(s.x, s.y, s.z) for s in S], spec_s, off, nl, dev, FT; derivatives=Val(true),
                                                items=collect(eachindex(S)), nodes=Val(true)))) : Src0
    FSrc = sinc ? merge(FSrc0, first(sinc_tables(g, med, [ipos(s.x, s.y, s.z) for s in FS], spec_v, off, nl, dev, FT; derivatives=Val(true),
                                                  items=collect(eachindex(FS)), nodes=Val(true)))) : FSrc0
    fcut = max(maximum((1 / s.trise for s in moments); init=0.0), maximum((1 / s.trise for s in forces); init=0.0),
               maximum((plane_wave_fcut(p, med, g) for p in planewaves); init=0.0), maximum((1 / w.trise for w in teleseismic); init=0.0))

    # stations owned by this rank
    function station_cell(st)
        I = position(st.x, st.y, st.z)
        # (of a DistributedModel only the ranks around a column know its surface, the owner of the station among them)
        st.surface && all(H + 1 .<= Base.front(I) .<= Base.front(g.n) .+ H) &&
            (I = (Base.front(I)..., held(med.kfs, Base.front(I)...) ? med.kfs[Base.front(I)...] + 1 + H : H + 1))
        inside(I) || error("station $(st.name) is outside the model")
        return I
    end
    SI = map(station_cell, stations)
    own, St = sinc ? sinc_stations(stations, SI, g, med, ipos, spec_v, spec_s, off, nl, dev, FT, Val(ijk)) :
                     cell_stations(SI, off, nl, dev, Val(ijk))
    components = mode === :psv ? (:x, :z) : mode === :sh ? (:y,) : (:x, :y, :z)
    tensor_components = mode === :psv ? (:xx, :zz, :xz) : mode === :sh ? (:yz, :xy) : (:xx, :yy, :zz, :yz, :xz, :xy)
    nc, ns = length(components), length(tensor_components)
    nrec = (nt - 1) ÷ record_every + 1
    rec = KernelAbstractions.zeros(backend, FT, nrec, 2nc + 2ns, max(length(own), 1))   # velocity, displacement, stress, strain
    acc = KernelAbstractions.zeros(backend, FT, nc + ns, max(length(own), 1))           # displacement and strain, integrated every step

    # wavefields
    z() = KernelAbstractions.zeros(backend, FT, nl...)
    zr() = KernelAbstractions.zeros(backend, FT, NM, nl...)
    F0 = mode === :psv ? (; Vx=z(), Vz=z(), Sxx=z(), Szz=z(), Sxz=z()) : mode === :sh ? (; Vy=z(), Syz=z(), Sxy=z()) :
         (; Vx=z(), Vy=z(), Vz=z(), Sxx=z(), Syy=z(), Szz=z(), Syz=z(), Sxz=z(), Sxy=z())
    memvars() = mode === :psv ? (; Rxx=zr(), Rzz=zr(), Rxz=zr()) : mode === :sh ? (; Ryz=zr(), Rxy=zr()) :
                (; Rxx=zr(), Ryy=zr(), Rzz=zr(), Ryz=zr(), Rxz=zr(), Rxy=zr())     # one per stress component
    F = attenuation ? merge(F0, memvars()) : F0
    scale = M0 * (N == 3 ? 1e-15 : 1e-12)     # wavefield units -> m/s (OpenSWPC convention)
    out = Output{N,mode,typeof(model)}(model, map(length, (axes_.x, axes_.y, axes_.z)), nl, dims, parallel, me, scale)
    # a teleseismic wave: tables of the incoming field and the strips of corrected cells along its box
    tw = isempty(teleseismic) ? nothing :
         (injection_setup(teleseismic[1], Val(mode), med, g, na, off, nl, dev, stf, scale, tbeg, dt, nt, FT)..., injection_kernels(Val(mode)))

    # plane waves: initial wavefield; with the PML the side boundaries let them pass
    for p in planewaves
        for (name, A) in plane_wave_fields(p, Val(mode), med, g, off, nl, stf, dt, scale)
            B = F[name]
            copyto!(B, Array(B) .+ FT.(A))
        end
    end
    edges = ntuple(a -> per[a] ? (false, false) : (off[a] == 0, off[a] + nl[a] - 2H == g.n[a]), N - 1)   # (periodic: no extrapolation)
    pw_edges = !isempty(planewaves) && absorber === :pml

    if verbose && me == 0
        @printf("SeismicWavePC: %s, %s cells (dx = %s km), %d steps of %g s, Courant number %.3f\n", mode === :p3d ? "3D" : uppercase(string(mode)),
                join(g.n, "×"), join(round.(g.d, sigdigits=4), ", "), nt, dt, courant)
        @printf("  %d source(s) (%s), %d station(s), backend %s, %s, %d MPI rank(s)\n", nused,
                isempty(forces) ? @sprintf("M0 = %.3e N m", M0) : @sprintf("%d moment tensor(s), %d force(s)", length(moments), length(forces)),
                length(stations), backend isa CPU ? "CPU ($(Threads.nthreads()) threads)" : string(nameof(typeof(backend))), T, prod(dims))
    end
    # internal: another driver of the time loop (misfit_gradient) gets the whole setup
    # cell ranges (offset, size, zone): the whole model (Cerjan), or the interior and the PML zones; the interior split by
    # rows into blocks with and without 2nd-order differences
    split = backend isa CPU
    if driver !== nothing
        regions = absorber === :cerjan ? ((ntuple(_ -> H, N), nl .- 2H, nothing),) :
                  pml_regions(Val(mode), g, na, dt, fcut, off, nl, R, dev, backend, FT; mpml, periodic=per)
        rows = plain_blocks(regions[1], med, off; split)
        res = driver((; F, M, Src, FSrc, St, rec, acc, c, vc, dtT=FT(dt), kernels=kernels(mode, sinc), regions, nt, nosource=isempty(S),
                        noforce=isempty(FS), nown=length(own), own, record_every, pw_edges, edges, backend, scale, nc, nrec, stations,
                        components, med, m, g, na, fq_ref, rho_unit=density_factor(density_unit),
                        mode, model, tbeg, dt, T=FT, parallel, nb, me, dims, nl, off, sources, moments, forces, M0, stf,
                        iS, iF, sinc, volume=map(volume, moments), fvolume=map(volume, forces), tw, rows, Wm, Wf, amp_m, amp_f, dev))
        parallel && IGG.finalize_global_grid(finalize_MPI=false)
        return res
    end
    planes = snapshots === nothing ? nothing : setup_planes(snapshots, Val(mode), g, med, off, nl, dev, backend, FT, snapshot_dir, me, parallel)
    movie = (snapshot_every > 0 && me == 0 && callback === nothing && snapshots === nothing) ? movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    wall = time()
    head = (F, M, Src, FSrc, St, rec, acc, c, vc, FT(dt), kernels(mode, sinc))
    ckp = checkpoint === nothing ? nothing : parallel ? "$(checkpoint)_rank$me.jls" : "$checkpoint.jls"
    restart && (ckp === nothing || !isfile(ckp)) && error("restart = true needs the checkpoint file $ckp")
    tail = (nt, isempty(S), isempty(FS), length(own), record_every, snapshot_every, snapshot_dir, callback, movie, out, tbeg, dt,
            verbose && me == 0, parallel, nb, ckp, checkpoint_every, restart, pw_edges, edges, planes, backend, tw, Wm, Wf, per)
    # (separate variables per branch: each call sees concrete regions)
    movie_out = if absorber === :cerjan
        whole = ((ntuple(_ -> H, N), nl .- 2H, nothing),)
        timeloop!(head..., (whole, plain_blocks(whole[1], med, off; split)), tail...)
    elseif mpml === nothing
        pml = pml_regions(Val(mode), g, na, dt, fcut, off, nl, R, dev, backend, FT; periodic=per)
        timeloop!(head..., (pml, plain_blocks(pml[1], med, off; split)), tail...)
    else
        mpl = pml_regions(Val(mode), g, na, dt, fcut, off, nl, R, dev, backend, FT; mpml, periodic=per)
        timeloop!(head..., (mpl, plain_blocks(mpl[1], med, off; split)), tail...)
    end
    movie_out === nothing || movie_paraview(pvd=movie_out, Finalize=true)
    planes === nothing || write_peaks(planes, snapshots.quantities, mode, scale, parallel, me)
    verbose && me == 0 && @printf("  finished in %.2f s\n", time() - wall)

    traces = zeros(nrec, 2nc + 2ns, length(stations))
    traces[:, :, own] .= Array(rec)[:, :, 1:length(own)]
    final = wavefield ? velocity(out, F) : nothing
    parallel && (MPI.Allreduce!(traces, +, MPI.COMM_WORLD); IGG.finalize_global_grid(finalize_MPI=false))
    part(r, f) = traces[:, r, :] .* (scale * f)  # wavefield units -> m/s, m, Pa, strain (OpenSWPC)
    return Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), part(1:nc, 1), part(nc+1:2nc, 1),
                       part(2nc+1:2nc+ns, 1e6), part(2nc+ns+1:2nc+2ns, 1e-3), stations, components, tensor_components, final)
end

"The arrays of a whole model (density in g/cm³, the internal unit); `nothing` for a model that no rank holds as a whole."
whole_model(model::CartData, density_unit, aniso=Val(false)) =
    (m = model_arrays(model, aniso); merge(m, (; rho=m.rho ./ density_scale(density_unit, maximum(m.rho)))))
whole_model(::DistributedModel, density_unit, aniso=Val(false)) = nothing

"""
Domain decomposition in x (and y) with ImplicitGlobalGrid: every rank owns `n / dims` cells plus `H` ghost cells on each side
(`nl` cells; `R` are their indices in the padded global grid, `off` its offset, `nb` the MPI neighbours left and right).
"""
function decompose(g::Grid{N}, nproc, backend; periodic=ntuple(_ -> false, Val(N - 1))) where {N}
    dims = ntuple(a -> a < N && a <= length(nproc) ? nproc[a] : 1, N)
    all(g.n .% dims .== 0) || error("cells per direction $(g.n) must be divisible by nproc $dims")
    nl = g.n .÷ dims .+ 2H
    parallel = prod(dims) > 1
    me, coords = 0, ntuple(_ -> 0, N)
    if parallel
        me, _, _, c3 = IGG.init_global_grid(nl..., ntuple(_ -> 1, 3 - N)...; dimx=dims[1], dimy=N == 3 ? dims[2] : 1, dimz=1,
                                            overlaps=(2H, 2H, 2H), halowidths=(H, H, H), init_MPI=!MPI.Initialized(),
                                            periodx=Int(periodic[1]), periody=N == 3 ? Int(periodic[2]) : 0,
                                            device_type=igg_device(backend), quiet=true)
        coords = ntuple(a -> Int(c3[a]), N)
    end
    nb = ntuple(a -> parallel ? (IGG.neighbor(1, a), IGG.neighbor(2, a)) : (-1, -1), N - 1)
    off = coords .* (nl .- 2H)
    R = ntuple(a -> off[a]+1:off[a]+nl[a], N)
    return (; dims, nl, parallel, me, nb, off, R)
end

"""
The domain decomposition, the medium and the Courant number. Of a `CartData` every rank sets up the whole medium (arrays of
the padded global grid); of a [`DistributedModel`](@ref) only the cells of its own subdomain ([`Window`](@ref)s), and the
numbers of the whole model are collected from all ranks.
"""
function decomposed_medium(::CartData, m, g::Grid{N}, nproc, backend, density_unit, na, fq_min, fq_max, fq_ref, dt, aniso;
                           periodic=ntuple(a -> false, Val(N - 1))) where {N}
    med = setup_medium(S -> model_block(m, S, Val(N)), g, ntuple(a -> 1:g.n[a]+2H, Val(N)), Val(false), na, fq_min, fq_max, fq_ref; periodic)
    courant = stable_courant(med, g, dt)
    return decompose(g, nproc, backend; periodic), med, courant
end
function decomposed_medium(model::DistributedModel, ::Nothing, g::Grid{N}, nproc, backend, density_unit, na, fq_min, fq_max, fq_ref, dt, aniso;
                           periodic=nothing) where {N}
    d = decompose(g, nproc, backend)
    try
        rho_max = Ref(-Inf)
        block(S) = (b = model_block(model, S, Val(N), density_factor(density_unit), aniso); rho_max[] = b.rho_max; b)
        # the cells of this rank and 3 columns around them: the 2nd-order zones of its ghost columns depend on those
        W = ntuple(a -> a < N ? (max(first(d.R[a]) - 3, 1):min(last(d.R[a]) + 3, g.n[a] + 2H)) : d.R[a], Val(N))
        local_ = setup_medium(block, g, W, Val(true), na, fq_min, fq_max, fq_ref)
        whole = if d.parallel
            comm = MPI.COMM_WORLD
            rho_max[] = MPI.Allreduce(rho_max[], max, comm)
            (; vmax=MPI.Allreduce(local_.vmax, max, comm), vlayer=MPI.Allreduce(local_.vlayer, max, comm),
               attenuated=MPI.Allreduce(Int(local_.attenuated), max, comm) > 0)
        else
            (; vmax=local_.vmax, vlayer=local_.vlayer, attenuated=local_.attenuated)
        end
        density_scale(density_unit, rho_max[])             # densities in the other unit?
        med = merge(local_, whole)
        return d, med, stable_courant(med, g, dt)
    catch
        d.parallel && IGG.finalize_global_grid(finalize_MPI=false)
        rethrow()
    end
end

"Courant number of the time step `dt`, which must be stable."
function stable_courant(med, g, dt)
    courant = courant_number(med, g, dt)
    courant < 6 / 7 || error("unstable time step: Courant number $courant ≥ 6/7, use dt < $(dt * 6 / 7 / courant)")
    return courant
end

"""
Which sources take part: point sources inside the model and not in the air (`vp = vs = 0`), and all others. Of a
[`DistributedModel`](@ref) the ranks that hold the cell of a source know; `shared` collects the answers of all ranks.
"""
function active_sources(sources, g::Grid, med, shared)
    active = map(sources) do s
        s isa Union{MomentTensorSource,ForceSource} || return true
        I = cell_position(g, s.x, s.y, s.z)
        all(H + 1 .<= I .<= g.n .+ H) && held(med.mu, I...) && !(med.mu[I...] == 0 && med.lam[I...] == 0)
    end
    return shared ? Vector{Bool}(MPI.Allreduce(Int.(active), max, MPI.COMM_WORLD) .> 0) : Vector{Bool}(active)
end

"Padded array indices of the cell that contains the point (x, y, z) [km], z up."
cell_position(g::Grid{3}, x, y, z) = (cell_index(g, 1, x), cell_index(g, 2, y), cell_index(g, 3, -z))
cell_position(g::Grid{2}, x, y, z) = (cell_index(g, 1, x), cell_index(g, 2, -z))

"Internal coordinates (x, [y,] depth) of a point (x, y, z) [km], z up."
internal_position(::Grid{3}, x, y, z) = (Float64(x), Float64(y), -Float64(z))
internal_position(::Grid{2}, x, y, z) = (Float64(x), -Float64(z))

"Stations at their cells: those owned by this rank and their local cell indices."
function cell_stations(SI, off, nl, dev, ::Val{ijk}) where {ijk}
    N = length(ijk)
    own = [n for n in eachindex(SI) if all(H + 1 .<= Base.front(SI[n]) .- Base.front(off) .<= Base.front(nl) .- H)]
    return own, NamedTuple{ijk}(ntuple(a -> dev(Int32[SI[n][a] - off[a] for n in own]), Val(N)))
end

"""
Stations between grid points (Kaiser-windowed sinc): those with points on this rank, the interpolation tables of the
velocity and stress components, and the cell of the strain (counted on the rank that owns it, `own`).
"""
function sinc_stations(stations, SI, g::Grid{N}, med, ipos, spec_v, spec_s, off, nl, dev, ::Type{FT}, ::Val{ijk}) where {N,FT,ijk}
    owns(I) = all(H + 1 .<= Base.front(I) .- Base.front(off) .<= Base.front(nl) .- H)
    spos = map(eachindex(stations)) do n
        st = stations[n]
        p = ipos(st.x, st.y, st.z)
        st.surface ? ntuple(a -> a < N ? p[a] : g.zc[SI[n][N]], N) : p     # surface: centre of the top cell
    end
    own = [n for n in eachindex(stations) if owns(SI[n]) || sinc_on_rank(g, med, spos[n], spec_v, off, nl) ||
                                               sinc_on_rank(g, med, spos[n], spec_s, off, nl)]
    tv = first(sinc_tables(g, med, spos, spec_v, off, nl, dev, FT; items=own))
    ts = first(sinc_tables(g, med, spos, spec_s, off, nl, dev, FT; items=own))
    cell(n, a) = Int32(clamp(SI[n][a] - off[a], H + 1, nl[a] - H))
    St = merge(NamedTuple{ijk}(ntuple(a -> dev(Int32[cell(n, a) for n in own]), Val(N))),
               (; own=dev(FT[owns(SI[n]) for n in own]), vi=tv.ix, vw=tv.w, si=ts.ix, sw=ts.w))
    return own, St
end

"Write the simulation state (all arrays in `state`) after step `it` to `file`."
function save_state(file, it, nt, state::Tuple)
    data = collect(map(a -> vec(Array(a)), state))
    open(io -> serialize(io, (it, nt, data)), file * ".tmp", "w")
    mv(file * ".tmp", file; force=true)                  # never leave a half-written checkpoint
end

"Read a checkpoint into the arrays of `state` (which must match); returns its step."
function load_state!(state::Tuple, file, nt)
    T = eltype(first(state))
    it, nt0, data = open(deserialize, file)::Tuple{Int,Int,Vector{Vector{T}}}
    nt0 == nt && length(data) == length(state) && sum(length, data) == sum(length, state) ||
        error("checkpoint $file does not belong to this simulation")
    ntuple(i -> copyto!(state[i], reshape(data[i], size(state[i]))), Val(length(state)))
    return it
end

"Cell indices (local) of the sources `S` as a `NamedTuple` `(i, [j,] k)` of device vectors."
function source_cells(S, position, off, dev, ::Val{N}) where {N}
    Is = [position(s.x, s.y, s.z) .- off for s in S]
    return NamedTuple{N == 3 ? (:i, :j, :k) : (:i, :k)}(ntuple(a -> dev(Int32[p[a] for p in Is]), Val(N)))
end

"Factor from `unit` to g/cm³; `rho_max` guards against densities given in the other unit."
function density_scale(unit, rho_max)
    factor = density_factor(unit)
    unit === :kg_m3 && rho_max < 100 && error("densities up to $rho_max look like g/cm³: use kg/m³, or pass density_unit=:g_cm3")
    unit === :g_cm3 && rho_max > 100 && error("densities up to $rho_max look like kg/m³: drop density_unit=:g_cm3")
    return factor
end
function density_factor(unit)
    unit === :kg_m3 || unit === :g_cm3 || error("density_unit must be :kg_m3 or :g_cm3, got :$unit")
    return unit === :kg_m3 ? 1000.0 : 1.0
end

function kernels(mode, sinc=false)
    k = mode === :psv ? (vel_psv!, stress_psv!, inject_psv!, force_psv!, record_psv!, plane_psv!) :
        mode === :sh ? (vel_sh!, stress_sh!, inject_sh!, force_sh!, record_sh!, plane_sh!) :
        (vel_3d!, stress_3d!, inject_3d!, force_3d!, record_3d!, plane_3d!)
    sinc || return k
    inject!, force!, _ = sinc_kernels(mode)
    return (k[1], k[2], inject!, force!, k[5], k[6])
end

velocities(F) = haskey(F, :Vx) ? (haskey(F, :Vy) ? (F.Vx, F.Vy, F.Vz) : (F.Vx, F.Vz)) : (F.Vy,)
stresses(F) = haskey(F, :Sxx) ? (haskey(F, :Syy) ? (F.Sxx, F.Syy, F.Szz, F.Syz, F.Sxz, F.Sxy) : (F.Sxx, F.Szz, F.Sxz)) : (F.Syz, F.Sxy)

# Function barrier: everything below is concretely typed.
function timeloop!(F, M, Src, FSrc, St, rec, acc, c, vc, dtT, (vel!, stress!, inject!, force!, record!, plane!), (regions, rows), nt, nosource, noforce, nown, record_every,
                   snapshot_every, snapshot_dir, callback, movie, out, tbeg, dt, verbose, parallel, nb, ckp, checkpoint_every, restart,
                   pw_edges, edges, planes, backend, tw, Wm, Wf, periodic)
    V, Σ = velocities(F), stresses(F)
    cells = (rows..., Base.tail(regions)...)                   # the interior by rows, and the PML zones
    report = max(nt ÷ 10, 1)
    state = (values(F)..., pml_state(regions)..., acc, rec)
    it0 = restart ? load_state!(state, ckp, nt) + 1 : 1
    restart && verbose && @printf("  restart from step %d\n", it0 - 1)
    for it in it0:nt
        itw = (it - 1) % record_every == 0 ? (it - 1) ÷ record_every + 1 : 0
        nown > 0 && foreach_item!(backend, record!, (rec, acc, F, St, c, dtT, itw), nown)
        if planes !== nothing
            store = snapshot_every > 0 && (it - 1) % snapshot_every == 0
            for p in planes
                isempty(p.own) || foreach_item!(backend, plane!, (p.B, p.U, p.PK, F, M, c, p.pts, dtT, store), length(p.own))
            end
            if store
                KernelAbstractions.synchronize(backend)
                write_planes(planes, solver(out), (it - 1) ÷ snapshot_every + 1, tbeg + (it - 1) * dt, out.scale, parallel, out.me)
            end
        end
        if snapshot_every > 0 && (it - 1) % snapshot_every == 0 && (planes === nothing || callback !== nothing)
            KernelAbstractions.synchronize(backend)
            data = velocity(out, F)
            t = tbeg + (it - 1) * dt
            if data !== nothing
                callback === nothing ? (movie = write_paraview(data, "wavefield_$(lpad(it - 1, 6, '0'))"; directory=snapshot_dir,
                                                               pvd=movie, time=t, verbose=false)) : callback(data, t)
            end
        end
        pw_edges && extrapolate_sides!(V, edges)
        foreach_region!(backend, stress!, (F, M, c, vc, dtT), cells)
        nosource || inject_sources!(backend, inject!, (F, Src, window_step!(Wm, it)), Src)
        inject_wave!(tw, Val(:stress), F, M, c, vc, dtT, eltype(dtT)(tbeg + (it - 0.5) * dt), backend)
        parallel && (KernelAbstractions.synchronize(backend); exchange_halo!(backend, nb, Σ...))
        !parallel && any(periodic) && wrap_halo!(periodic, Σ...)
        pw_edges && extrapolate_sides!(Σ, edges)
        foreach_region!(backend, vel!, (F, M, c, dtT), cells)
        noforce || inject_sources!(backend, force!, (F, M, FSrc, window_step!(Wf, it)), FSrc)
        inject_wave!(tw, Val(:velocity), F, M, c, vc, dtT, eltype(dtT)(tbeg + it * dt), backend)
        parallel && (KernelAbstractions.synchronize(backend); exchange_halo!(backend, nb, V...))
        !parallel && any(periodic) && wrap_halo!(periodic, V...)
        if verbose && it % report == 0
            KernelAbstractions.synchronize(backend)
            vmax = maximum(A -> maximum(abs, A), V) * out.scale
            @printf("  step %*d/%d  t = %8.4f s  max|v| = %.3e m/s\n", ndigits(nt), it, nt, tbeg + it * dt, vmax)
        end
        if ckp !== nothing && checkpoint_every > 0 && it % checkpoint_every == 0 && it < nt
            KernelAbstractions.synchronize(backend)
            save_state(ckp, it, nt, state)
        end
    end
    KernelAbstractions.synchronize(backend)
    return movie
end

# Halo exchange between MPI ranks along the decomposed axes (x, and y in 3D): ImplicitGlobalGrid on GPUs. On CPUs it
# costs ~0.4 ms per call (a task per field and neighbour), more than the time step of a small model: plain MPI there.
# (ImplicitGlobalGrid ≥ 0.18 does not wait for kernels still running on GPUs: synchronize first)
exchange_halo!(backend, nb, A...) = (KernelAbstractions.synchronize(backend); IGG.update_halo!(A...))
"""
One process with periodic sides (`periodic` per horizontal axis): the ghost cells get the cells next to the opposite side
(with MPI the halo exchange does this: ImplicitGlobalGrid with periodic directions).
"""
function wrap_halo!(periodic, A...)
    for f in A, a in eachindex(periodic)
        periodic[a] || continue
        n = size(f, a)
        sel(r) = ntuple(b -> b == a ? r : Colon(), ndims(f))
        copyto!(view(f, sel(1:H)...), view(f, sel(n-2H+1:n-H)...))
        copyto!(view(f, sel(n-H+1:n)...), view(f, sel(H+1:2H)...))
    end
end

function exchange_halo!(::CPU, nb, A::Vararg{Array{T,N}}) where {T,N}
    comm = IGG.comm()
    for a in 1:N-1
        left, right = nb[a]
        n = size(A[1], a)
        sel(r) = ntuple(b -> b == a ? r : 1:size(A[1], b), Val(N))
        pack(r) = reduce(vcat, (vec(f[sel(r)...]) for f in A))     # one message per neighbour with all fields
        sl, sr = pack(H+1:2H), pack(n-2H+1:n-H)                     # cells next to the ghost cells go to the neighbour's ghost cells
        rl, rr = similar(sl), similar(sr)
        MPI.Waitall([MPI.Irecv!(rl, comm; source=left, tag=0), MPI.Irecv!(rr, comm; source=right, tag=1),
                     MPI.Isend(sl, comm; dest=left, tag=1), MPI.Isend(sr, comm; dest=right, tag=0)])
        m = length(sl) ÷ length(A)
        for (i, f) in enumerate(A)
            left == MPI.PROC_NULL || copyto!(view(f, sel(1:H)...), view(rl, (i-1)*m+1:i*m))
            right == MPI.PROC_NULL || copyto!(view(f, sel(n-H+1:n)...), view(rr, (i-1)*m+1:i*m))
        end
    end
end

"How to turn the local wavefields into a `CartData` of the whole model."
struct Output{N,mode,M}
    model::M
    size::NTuple{3,Int}
    nl::NTuple{N,Int}
    dims::NTuple{N,Int}
    parallel::Bool
    me::Int
    scale::Float64
end

solver(::Output{N,mode}) where {N,mode} = mode

"Cell-centred velocity field `V` [m/s, z up] as `CartData` on rank 0 (`nothing` on other ranks)."
function velocity(o::Output{N,mode}, F) where {N,mode}
    in_ = ntuple(a -> H+1:o.nl[a]-H, N)
    centred(A, a) = (A[in_...] .+ A[ntuple(b -> b == a ? in_[b] .- 1 : in_[b], N)...]) ./ 2
    loc = mode === :psv ? (centred(F.Vx, 1), -centred(F.Vz, 2)) :
          mode === :sh ? (F.Vy[in_...],) : (centred(F.Vx, 1), centred(F.Vy, 2), -centred(F.Vz, 3))
    glob = map(loc) do A
        A = Array(A)
        o.parallel || return A
        G = o.me == 0 ? similar(A, size(A) .* o.dims) : nothing
        IGG.gather!(A, G)
        G
    end
    o.me == 0 || return nothing
    f(A) = reshape(reverse(Float64.(A) .* o.scale, dims=N), o.size)
    zero_ = zeros(o.size)
    V = mode === :psv ? (f(glob[1]), zero_, f(glob[2])) : mode === :sh ? (zero_, f(glob[1]), zero_) : f.(glob)
    return CartData(coordinates(o.model)..., (; V))
end
coordinates(m::CartData) = (m.x.val, m.y.val, m.z.val)
coordinates(m::DistributedModel) = xyz_grid(m.x, m.y, m.z)
