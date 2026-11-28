# Born (linearized) modelling: the seismograms of a model perturbation δm to first order, J δm, with J the Jacobian of the
# seismograms with respect to the model. The simulation runs with numbers that carry a tangent (forward-mode
# differentiation): the medium holds its perturbation δM (ρ, λ, μ, τp, τs from δm, the chain rule of `model_gradient`
# forwards), and the tangent of the wavefield is the Born (scattered) wavefield. This is the exact linearization of the
# discrete scheme (free surface, PML, attenuation, the averaged moduli and buoyancies), the transpose of the adjoint-state
# gradient: ⟨J δm, r⟩ = ⟨δm, Jᵀ r⟩ to rounding errors. It runs with the kernels of `simulate`, on CPU threads and GPUs.

"A number with a tangent: `v + ε d` with ε² = 0 (forward-mode differentiation through the kernels)."
struct Tan{T<:AbstractFloat} <: Real
    v::T
    d::T
end
Tan(v::Real, d::Real) = Tan(promote(float(v), float(d))...)
Tan{T}(x::Real) where {T} = Tan{T}(T(x), zero(T))
Tan{T}(x::Tan) where {T} = Tan{T}(T(x.v), T(x.d))
const Num = Union{AbstractFloat,Integer}          # the plain numbers of the kernels (not other number types with tangents)
Base.promote_rule(::Type{Tan{T}}, ::Type{S}) where {T,S<:Num} = Tan{promote_type(T, S)}
Base.promote_rule(::Type{Tan{T}}, ::Type{Tan{S}}) where {T,S} = Tan{promote_type(T, S)}
@inline Base.zero(::Type{Tan{T}}) where {T} = Tan{T}(zero(T), zero(T))
@inline Base.one(::Type{Tan{T}}) where {T} = Tan{T}(one(T), zero(T))
@inline Base.zero(x::Tan) = zero(typeof(x))
@inline Base.one(x::Tan) = one(typeof(x))
@inline Base.eps(::Type{Tan{T}}) where {T} = Tan{T}(eps(T), zero(T))
@inline Base.float(x::Tan) = x
@inline Base.:+(a::Tan, b::Tan) = Tan(a.v + b.v, a.d + b.d)
@inline Base.:-(a::Tan, b::Tan) = Tan(a.v - b.v, a.d - b.d)
@inline Base.:-(a::Tan) = Tan(-a.v, -a.d)
@inline Base.:*(a::Tan, b::Tan) = Tan(a.v * b.v, a.d * b.v + a.v * b.d)
@inline Base.:/(a::Tan, b::Tan) = (q = a.v / b.v; Tan(q, (a.d - q * b.d) / b.v))
@inline Base.:+(a::Tan, b::Num) = Tan(a.v + b, a.d)
@inline Base.:+(a::Num, b::Tan) = Tan(a + b.v, b.d)
@inline Base.:-(a::Tan, b::Num) = Tan(a.v - b, a.d)
@inline Base.:-(a::Num, b::Tan) = Tan(a - b.v, -b.d)
@inline Base.:*(a::Tan, b::Num) = Tan(a.v * b, a.d * b)
@inline Base.:*(a::Num, b::Tan) = Tan(a * b.v, a * b.d)
@inline Base.:*(a::Bool, b::Tan) = ifelse(a, b, zero(b))
@inline Base.:*(a::Tan, b::Bool) = ifelse(b, a, zero(a))
@inline Base.:/(a::Tan, b::Num) = Tan(a.v / b, a.d / b)
@inline Base.:/(a::Num, b::Tan) = (q = a / b.v; Tan(q, -q * b.d / b.v))
Base.:(==)(a::Tan, b::Tan) = a.v == b.v && a.d == b.d
Base.:<(a::Tan, b::Tan) = a.v < b.v
Base.:<=(a::Tan, b::Tan) = a.v <= b.v
Base.isfinite(a::Tan) = isfinite(a.v) && isfinite(a.d)
@inline Base.sqrt(a::Tan) = (r = sqrt(a.v); Tan(r, ifelse(r > 0, a.d / 2r, zero(r))))     # 0 at 0 (air)
@inline Base.sincos(a::Tan) = ((s, c) = sincos(a.v); (Tan(s, c * a.d), Tan(c, -s * a.d)))         # (TTI: the tilt)
value(x::Tan) = x.v
value(x::Real) = x
tangent(x::Tan) = x.d

"""
Constants without their (zero) tangents: coefficients, time step, sponge, PML profiles and source amplitudes enter the kernels
as plain numbers (fewer operations: CPU loops vectorize better). Fields and memory variables (`mem`) keep their tangents.
"""
plain(x::Tan) = x.v
plain(x::Union{Tuple,NamedTuple}) = map(plain, x)
plain(A::AbstractArray{<:Tan}) = copyto!(similar(A, typeof(value(zero(eltype(A))))), value.(Array(A)))
plain(A::AbstractArray{<:Tuple}) = (B = map(plain, Array(A)); copyto!(similar(A, eltype(B)), B))
plain(x) = x
plain_zone(::Nothing) = nothing
plain_zone(P::NamedTuple) = NamedTuple{keys(P)}(map((k, v) -> k === :mem ? v : plain(v), keys(P), values(P)))
plain_region((lo, ndr, P)) = (lo, ndr, plain_zone(P))

"""
    background, scattered = born(model, δm; sources, stations, T=Float32, kwargs...)

Born (linearized) modelling: the seismograms `scattered` of the model perturbation `δm` to first order, J δm (J: the
Jacobian of the seismograms with respect to the model), and the seismograms `background` of `model`, from one simulation.

- `δm`: a `CartData` on the grid of `model` with any of the fields `rho`, `vp`, `vs`, `Qp`, `Qs` (and, for VTI models,
  `epsilon`, `delta`, `gamma`; missing fields: no perturbation), in the units of the model.
- `kwargs` are those of [`simulate`](@ref) (`dt`, `nt`, `stf`, `wave`, `interpolation`, `absorber`, `backend`, …);
  `T` is the floating-point type (`Float64` for tests of the linearization).

It is the exact derivative of the discrete simulation, the transpose of the gradient of [`misfit_gradient`](@ref): for the
L2 misfit, `misfit_gradient` gives Jᵀ r for the residuals r, and ⟨J δm, r⟩ = ⟨δm, Jᵀ r⟩. Together they give Gauss-Newton
Hessian-vector products JᵀJ δm, and least-squares migration. It costs about 2.5 forward simulations on GPUs and 4.5 on CPU
threads, on one or several MPI processes (`nproc`), on the staggered and on curvilinear grids. Not (yet) for teleseismic waves (plane waves are not differentiated, as in
`misfit_gradient`).
"""
function born(model::CartData, δm::CartData; T::Type{<:AbstractFloat}=Float32, kwargs...)
    bad = intersect(keys(kwargs), (:snapshots, :callback, :checkpoint, :restart, :driver))
    isempty(bad) || error("born does not take the keywords $(join(bad, ", "))")
    size(δm) == size(model) || error("δm must be on the grid of the model (size $(size(model)), got $(size(δm)))")
    keys(δm.fields) ⊆ (:rho, :vp, :vs, :Qp, :Qs, :epsilon, :delta, :gamma, :tilt, :azimuth, :z, :Z) ||    # (z, Z: the heights of curvilinear meshes)
        error("δm may have the fields rho, vp, vs, Qp, Qs (and epsilon, delta, gamma, tilt, azimuth), got $(keys(δm.fields))")
    any(k -> haskey(δm.fields, k) && !haskey(model.fields, k), (:tilt, :azimuth)) &&
        error("δm perturbs the tilt or azimuth of the symmetry axis: give the model these fields")
    has_anisotropy(δm) && !has_anisotropy(model) &&
        error("δm perturbs epsilon, delta or gamma: give the model these fields (zero for an isotropic background)")
    return simulate(model; kwargs..., T=Tan{T}, verbose=false, driver=BornDriver(δm))
end

struct BornDriver{D}
    δm::D
end

"""
The perturbation of the internal (padded) arrays `rho`, `lam`, `mu`, `taup`, `taus` by the perturbation `δm` of the model
fields: the derivative of `setup_medium` (with the absorber padding and the units), whose transpose is `model_gradient`.
"""
function medium_perturbation(s, δm)
    haskey(s.med, :wx) && return medium_perturbation_vti(s, δm)        # VTI (anisotropy.jl)
    m, g, N = s.m, s.g, length(s.g.n)
    ts, f = s.med.ts, s.fq_ref
    pad(A) = pad_absorber(internal(A, Val(N)), g.n, s.na)
    field(k) = haskey(δm.fields, k) ? pad(convert(Array{Float64,3}, δm.fields[k])) : zeros(size(s.med.rho))
    rho_in = pad(m.rho)
    rho, vp, vs, qp, qs = max.(rho_in, 0.001), pad(m.vp), pad(m.vs), pad(m.Qp), pad(m.Qs)
    taup, taus = s.med.taup, s.med.taus
    δrho = field(:rho) ./ s.rho_unit .* (rho_in .>= 0.001)
    δvp, δvs, δqp, δqs = field(:vp), field(:vs), field(:Qp), field(:Qs)
    δtp = @. ifelse(isfinite(qp), -taup / qp * δqp, 0.0)                 # tau = NM zeta / Q
    δts = @. ifelse(isfinite(qs), -taus / qs * δqs, 0.0)
    χp, χs = visco_chi.((ts,), taup, f), visco_chi.((ts,), taus, f)
    dχp, dχs = visco_dchi.((ts,), taup, f), visco_dchi.((ts,), taus, f)
    δmu = @. δrho * vs^2 / χs^2 + 2rho * vs * δvs / χs^2 - 2rho * vs^2 / χs^3 * dχs * δts
    δlam = @. δrho * vp^2 / χp^2 + 2rho * vp * δvp / χp^2 - 2rho * vp^2 / χp^3 * dχp * δtp - 2δmu
    return (; rho=δrho, lam=δlam, mu=δmu, taup=δtp, taus=δts)
end

function (d::BornDriver)(s)
    s.tw === nothing || error("born: teleseismic waves are not supported yet")
    (; F, M, St, rec, acc, regions, backend, nt, nc, scale) = s
    FT = typeof(value(zero(s.T)))
    δM = medium_perturbation(s, d.δm)
    # the medium with its perturbation as tangent (the part of the padded global grid on this rank)
    R = ntuple(a -> s.off[a]+1:s.off[a]+s.nl[a], length(s.nl))
    function perturbed(A, δ)
        B = Tan{FT}.(value.(Array(A)), FT.(δ[R...]))
        return copyto!(similar(A), B)
    end
    medium = keys(δM)
    Mp = merge(map(plain, Base.structdiff(M, NamedTuple{medium})),
               map(k -> perturbed(getfield(M, k), getfield(δM, k)), NamedTuple{medium}(medium)))
    # (the source amplitudes in plain numbers, in windows of their own: source_window.jl)
    Wm = source_window(s.amp_m, size(s.Wm.host, 1), nt, s.dev, FT, backend)
    Wf = source_window(s.amp_f, size(s.Wf.host, 1), nt, s.dev, FT, backend)
    p = merge(s, (; c=plain(s.c), vc=plain(s.vc), dtT=plain(s.dtT), Wm, Wf))
    reg = map(plain_region, regions)
    Src, FSrc, Stp = merge(plain(s.Src), (; s=Wm.s)), merge(plain(s.FSrc), (; s=Wf.s)), plain(St)   # read only: cells, sinc weights
    for it in 1:nt
        forward_step!(F, Mp, Src, FSrc, Stp, rec, acc, reg, it, p, true)
    end
    KernelAbstractions.synchronize(backend)
    rec_host = Array(rec)
    ns = (size(rec_host, 2) - 2nc) ÷ 2
    tensor = s.mode === :psv ? (:xx, :zz, :xz) : s.mode === :sh ? (:yz, :xy) : (:xx, :yy, :zz, :yz, :xz, :xy)
    t = collect(s.tbeg .+ (0:s.nrec-1) .* (s.record_every * s.dt))
    function seismograms(part)
        traces = zeros(s.nrec, 2nc + 2ns, length(s.stations))
        traces[:, :, s.own] .= part.(rec_host[:, :, 1:s.nown])
        s.parallel && MPI.Allreduce!(traces, +, MPI.COMM_WORLD)
        p(r, f) = traces[:, r, :] .* (scale * f)
        return Seismograms(t, p(1:nc, 1), p(nc+1:2nc, 1), p(2nc+1:2nc+ns, 1e6), p(2nc+ns+1:2nc+2ns, 1e-3), s.stations,
                           s.components, tensor, nothing)
    end
    return seismograms(value), seismograms(tangent)
end
