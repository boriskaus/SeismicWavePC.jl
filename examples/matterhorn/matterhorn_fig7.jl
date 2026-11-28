# Fig. 7 of Limberger et al. (2025): the 2D north-south section through the Matterhorn summit (x north) under a plane
# wave arriving vertically, at 45° from the north and at 45° from the south. Per node, the peak over time of the
# horizontal and vertical ground velocity and of the deviatoric stress dQ = √J2 (J2 of the deviatoric part of the stress,
# plane strain: σyy = ν (σxx + σzz); supplement of the paper), scaled to the paper's scenarios A (summit PGV 1 cm/s) and B
# (30 cm/s).
#
# The paper's forces (rotated by 45°) send a P and an S wave in the same direction; here two runs (TeleseismicWave :p and
# :sv, same angle and backazimuth; one plane wave per run) are summed as P + 1.7 SV (S 1.7× = vp/vs stronger in velocity,
# as from a force), the Ormsby wavelet (0.3-3.5 Hz) as the incoming particle velocity. The velocities of the P run are kept
# (every 10 ms, upper part of the section) and added to the SV run in its callback, which integrates the displacement and
# computes the strain on the mesh and the elastic plane-strain stress (λ, μ of the model; the small dispersion of Q is
# neglected) and the running maxima.
#
#   julia -t auto --project=examples/matterhorn examples/matterhorn/matterhorn_fig7.jl [dx_km] [cpu|metal] [z0|halfspace] [plus|minus]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
const DX7 = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.04
const DEVICE = length(ARGS) >= 2 ? ARGS[2] : "cpu"
DEVICE == "metal" && using Metal
const BACKEND, FT = DEVICE == "metal" ? (MetalBackend(), Float32) : (CPU(), Float64)
const NA7, TMAX7, DT_CB = 20, 25.0, 0.01
# The angle of the oblique waves: `TeleseismicWave`'s angle is the incidence in the half-space below the box (vp 6 km/s);
# the paper tilts its source plane by 45° at z = 0 (vp(0) = 5.07 km/s). "z0": 45° at z = 0, i.e. asin(sin 45° · vp(Z_BOT)/vp(0))
# = 56.7° in the half-space (Snell; the same for SV, as vp/vs is constant); "halfspace": 45° in the half-space.
const ANGLE_AT = length(ARGS) >= 3 ? ARGS[3] : "z0"
const A45 = ANGLE_AT == "z0" ? asind(sind(45) * vp_of(Z_BOT - 1.0) / vp_of(0.0)) : 45.0
const SIG7 = 0.1                                                  # DEM smoothing [km]: a fixed scale, independent of dx

mesh, model, sec = section_model(DX7, NA7; σ=SIG7)
X, Z = mesh.x.val[:, 1, :], mesh.z.val[:, 1, :]                  # km
nx, nz = size(X)
I = findall(x -> abs(x) <= 4.3, X[:, end])                        # the columns of the figure
k0 = findfirst(k -> minimum(Z[I, k]) >= 1.0, 1:nz)                # rows above ~1 km
K = k0:nz
Xr, Zr = 1e3 .* X[I, K], 1e3 .* Z[I, K]                           # m
ρ, vp, vs = (model.fields[f][I, 1, K] for f in (:rho, :vp, :vs))
const μ = ρ .* (1e3vs) .^ 2                                      # (const: used in the callback, which must not allocate)
const λ = ρ .* (1e3vp) .^ 2 .- 2μ
const ν = λ ./ (2 .* (λ .+ μ))

"""
Derivative along the dimension `dim` of a matrix in index space (central inside, 2nd-order one-sided at the ends), into
`B`. (Type-stable loops: this runs in the snapshot callback thousands of times; views chosen per call allocated per element.)
"""
function d1!(B, A, dim)
    n1, n2 = size(A)
    if dim == 1
        @inbounds for j in 1:n2
            for i in 2:n1-1
                B[i, j] = (A[i+1, j] - A[i-1, j]) / 2
            end
            B[1, j] = (-3A[1, j] + 4A[2, j] - A[3, j]) / 2
            B[n1, j] = (3A[n1, j] - 4A[n1-1, j] + A[n1-2, j]) / 2
        end
    else
        @inbounds for j in 2:n2-1, i in 1:n1
            B[i, j] = (A[i, j+1] - A[i, j-1]) / 2
        end
        @inbounds for i in 1:n1
            B[i, 1] = (-3A[i, 1] + 4A[i, 2] - A[i, 3]) / 2
            B[i, n2] = (3A[i, n2] - 4A[i, n2-1] + A[i, n2-2]) / 2
        end
    end
    return B
end
d1(A, dim) = d1!(similar(A), A, dim)
const XI, XK, ZI, ZK = d1(Xr, 1), d1(Xr, 2), d1(Zr, 1), d1(Zr, 2)
const JAC = XI .* ZK .- XK .* ZI

"Running maxima of |v_h|, |v_z| and dQ, updated with the velocities `vx`, `vz` of one snapshot (interval `h`)."
struct Peaks
    ux::Matrix{Float64}; uz::Matrix{Float64}; vxo::Matrix{Float64}; vzo::Matrix{Float64}
    vh::Matrix{Float64}; vv::Matrix{Float64}; dq::Matrix{Float64}
    w::NTuple{4,Matrix{Float64}}                                  # (work arrays: index-space derivatives)
end
Peaks() = Peaks((zeros(size(Xr)) for _ in 1:7)..., ntuple(_ -> zeros(size(Xr)), 4))
"""
Update with the velocities of one snapshot: the displacement (trapezoidal), its gradient (index-space differences and the
chain rule), the elastic plane-strain stress (σyy = ν (σxx + σzz)) and dQ = √J2 of its deviatoric part.
"""
function update!(q::Peaks, vx, vz, h)
    q.ux .+= (vx .+ q.vxo) .* (h / 2); q.uz .+= (vz .+ q.vzo) .* (h / 2)
    q.vxo .= vx; q.vzo .= vz
    q.vh .= max.(q.vh, abs.(vx)); q.vv .= max.(q.vv, abs.(vz))
    uxi, uxk, uzi, uzk = d1!(q.w[1], q.ux, 1), d1!(q.w[2], q.ux, 2), d1!(q.w[3], q.uz, 1), d1!(q.w[4], q.uz, 2)
    @inbounds for n in eachindex(q.dq)
        uxx, uxz = (uxi[n] * ZK[n] - uxk[n] * ZI[n]) / JAC[n], (uxk[n] * XI[n] - uxi[n] * XK[n]) / JAC[n]
        uzx, uzz = (uzi[n] * ZK[n] - uzk[n] * ZI[n]) / JAC[n], (uzk[n] * XI[n] - uzi[n] * XK[n]) / JAC[n]
        l, m = λ[n], μ[n]
        sxx = (l + 2m) * uxx + l * uzz
        szz = l * uxx + (l + 2m) * uzz
        sxz = m * (uxz + uzx)
        syy = ν[n] * (sxx + szz)
        P = -(sxx + syy + szz) / 3
        J2 = ((sxx + P)^2 + (syy + P)^2 + (szz + P)^2 + 2sxz^2) / 2
        q.dq[n] = max(q.dq[n], sqrt(J2))
    end
    return q
end

"""
The peaks of |v_h|, |v_z| and dQ for P and SV plane waves (angle in the half-space, backazimuth), for both relative
polarities, P + 1.7 SV (`plus`) and P - 1.7 SV (`minus`; the paper does not state it), from the same snapshots.
"""
function peaks(angle, baz)
    file = joinpath(@__DIR__, "..", "data", "matterhorn_fig7_$(round(Int, angle))_$(round(Int, baz))_$(round(Int, 1e3DX7))m_s$(round(Int, 1e3SIG7))_pm.jls")
    isfile(file) && return deserialize(file)
    stf = ormsby_stf()
    box = (-BOX, BOX, Z_BOT)
    kw = (; grid=:curvilinear, tmax=TMAX7, stf, absorber=:pml, na=NA7, T=FT, backend=BACKEND, verbose=false,
          stations=[Station("summit", 0.0, 0.0, 0.0; surface=true)])
    # the time step: from a short run (the same model, so the same dt)
    r0 = simulate(model; kw..., tmax=0.05, sources=TeleseismicWave(:p; angle, backazimuth=baz, box, trise=1.0))
    dt = r0.t[2] - r0.t[1]
    every = max(1, round(Int, DT_CB / dt))
    store = Vector{NTuple{2,Matrix{Float32}}}()
    simulate(model; kw..., sources=TeleseismicWave(:p; angle, backazimuth=baz, box, trise=1.0), snapshot_every=every,
             callback=(d, t) -> push!(store, (Float32.(d.fields.V[1][I, 1, K]), Float32.(d.fields.V[3][I, 1, K]))))
    qp, qm = Peaks(), Peaks()
    n = Ref(0)
    function cb(d, t)
        n[] += 1
        n[] > length(store) && return
        sx, sz = 1.7 .* d.fields.V[1][I, 1, K], 1.7 .* d.fields.V[3][I, 1, K]
        update!(qp, store[n[]][1] .+ sx, store[n[]][2] .+ sz, every * dt)
        update!(qm, store[n[]][1] .- sx, store[n[]][2] .- sz, every * dt)
    end
    time = @elapsed simulate(model; kw..., sources=TeleseismicWave(:sv; angle, backazimuth=baz, box, trise=1.0), snapshot_every=every, callback=cb)
    res = (; plus=(; vh=qp.vh, vv=qp.vv, dq=qp.dq), minus=(; vh=qm.vh, vv=qm.vv, dq=qm.dq), every, dt, nsnap=length(store))
    serialize(file, res)
    @printf("angle %5.2f°, backazimuth %3.0f°: %d snapshots every %d steps (dt = %.3f ms), SV run %.0f s\n", angle, baz, length(store), every, 1e3dt, time)
    return res
end

cases = [(0.0, 270.0, "vertical incidence"), (A45, 90.0, "45° from the north"), (A45, 270.0, "45° from the south")]
@printf("oblique waves: %.2f° in the half-space (%s)\n", A45, ANGLE_AT == "z0" ? "45° at z = 0" : "45° in the half-space")
const POL = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :plus          # :plus (P + 1.7 SV) or :minus (P - 1.7 SV)
RR = [peaks(a, b) for (a, b, _) in cases]
R = [getfield(r, POL) for r in RR]

# --- numbers: scaled to the scenarios (summit horizontal PGV 1 and 30 cm/s)
top = size(Xr, 2)
isum = argmin(abs.(Xr[:, top]))
zs = Zr[:, top]
println("Fig. 7 (2D N-S section, $(round(Int, 1e3DX7)) m), peaks scaled to a summit horizontal PGV of 1 cm/s (A) / 30 cm/s (B)")
for ((a, b, lab), r) in zip(cases, R)
    s = 0.01 / r.vh[isum, top]                                     # scenario A
    ih = argmax(r.vh[:, top]); iv = argmax(r.vv[:, top]); iq = argmax(r.dq[:, top])
    valley = r.vh[argmin(abs.(Xr[:, top] .+ 2500)), top] * s
    north = [i for i in axes(Xr, 1) if Xr[i, top] > 300]; south = [i for i in axes(Xr, 1) if Xr[i, top] < -300]
    qn, qs = argmax(r.dq[north, top]), argmax(r.dq[south, top])
    @printf("  %-20s max |v_h| at x = %+5.0f m (z %4.0f m); max |v_z| at x = %+5.0f m (z %4.0f m); valley (2.5 km S) |v_h| %.2f cm/s\n",
            lab, Xr[ih, top], zs[ih], Xr[iv, top], zs[iv], 100valley)
    @printf("  %-20s surface dQ max %.0f kPa (A) / %.1f MPa (B) at x = %+5.0f m, z = %4.0f m; summit %.0f kPa; north flank max %.0f kPa at %4.0f m, south flank max %.0f kPa at %4.0f m (A)\n",
            "", 1e-3s * r.dq[iq, top], 30e-6s * r.dq[iq, top], Xr[iq, top], zs[iq], 1e-3s * r.dq[isum, top],
            1e-3s * r.dq[north[qn], top], zs[north[qn]], 1e-3s * r.dq[south[qs], top], zs[south[qs]])
end

# surface dQ (A) by flank and elevation band: the paper describes the southern flank at 3800-4300 m and a salient point on
# the northern side near 3400 m (here the corner between the north face and the shoulder, whose peak grows with
# resolution: a stress concentration at a re-entrant corner)
println("  surface dQ (scenario A, kPa) by band: S 3800-4300 m | N 3800-4300 m | N shoulder 3300-3700 m | S 3000-3800 m | summit")
for ((a, b, lab), r) in zip(cases, R)
    s = 0.01 / r.vh[isum, top]
    q = 1e-3s .* r.dq[:, top]
    sel(side, z1, z2) = [i for i in eachindex(q) if side * Xr[i, top] > 50 && z1 <= zs[i] <= z2]
    mx(v) = isempty(v) ? NaN : maximum(q[v])
    @printf("  %-20s %5.0f | %5.0f | %5.0f | %5.0f | %5.0f\n", lab, mx(sel(-1, 3800, 4300)), mx(sel(1, 3800, 4300)), mx(sel(1, 3300, 3700)),
            mx(sel(-1, 3000, 3800)), q[isum])
end

# --- the 3 × 3 figure (A-C horizontal, D-F vertical PGV, G-I dQ; scenario A values, B = 30×)
nI, nK = size(Xr)
verts = [vec(Xr) ./ 1e3 vec(Zr) ./ 1e3]                           # node (i, k) is row i + (k - 1) nI
idx(i, k) = (k - 1) * nI + i
faces = vcat([[idx(i, k) idx(i + 1, k) idx(i + 1, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...,
             [[idx(i, k) idx(i + 1, k + 1) idx(i, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...)
fig = Figure(size=(1500, 1000), fontsize=14)
s_all = [0.01 / r.vh[isum, top] for r in R]
for (j, ((a, b, lab), r, s)) in enumerate(zip(cases, R, s_all))
    for (row, (F, name, crange, unit, cmap)) in enumerate(((100s .* r.vh, "horizontal", (0.1, 1.0), "max horizontal PGV (cm/s, A; B: ×30)", :batlow),
                                                         (100s .* r.vv, "vertical", (0.1, 1.0), "max vertical PGV (cm/s, A; B: ×30)", :batlow),
                                                         (1e-3s .* r.dq, "dQ", (1.0, 40.0), "max dQ (kPa, A; B: ×30)", :vik)))
        ax = Axis(fig[row, j], title="$(('A':'I')[3(row-1)+j]) $lab", xlabel=row == 3 ? "south ← x (km) → north" : "",
                  ylabel=j == 1 ? "elevation (km)" : "", aspect=DataAspect())
        m = mesh!(ax, verts, faces, color=clamp.(vec(F), crange...), colormap=cmap, colorscale=log10, colorrange=crange, shading=NoShading)
        lines!(ax, Xr[:, top] ./ 1e3, Zr[:, top] ./ 1e3, color=:black, linewidth=1)
        xlims!(ax, -1.3, 1.5); ylims!(ax, 2.85, 4.55)              # (the stored rows start at 2.81 km below the summit)
        j == 3 && Colorbar(fig[row, 4], m, label=unit)
        dir = a == 0 ? 0 : b == 90.0 ? -1 : 1                    # P-wave direction (x north, z up)
        x0 = dir == 0 ? collect(-0.6:0.3:0.6) : [-dir * 0.9]
        arrows2d!(ax, x0, fill(2.9, length(x0)), fill(dir * 0.25, length(x0)), fill(0.25, length(x0)), color=:white)
    end
end
save(joinpath(@__DIR__, "matterhorn_fig7_$(round(Int, 1e3DX7))m_s$(round(Int, 1e3SIG7))_$(ANGLE_AT)_$(POL).png"), fig)
println("saved matterhorn_fig7_$(round(Int, 1e3DX7))m_s$(round(Int, 1e3SIG7))_$(ANGLE_AT)_$(POL).png")
