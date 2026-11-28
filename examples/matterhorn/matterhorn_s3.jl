# Fig. S3 of Limberger et al. (2025): the peak ground velocity and the peak deviatoric stress dQ over the surface of the
# 3D Matterhorn model for a vertically incident plane wave, a Ricker wavelet of 0.4 Hz (as their Fig. 5A). The three
# incoming waves (P, S polarized NS and EW; summed as P + 1.7 S as in matterhorn_3d.jl) run one after the other; the top
# three rows of nodes within 5 km of the summit are kept (every ~25 ms) from the first two runs and added in the callback
# of the third, which integrates the displacement and computes the strain at the surface (index-space differences, the
# chain rule for vertical columns, second order one-sided downwards), the elastic stress, dQ = √J2 (3D), and the running
# maxima of |v| and dQ. The 0.4 Hz wavelet needs no fine grid: 100 m (≥ 17 points per S wavelength below 1 Hz).
#
#   julia -t auto --project=examples/matterhorn examples/matterhorn/matterhorn_s3.jl [metal|cpu]
#
using SeismicWavePC, GeophysicalModelGenerator, Serialization, Printf
const DEVICE = length(ARGS) >= 1 ? ARGS[1] : "metal"
DEVICE == "metal" && using Metal
const backend = DEVICE == "metal" ? MetalBackend() : CPU()
include("model.jl")
const DXS, NAS, TMAXS, DT_CB = 0.1, 12, 25.0, 0.025

function run_s3()
    file = joinpath(@__DIR__, "..", "data", "matterhorn_s3_100m.jls")
    isfile(file) && return deserialize(file)
    margin = (NAS + SeismicWavePC.SBP_NB + 10) * DXS
    L = BOX + margin
    n = round(Int, 2L / DXS) + 1
    bottom = Z_BOT - (NAS + SeismicWavePC.SBP_NB + 12) * 0.75DXS
    mesh = matterhorn_mesh(range(-L, L, length=n), range(-L, L, length=n), matterhorn_surface(); dz=0.75DXS, bottom)
    model = matterhorn_model(mesh, DXS)
    X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
    I = findall(x -> abs(x) <= 5.0, X[:, 1, 1]); J = findall(y -> abs(y) <= 5.0, Y[1, :, 1])
    nz = size(Z, 3); K = nz-2:nz
    Xm, Ym, Zm = 1e3 .* X[I, 1, 1], 1e3 .* Y[1, J, 1], 1e3 .* Z[I, J, K]          # m
    ρ, vp, vs = (model.fields[f][I, J, nz] for f in (:rho, :vp, :vs))
    μ = ρ .* (1e3vs) .^ 2; λ = ρ .* (1e3vp) .^ 2 .- 2μ
    # stf: the 0.4 Hz Ricker wavelet as the incoming particle velocity, centred at 3 s
    ts = collect(0:0.005:6.0)
    stf = SampledSTF(ts, ricker(ts .- 3.0, 0.4))
    kw = (; grid=:curvilinear, tmax=TMAXS, stf, absorber=:pml, na=NAS, backend, T=Float32, verbose=false,
          stations=[Station("summit", 0.0, 0.0, 0.0; surface=true)])
    w(wave, baz) = TeleseismicWave(wave; angle=0.0, backazimuth=baz, box=(-BOX, BOX, -BOX, BOX, Z_BOT), t0=0.0, trise=1.0, amplitude=1.0)
    r0 = simulate(model; kw..., tmax=0.05, sources=w(:p, 0.0))
    dt = r0.t[2] - r0.t[1]
    every = max(1, round(Int, DT_CB / dt))
    grab(d) = Float32.(cat((d.fields.V[c][I, J, K] for c in 1:3)...; dims=4))     # (i, j, 3 rows, 3 components)
    stores = map(((:p, 0.0, 1.0), (:sv, 180.0, 1.7))) do (wave, baz, a)
        S = Array{Float32,4}[]
        t = @elapsed simulate(model; kw..., sources=w(wave, baz), snapshot_every=every, callback=(d, _) -> push!(S, a .* grab(d)))
        @printf("  %s run: %.0f s\n", wave, t)
        S
    end
    u = zeros(length(I), length(J), 3, 3); vo = zero(u)
    pgv, dq = zeros(length(I), length(J)), zeros(length(I), length(J))
    di(A) = (B = similar(A); n1 = size(A, 1); for i in 1:n1
                 B[i, :, :] .= i == 1 ? A[2, :, :] .- A[1, :, :] : i == n1 ? A[n1, :, :] .- A[n1-1, :, :] : (A[i+1, :, :] .- A[i-1, :, :]) ./ 2
             end; B)
    dj(A) = permutedims(di(permutedims(A, (2, 1, 3))), (2, 1, 3))
    dk(A) = (3 .* A[:, :, 3] .- 4 .* A[:, :, 2] .+ A[:, :, 1]) ./ 2                    # at the top row, one-sided
    Xi = [Xm[min(i + 1, end)] - Xm[max(i - 1, 1)] for i in eachindex(Xm)] ./ [min(i + 1, length(Xm)) - max(i - 1, 1) for i in eachindex(Xm)]
    Yj = [Ym[min(j + 1, end)] - Ym[max(j - 1, 1)] for j in eachindex(Ym)] ./ [min(j + 1, length(Ym)) - max(j - 1, 1) for j in eachindex(Ym)]
    Zi, Zj, Zk = di(Zm)[:, :, 3], dj(Zm)[:, :, 3], dk(Zm)
    "∂f/∂x, ∂f/∂y, ∂f/∂z at the top row of a field on the three top rows (vertical columns: x = X(i), y = Y(j))."
    function grad(f)
        fi, fj, fk = di(f)[:, :, 3], dj(f)[:, :, 3], dk(f)
        fz = fk ./ Zk
        return (fi .- fz .* Zi) ./ Xi, (fj .- fz .* Zj) ./ Yj', fz
    end
    n = Ref(0)
    function cb(d, _)
        n[] += 1
        n[] > min(length(stores[1]), length(stores[2])) && return
        v = stores[1][n[]] .+ stores[2][n[]] .+ 1.7f0 .* grab(d)
        h = every * dt
        u .+= (v .+ vo) .* (h / 2); vo .= v
        pgv .= max.(pgv, sqrt.(v[:, :, 3, 1] .^ 2 .+ v[:, :, 3, 2] .^ 2 .+ v[:, :, 3, 3] .^ 2))
        G = [grad(u[:, :, :, c]) for c in 1:3]                      # G[c][a] = ∂u_c/∂x_a
        e(a, b) = (G[a][b] .+ G[b][a]) ./ 2
        tr = e(1, 1) .+ e(2, 2) .+ e(3, 3)
        s(a, b) = 2μ .* e(a, b) .+ (a == b ? λ .* tr : 0)
        P = (s(1, 1) .+ s(2, 2) .+ s(3, 3)) ./ 3
        J2 = ((s(1, 1) .- P) .^ 2 .+ (s(2, 2) .- P) .^ 2 .+ (s(3, 3) .- P) .^ 2) ./ 2 .+ s(1, 2) .^ 2 .+ s(1, 3) .^ 2 .+ s(2, 3) .^ 2
        dq .= max.(dq, sqrt.(J2))
    end
    t = @elapsed simulate(model; kw..., sources=w(:sv, 270.0), snapshot_every=every, callback=cb)
    @printf("  sv (EW) run with the stress: %.0f s; %d snapshots every %d steps (dt = %.2f ms); %.1f M nodes\n", t, n[], every, 1e3dt,
            prod(size(mesh)) / 1e6)
    res = (; x=Xm, y=Ym, z=Zm[:, :, 3], pgv, dq, dt, every, nodes=prod(size(mesh)))
    serialize(file, res)
    return res
end

run_s3()
