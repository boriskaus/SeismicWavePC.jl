# 2D north-south section through the Matterhorn summit (Limberger et al., 2025, Fig. 6; see model.jl), x pointing north:
# vertically incident P and SV plane waves (summed as P + 1.7 SV, as the paper's forces excite them), Ormsby source
# 0.3-3.5 Hz; spectra at the summit and the valley (2.5 km south). Model 5 of the paper (no permafrost) and a permafrost
# cap (velocities × `fv`, density × 1.1, up to `thick` km thick below the summit, thinning to 0 at 2.9 km height),
# compared by the horizontal spectrum at the summit.
#
#   julia -t auto --project=examples/matterhorn examples/matterhorn/matterhorn_2d.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
const DX2 = 0.04
const NA2 = 20
const T2 = 25.0


"Material with a permafrost cap: nodes less than thick·w below the surface (w: 0 at 2.9 km, 1 at the summit)."
function permafrost!(model, mesh, sec, thick, fv)
    X, Z = mesh.x.val, mesh.z.val
    zs = sec.(X)
    w = clamp.((zs .- 2.9) ./ (Z_TOP - 2.9), 0, 1)
    frozen = (zs .- Z) .< thick .* w
    f = model.fields
    return addfield(mesh, (; rho=ifelse.(frozen, 1.1 .* f.rho, f.rho), vp=ifelse.(frozen, fv .* f.vp, f.vp),
                          vs=ifelse.(frozen, fv .* f.vs, f.vs), Qp=f.Qp, Qs=f.Qs))
end

function run2d(thick, fv)
    file = joinpath(@__DIR__, "..", "data", "matterhorn2d_$(round(Int, 1e3thick))m_$(round(Int, 100fv))_$(round(Int, 1e3DX2))m.jls")
    isfile(file) && return deserialize(file)
    mesh, model, sec = section_model(DX2, NA2)
    thick > 0 && (model = permafrost!(model, mesh, sec, thick, fv))
    stations = [Station(@sprintf("S%+05d", round(Int, 1e3x)), x, 0.0, 0.0; surface=true) for x in -4:0.1:4]
    v = map((:p, :sv)) do wave
        w = TeleseismicWave(wave; angle=0.0, backazimuth=270.0, box=(-BOX, BOX, Z_BOT), t0=0.0, trise=0.5, amplitude=1.0)
        r = simulate(model; grid=:curvilinear, tmax=T2, sources=w, stations, stf=:gaussian, absorber=:pml, na=NA2, T=Float64, verbose=false)
        (r.t, r.v)
    end
    res = (; t=v[1][1], v=v[1][2] .+ 1.7 .* v[2][2], vp=v[1][2], vsv=v[2][2], x=[s.x for s in stations], names=[s.name for s in stations])
    mkpath(dirname(file)); serialize(file, res)
    @printf("2D, permafrost %.0f m × %.1f: %d nodes, %d steps\n", 1e3thick, fv, prod(size(mesh)), length(res.t))
    return res
end

function spectrum(v, dt; w=0.05)
    n = length(v); L = nextpow(2, 4n)
    A = abs.(SeismicWavePC.rfft([v; zeros(L - n)])) .* dt
    f = (0:length(A)-1) ./ (L * dt)
    k = max(1, round(Int, w / (f[2] - f[1])))
    return f, [sum(A[max(1, i - k):min(end, i + k)]) / length(max(1, i - k):min(length(A), i + k)) for i in eachindex(A)]
end

cases = [(0.0, 1.0, "no permafrost (model 5)"), (0.6, 1.3, "600 m, +30 % (model 1)"), (0.6, 1.5, "600 m, +50 %")]
R = Dict((th, fv) => run2d(th, fv) for (th, fv, _) in cases)
r0 = R[(0.0, 1.0)]
summit, valley = argmin(abs.(r0.x)), argmin(abs.(r0.x .+ 2.5))
fg = collect(0:0.005:4.0)                                         # common frequencies (the runs differ in dt)
"Ormsby response of a run, and the smoothed amplitude spectrum of component `c` at station `k` on fg."
function spec(r, c, k)
    dt = r.t[2] - r.t[1]
    o = reshape_source(r.v[:, c:c, k:k], dt, 0.5, τ -> ormsby(τ), 4.0)
    f, S = spectrum(vec(o), dt)
    return [S[clamp(searchsortedlast(f, q), 1, length(f) - 1)] + (S[clamp(searchsortedlast(f, q), 1, length(f) - 1)+1] -
            S[clamp(searchsortedlast(f, q), 1, length(f) - 1)]) * (q - f[clamp(searchsortedlast(f, q), 1, length(f) - 1)]) /
            (f[2] - f[1]) for q in fg], o
end
band_peak(S, a, b) = (m = argmax(S .* (a .<= fg .<= b)); (fg[m], S[m]))
fig = Figure(size=(900, 650), fontsize=15)
axh = Axis(fig[1, 1], ylabel="horizontal (rel.)", title="2D N-S section: spectra at the summit (dashed: valley)")
axv = Axis(fig[2, 1], ylabel="vertical (rel.)", xlabel="frequency (Hz)")
S0, _ = spec(r0, 1, summit)
norm = maximum(S0)
p1, p2 = band_peak(S0, 0.25, 0.6), band_peak(S0, 1.3, 1.55)
println("2D section ($(round(Int, 1e3DX2)) m): horizontal spectrum at the summit (Ormsby source), peak heights relative to model 5")
for ((th, fv, lab), col) in zip(cases, (:green, :black, :crimson))
    r = R[(th, fv)]
    Sh, o = spec(r, 1, summit); Sv, _ = spec(r, 2, summit)
    _, ov = spec(r, 1, valley)
    lines!(axh, fg, Sh ./ norm, color=col, label=lab); lines!(axv, fg, Sv ./ norm, color=col)
    b = 0.3 .<= fg .<= 3.5
    pk = [fg[i] for i in findall(b)[2:end-1] if Sh[i] > Sh[i-1] && Sh[i] > Sh[i+1] && Sh[i] > 0.3maximum(Sh[b])]
    q1, q2 = band_peak(Sh, 0.25, 0.6), band_peak(Sh, 1.3, 1.55)
    @printf("  %-24s peaks %s Hz; fundamental %.2f Hz (%+.0f %%), second %.2f Hz (%+.0f %%); summit/valley PGV %.1f\n",
            lab, join((@sprintf("%.2f", p) for p in pk), ", "), q1[1], 100 * (q1[2] / p1[2] - 1), q2[1], 100 * (q2[2] / p2[2] - 1),
            maximum(abs, o) / maximum(abs, ov))
end
Q, _ = spec(r0, 1, valley)
lines!(axh, fg, Q ./ norm, color=:gray, linestyle=:dash, label="valley (model 5)")
for ax in (axh, axv); xlims!(ax, 0, 4); end
axislegend(axh, position=:rt)
save(joinpath(@__DIR__, "matterhorn_2d_spectra.png"), fig)
println("saved matterhorn_2d_spectra.png")
