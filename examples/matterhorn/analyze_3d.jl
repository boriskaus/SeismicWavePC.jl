# Figures and numbers of the 3D Matterhorn runs (matterhorn_3d.jl): record sections (Fig. 2 of the paper), summit and
# valley seismograms, spectra and spectral ratios (Fig. 4A-C), and the amplification map for a 0.4 Hz Ricker wavelet
# (Fig. 5A). The three runs (P, S polarized NS and EW) are summed as the paper's x, y, z forces excite them: S waves 1.7×
# (vp/vs) stronger in velocity than the P wave.
#
#   julia --project=examples/matterhorn examples/matterhorn/analyze_3d.jl [60m]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "60m"
load(name) = deserialize(joinpath(@__DIR__, "..", "data", "matterhorn3d_$(name)_$(TAG).jls"))
runs = Dict(n => load(n) for n in ("P", "S_NS", "S_EW"))
d = runs["P"]
t, dt = d.t, d.t[2] - d.t[1]
const COMP = ("EW", "NS", "Z")
total(runs) = runs["P"].v .+ 1.7 .* (runs["S_NS"].v .+ runs["S_EW"].v)
V = total(runs)
orm = reshape_source(V, dt, 0.5, τ -> ormsby(τ), 4.0)              # Ormsby 0.3-3.5 Hz, centred at 4 s
rick = reshape_source(V, dt, 0.5, τ -> ricker(τ, 0.4), 4.0)        # Ricker 0.4 Hz
st(name) = findfirst(==(name), d.names)
summit, valley = st("NS+0000"), st("NS-2500")

"Amplitude spectrum, smoothed with a running mean over ± `w` [Hz]."
function spectrum(v; w=0.05)
    n = length(v)
    L = nextpow(2, 4n)
    A = abs.(SeismicWavePC.rfft([v; zeros(L - n)])) .* dt
    f = (0:length(A)-1) ./ (L * dt)
    k = max(1, round(Int, w / (f[2] - f[1])))
    return f, [sum(A[max(1, i - k):min(end, i + k)]) / length(max(1, i - k):min(length(A), i + k)) for i in eachindex(A)]
end

# --- numbers
println("Matterhorn 3D ($TAG, $(round(d.nodes / 1e6, digits=1)) M nodes): summit (NS+0000) / valley (NS-2500, 2.6 km), Ormsby source")
for c in 1:3
    f, S = spectrum(orm[:, c, summit])
    R = spectrum(orm[:, c, summit]; w=0.1)[2] ./ spectrum(orm[:, c, valley]; w=0.1)[2]   # ratio of spectra smoothed over ±0.1 Hz
    band = 0.3 .<= f .<= 3.5
    fb, Rb, Sb = f[band], R[band], S[band]
    k = argmax(Rb)
    peaks = [fb[i] for i in 2:length(Sb)-1 if Sb[i] > Sb[i-1] && Sb[i] > Sb[i+1] && Sb[i] > 0.5maximum(Sb)]
    @printf("  %-2s: max spectral ratio %.1f at %.2f Hz; ratio at 0.4/1.4/1.8 Hz: %.1f/%.1f/%.1f; summit spectral peaks (>50%%) at %s Hz; PGV ratio %.1f\n",
            COMP[c], Rb[k], fb[k], (Rb[argmin(abs.(fb .- q))] for q in (0.4, 1.4, 1.8))..., join((@sprintf("%.2f", p) for p in peaks), ", "),
            maximum(abs, orm[:, c, summit]) / maximum(abs, orm[:, c, valley]))
end
env = abs.(orm[:, 2, summit])
tp = t[argmax(env)]
tl = t[findlast(>(0.1maximum(env)), env)]
@printf("  NS summit: peak at %.1f s, above 10 %% of the peak until %.1f s (%.1f s of ringing)\n", tp, tl, tl - tp)
for (n, r) in runs                                                  # each incoming wave alone
    o = reshape_source(r.v, dt, 0.5, τ -> ormsby(τ), 4.0)
    @printf("  %-5s alone: PGV ratio summit/valley EW %.1f, NS %.1f, Z %.1f\n", n,
            (maximum(abs, o[:, c, summit]) / maximum(abs, o[:, c, valley]) for c in 1:3)...)
end

# --- Fig. 2: record sections
fig = Figure(size=(1100, 1100), fontsize=15)
for (j, (pre, lab)) in enumerate((("NS", "south → north"), ("EW", "west → east")))
    idx = [k for k in eachindex(d.names) if startswith(d.names[k], pre)]
    pos = [pre == "NS" ? d.y[k] : d.x[k] for k in idx]
    for c in 1:3
        ax = Axis(fig[c, j], title="$lab profile, $(COMP[c]) component", xlabel=c == 3 ? "distance from the summit (km)" : "",
                  ylabel="time (s)", yreversed=true)
        sc = maximum(abs, orm[:, c, idx]) / 0.15
        for (p, k) in zip(pos, idx)
            lines!(ax, p .+ orm[:, c, k] ./ sc, t, color=pre == "NS" ? :royalblue : :crimson, linewidth=0.6)
        end
        ylims!(ax, 16, 1)
    end
end
save(joinpath(@__DIR__, "matterhorn_3d_sections_$TAG.png"), fig)

# --- Fig. 4A-C: seismograms, spectra, spectral ratios
fig = Figure(size=(1200, 800), fontsize=15)
for c in 1:3
    ax = Axis(fig[c, 1], ylabel="$(COMP[c]) (rel.)", xlabel=c == 3 ? "time (s)" : "")
    sc = maximum(abs, orm[:, :, summit])
    lines!(ax, t, orm[:, c, summit] ./ sc, color=:black, label="summit")
    lines!(ax, t, orm[:, c, valley] ./ sc, color=:royalblue, label="valley")
    xlims!(ax, 0, 16); c == 1 && axislegend(ax, position=:rt)
    f, S = spectrum(orm[:, c, summit]); _, Q = spectrum(orm[:, c, valley])
    smax = maximum(spectrum(orm[:, 2, summit])[2])
    ax = Axis(fig[c, 2], ylabel="spectrum (rel.)", xlabel=c == 3 ? "frequency (Hz)" : "")
    band!(ax, f, 0 .* f, S ./ smax, color=(:gray, 0.6)); band!(ax, f, 0 .* f, Q ./ smax, color=(:royalblue, 0.6))
    xlims!(ax, 0, 4); ylims!(ax, 0, 1.05)
    ax = Axis(fig[c, 3], ylabel="spectral ratio", xlabel=c == 3 ? "frequency (Hz)" : "")
    b = 0.3 .<= f .<= 3.5
    R = spectrum(orm[:, c, summit]; w=0.1)[2] ./ spectrum(orm[:, c, valley]; w=0.1)[2]
    band!(ax, f[b], 0 .* f[b], R[b], color=(:salmon, 0.7)); xlims!(ax, 0, 4); ylims!(ax, 0, 12)
end
save(joinpath(@__DIR__, "matterhorn_3d_spectra_$TAG.png"), fig)

# --- Fig. 5A: amplification map (0.4 Hz Ricker), peak magnitude of the ground velocity normalized to its minimum
idx = [k for k in eachindex(d.names) if startswith(d.names[k], "M")]
pgv = [maximum(sqrt.(rick[:, 1, k] .^ 2 .+ rick[:, 2, k] .^ 2 .+ rick[:, 3, k] .^ 2)) for k in idx]
xs, ys = sort(unique(d.x[idx])), sort(unique(d.y[idx]))
A = fill(NaN, length(xs), length(ys))
for (k, p) in zip(idx, pgv)
    A[searchsortedfirst(xs, d.x[k]), searchsortedfirst(ys, d.y[k])] = p
end
A ./= minimum(A)
@printf("  map (0.4 Hz Ricker): max/min PGV %.1f at (%.2f, %.2f) km; summit %.1f\n", maximum(A), xs[argmax(A)[1]], ys[argmax(A)[2]],
        A[searchsortedfirst(xs, 0.0), searchsortedfirst(ys, 0.0)])
topo = matterhorn_surface()
fig = Figure(size=(800, 700), fontsize=15)
ax = Axis(fig[1, 1], aspect=1, xlabel="east (km)", ylabel="north (km)", title="amplification, 0.4 Hz Ricker (PGV / min)")
hm = heatmap!(ax, xs, ys, A, colormap=:turbo)
xf = -5:0.05:5
contour!(ax, xf, xf, [topo(a, b) for a in xf, b in xf], levels=2.0:0.25:4.5, color=(:white, 0.6), linewidth=0.7)
Colorbar(fig[1, 2], hm)
save(joinpath(@__DIR__, "matterhorn_3d_map_$TAG.png"), fig)
println("saved matterhorn_3d_{sections,spectra,map}_$TAG.png")
