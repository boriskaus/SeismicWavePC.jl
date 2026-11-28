# Figs. 2, 4A-C and 5A of Limberger et al. (2025) in the paper's layout, from our 3D runs (matterhorn_3d.jl, 60 m):
# receiver lines and record sections (Fig. 2B-E), summit and valley seismograms, spectra and spectral ratios (Fig. 4A-C),
# and the 0.4 Hz amplification map on the topography (Fig. 5A). Same processing as analyze_3d.jl (P + 1.7 (S_NS + S_EW),
# Ormsby or 0.4 Hz Ricker source). With MATTERHORN_PRIVATE and the paper's figures in its paper_figs/, also comparison
# pages (paper on top, ours below) there.
#
#   julia --project=examples/matterhorn examples/matterhorn/paper_style_3d.jl [60m]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
include("compare.jl")
include("relief.jl")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "60m"
load3d(name) = deserialize(joinpath(@__DIR__, "..", "data", "matterhorn3d_$(name)_$(TAG).jls"))
runs = Dict(n => load3d(n) for n in ("P", "S_NS", "S_EW"))
d = runs["P"]
t, dt = d.t, d.t[2] - d.t[1]
V = runs["P"].v .+ 1.7 .* (runs["S_NS"].v .+ runs["S_EW"].v)
"reshape_source over chunks of stations (all ~1800 at once: a zero-padded Float64 FFT of tens of GB)."
function reshape_chunks(V, w; chunk=100)
    out = similar(V, Float64)
    for k in Iterators.partition(axes(V, 3), chunk)
        out[:, :, k] .= reshape_source(Float64.(V[:, :, k]), dt, 0.5, w, 4.0)
    end
    return out
end
orm = reshape_chunks(V, τ -> ormsby(τ))
rick = reshape_chunks(V, τ -> ricker(τ, 0.4))
runs = nothing; GC.gc()
st(name) = findfirst(==(name), d.names)
summit, valley = st("NS+0000"), st("NS-2500")
# time axis: the paper gives neither the depth of its source plane nor its time origin; ours is shifted so that the peak of
# the valley EW trace is at 4.4 s, as in the paper's Fig. 4A (read by eye)
const TSH = t[argmax(abs.(orm[:, 1, valley]))] - 4.4
tp = t .- TSH
@printf("time axis shifted by %.2f s (valley EW peak at 4.4 s as in the paper)\n", -TSH)
const COMP = ("EW", "NS", "ZZ")
const RES = "3D, curvilinear SBP-SAT, 60 m nodes ($(round(d.nodes / 1e6, digits=1)) M), Float32 on the GPU; vertical plane " *
            "waves P + 1.7 (SV_NS + SV_EW) (the paper: equal x, y, z forces); DEM swissALTIRegio smoothed with σ = 40 m. " *
            "Paper: Salvus, elements < 200 m, order 4."

function spectrum(v; w=0.05)
    n = length(v); L = nextpow(2, 4n)
    A = abs.(SeismicWavePC.rfft([v; zeros(L - n)])) .* dt
    f = (0:length(A)-1) ./ (L * dt)
    k = max(1, round(Int, w / (f[2] - f[1])))
    return f, [sum(A[max(1, i - k):min(end, i + k)]) / length(max(1, i - k):min(length(A), i + k)) for i in eachindex(A)]
end
page(paper, ours, name, notes; crop=nothing) =
    isfile(paper) && println("saved ", compare_page(paper, ours, name; top="Limberger et al. (2025), $(replace(name, "compare_fig" => "Fig. ", ".png" => "")) (Salvus)",
                                                    bottom="SeismicWavePC, curvilinear SBP, $TAG grid", notes=wrap(notes), crop))

# --- Fig. 4A-C (Matterhorn): seismograms, spectra (grey summit, blue valley), spectral ratio summit/valley
fig = Figure(size=(1500, 1000), fontsize=22, backgroundcolor=:white)
sc = maximum(abs, orm[:, :, summit])
smax = maximum(c -> maximum(spectrum(orm[:, c, summit])[2]), 1:3)
for c in 1:3
    ax = Axis(fig[c, 1], ylabel="Rel. Amplitude ($(COMP[c]))", xlabel=c == 3 ? "Time (s)" : "", yticks=-1:0.5:1)
    lines!(ax, tp, orm[:, c, summit] ./ sc, color=:black, label="Summit"); lines!(ax, tp, orm[:, c, valley] ./ sc, color=:blue, label="Valley")
    xlims!(ax, 2, 15); ylims!(ax, -1.1, 1.1); c == 1 && axislegend(ax, position=:rt, framevisible=false)
    c < 3 && hidexdecorations!(ax, grid=false)
    f, S = spectrum(orm[:, c, summit]); _, Q = spectrum(orm[:, c, valley])
    ax = Axis(fig[c, 2], ylabel="Spec. amplitude ($(COMP[c]))", xlabel=c == 3 ? "Frequency (Hz)" : "")
    band!(ax, f, 0 .* f, S ./ smax, color=(:gray, 0.8)); lines!(ax, f, S ./ smax, color=:black, linewidth=1)
    band!(ax, f, 0 .* f, Q ./ smax, color=(:royalblue, 0.7)); lines!(ax, f, Q ./ smax, color=:blue, linewidth=1)
    xlims!(ax, 0, 4); ylims!(ax, 0, 1.05); c < 3 && hidexdecorations!(ax, grid=false)
    R = spectrum(orm[:, c, summit]; w=0.1)[2] ./ spectrum(orm[:, c, valley]; w=0.1)[2]
    b = 0.05 .<= f .<= 4.0
    ax = Axis(fig[c, 3], ylabel="Spectral Ratio ($(COMP[c]))", xlabel=c == 3 ? "Frequency (Hz)" : "")
    band!(ax, f[b], 0 .* f[b], R[b], color=(:salmon, 0.6)); lines!(ax, f[b], R[b], color=:red, linewidth=1)
    xlims!(ax, 0, 4); ylims!(ax, 0, 12); c < 3 && hidexdecorations!(ax, grid=false)
end
for (j, l) in enumerate(("(A)", "(B)", "(C)")); Label(fig[0, j], l, fontsize=30, halign=:left, tellwidth=false); end
ours = joinpath(@__DIR__, "matterhorn_fig4_$TAG.png"); save(ours, fig); println("saved $ours")
page(joinpath(PRIVATE, "paper_figs", "fig4.png"), ours, "compare_fig4.png",
     "Paper cropped to the Matterhorn panels (A-C). Ours: summit (0, 0) and valley 2.5 km south (the paper's valley receiver, " *
     "the square in its Fig. 2B), Ormsby source 0.3-3.5 Hz; amplitudes relative to the summit maximum, spectra smoothed over " *
     "±0.05 Hz (ratios ±0.1 Hz). Our time axis shifted by $(@sprintf("%.1f", -TSH)) s (valley EW peak at 4.4 s as in the paper; " *
     "its source depth and time origin are not given). " * RES; crop=(:, 1:1000))

# --- Fig. 5A: amplification factor (peak |v| / minimum, 0.4 Hz Ricker) on the topography
idx = [k for k in eachindex(d.names) if startswith(d.names[k], "M")]
pgv = [maximum(sqrt.(rick[:, 1, k] .^ 2 .+ rick[:, 2, k] .^ 2 .+ rick[:, 3, k] .^ 2)) for k in idx]
xs, ys = sort(unique(d.x[idx])), sort(unique(d.y[idx]))
A = fill(NaN, length(xs), length(ys))
for (k, p) in zip(idx, pgv); A[searchsortedfirst(xs, d.x[k]), searchsortedfirst(ys, d.y[k])] = p; end
A ./= minimum(A)
topo = matterhorn_surface()
xf = range(-5, 5, length=201)
function interp(x, y)
    i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1); j = clamp(searchsortedlast(ys, y), 1, length(ys) - 1)
    a = (x - xs[i]) / (xs[i+1] - xs[i]); b = (y - ys[j]) / (ys[j+1] - ys[j])
    return (1 - a) * (1 - b) * A[i, j] + a * (1 - b) * A[i+1, j] + (1 - a) * b * A[i, j+1] + a * b * A[i+1, j+1]
end
fig = Figure(size=(1300, 1000), fontsize=24, backgroundcolor=:white)
Label(fig[1, 1], "(A) Matterhorn (Switzerland)", fontsize=32, halign=:left, tellwidth=false)
ax = Axis3(fig[2, 1], azimuth=1.32π, elevation=0.2π, aspect=(1, 1, 0.3), xlabel="east (km)", ylabel="north (km)",
           zlabel="Elevation (m)", zlabeloffset=90, perspectiveness=0.3)
s = surface!(ax, xf, xf, [1e3topo(a, b) for a in xf, b in xf]; color=[interp(a, b) for a in xf, b in xf], colormap=:batlowW,
             colorrange=(1, 10), shading=NoShading)
text!(ax, 0.0, 0.0, 1e3topo(0.0, 0.0) + 300, text="Matterhorn", align=(:center, :bottom), color=:white, fontsize=24)
text!(ax, -3.8, -4.6, 2400.0, text="0.4 Hz", color=:white, fontsize=36)
Colorbar(fig[3, 1], s, vertical=false, label="Amplification factor", width=Relative(0.4), ticks=[1, 10])
ours = joinpath(@__DIR__, "matterhorn_fig5a_$TAG.png"); save(ours, fig); println("saved $ours")
@printf("Fig. 5A: max amplification %.1f at (%.2f, %.2f) km\n", maximum(A), xs[argmax(A)[1]], ys[argmax(A)[2]])
page(joinpath(PRIVATE, "paper_figs", "fig5.png"), ours, "compare_fig5.png",
     "Paper cropped to the Matterhorn panel (A). Ours: peak ground-velocity magnitude for a 0.4 Hz Ricker wavelet at receivers " *
     "every 250 m (10 × 10 km), normalized to the minimum, on the smoothed DEM (local coordinates, summit at the origin). " * RES;
     crop=(:, 1:1010))

# --- Fig. 2B-E: receiver lines on the DEM and record sections (each subfigure normalized to its maximum)
fig = Figure(size=(1900, 1350), fontsize=22, backgroundcolor=:white)
axm, crm = relief_map!(fig[1:3, 1])                              # (shaded relief, raw 10 m DEM; relief.jl)
axm.title = "(B)"; axm.titlealign = :left; axm.titlesize = 30
ns, ew = [k for k in eachindex(d.names) if startswith(d.names[k], "NS")], [k for k in eachindex(d.names) if startswith(d.names[k], "EW")]
scatter!(axm, d.x[ns], d.y[ns], color=:blue, markersize=8, label="Receiver line NS")
scatter!(axm, d.x[ew], d.y[ew], color=:red, markersize=8, label="Receiver line EW")
scatter!(axm, [0.0], [0.0], marker=:circle, color=:transparent, strokecolor=:black, strokewidth=2, markersize=22)
scatter!(axm, [0.0], [-2.5], marker=:rect, color=:transparent, strokecolor=:black, strokewidth=2, markersize=16)
axislegend(axm, position=:rt); Colorbar(fig[1:3, 2], colormap=RELIEF_CMAP, colorrange=crm, label="Elevation (m)", height=Relative(0.45))
for (j, (sel, pos, col, ends)) in enumerate(((ns, d.y, :blue, ("S", "N")), (ew, d.x, :red, ("W", "E"))))
    o = sortperm(pos[sel]); sel = sel[o]
    for c in 1:3
        ax = Axis(fig[c, 2+j], yreversed=true, ylabel=j == 1 ? "Time (s)" : "", xlabel=c == 3 ? "Distance from peak (m)" : "",
                  title=c == 1 ? (j == 1 ? "south-to-north profile" : "west-to-east profile") : "")
        scl = maximum(abs, orm[:, c, sel]) / 700                                     # (m of offset per unit; as the paper: traces overlap)
        for k in sel
            lines!(ax, 1e3pos[k] .+ orm[:, c, k] ./ scl, tp, color=(col, 0.6), linewidth=0.8)
        end
        xlims!(ax, -2800, 2800); ylims!(ax, 13, 3)
        text!(ax, -2650, 3.3, text=ends[1], fontsize=34, align=(:left, :top)); text!(ax, 2650, 3.3, text=ends[2], fontsize=34, align=(:right, :top))
        c < 3 && hidexdecorations!(ax, grid=false); j == 2 && hideydecorations!(ax, grid=false)
        j == 2 && Label(fig[c, 5], "$(("EW", "NS", "Z")[c]) component", rotation=-π / 2, fontsize=28, tellheight=false)
    end
end
for (c, l) in enumerate(("(C)", "(D)", "(E)")); Label(fig[c, 3, TopLeft()], l, fontsize=30); end
colsize!(fig.layout, 1, Relative(0.4))                           # (the map as large as the record sections allow)
ours = joinpath(@__DIR__, "matterhorn_fig2_$TAG.png"); save(ours, fig); println("saved $ours")
page(joinpath(PRIVATE, "paper_figs", "fig2.png"), ours, "compare_fig2.png",
     "Paper's panel A is a Google Earth view (not reproduced). Map (B): the raw 10 m DEM, shaded relief (the simulation uses " *
     "it smoothed with σ = 40 m). Ours: receivers every 100 m along the N-S and W-E lines " *
     "through the summit (local coordinates), Ormsby source; traces of each subfigure normalized to its maximum; time axis " *
     "shifted by $(@sprintf("%.1f", -TSH)) s as in Fig. 4. " * RES)
