# Fig. S4 of Limberger et al. (2025) from our 3D ridge runs (matterhorn_3d.jl metal 0.06 ridges): receivers along the two
# ridge profiles through the summit (read from the paper's map: NW-SE along x = -0.885 y and WSW-ENE along y = 0.39 x,
# 100 m apart) on the shaded-relief map, and record sections of the three components, each subfigure normalized to its
# maximum (P + 1.7 (SV_NS + SV_EW), Ormsby source; the time axis shifted as in Fig. 2/4, paper_style_3d.jl). With
# MATTERHORN_PRIVATE and the paper's figure in its paper_figs/figS4.png, also a comparison page there.
#
#   julia --project=examples/matterhorn examples/matterhorn/plot_s4.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
include("compare.jl")
include("relief.jl")
const TSH = 1.20             # [s] the shift of paper_style_3d.jl (its valley EW peak at 4.4 s, as the paper's Fig. 4A)

runs = Dict(n => deserialize(joinpath(@__DIR__, "..", "data", "matterhorn3d_ridges_$(n)_60m.jls")) for n in ("P", "S_NS", "S_EW"))
d = runs["P"]
t, dt = d.t, d.t[2] - d.t[1]
orm = reshape_source(runs["P"].v .+ 1.7 .* (runs["S_NS"].v .+ runs["S_EW"].v), dt, 0.5, τ -> ormsby(τ), 4.0)
tp = t .- TSH
# signed distance along the profile from the summit [m] (south/west negative)
dist(k) = 1e3 * sign(startswith(d.names[k], "RNS") ? d.y[k] : d.x[k]) * hypot(d.x[k], d.y[k])

fig = Figure(size=(1900, 1350), fontsize=22, backgroundcolor=:white)
axm, crm = relief_map!(fig[1:3, 1]; half=5.5)
axm.title = "Profiles along ridges"; axm.titlesize = 30
ns, ew = [k for k in eachindex(d.names) if startswith(d.names[k], "RNS")], [k for k in eachindex(d.names) if startswith(d.names[k], "REW")]
scatter!(axm, d.x[ns], d.y[ns], color=:blue, markersize=8, label="Receiver line NS")
scatter!(axm, d.x[ew], d.y[ew], color=:red, markersize=8, label="Receiver line EW")
axislegend(axm, position=:rt)
Colorbar(fig[1:3, 2], colormap=RELIEF_CMAP, colorrange=crm, label="Elevation (m)", height=Relative(0.45))
for (j, (sel, col, ends)) in enumerate(((ns, :blue, ("S", "N")), (ew, :red, ("W", "E"))))
    sel = sel[sortperm(dist.(sel))]
    for c in 1:3
        ax = Axis(fig[c, 2+j], yreversed=true, ylabel=j == 1 ? "Time (s)" : "", xlabel=c == 3 ? "Distance from peak (m)" : "",
                  title=c == 1 ? (j == 1 ? "south-to-north profile" : "west-to-east profile") : "")
        scl = maximum(abs, orm[:, c, sel]) / 700
        for k in sel
            lines!(ax, dist(k) .+ orm[:, c, k] ./ scl, tp, color=(col, 0.6), linewidth=0.8)
        end
        L = maximum(abs, dist.(sel)) + 300
        xlims!(ax, -L, L); ylims!(ax, 13, 3)
        text!(ax, -0.95L, 3.3, text=ends[1], fontsize=34, align=(:left, :top)); text!(ax, 0.95L, 3.3, text=ends[2], fontsize=34, align=(:right, :top))
        c < 3 && hidexdecorations!(ax, grid=false); j == 2 && hideydecorations!(ax, grid=false)
        j == 2 && Label(fig[c, 5], "$(("EW", "NS", "Z")[c]) component", rotation=-π / 2, fontsize=28, tellheight=false)
    end
end
colsize!(fig.layout, 1, Relative(0.4))
ours = joinpath(@__DIR__, "matterhorn_figS4_60m.png"); save(ours, fig); println("saved $ours")
paper = joinpath(PRIVATE, "paper_figs", "figS4.png")
if isfile(paper)
    notes = wrap("Ours: 3D, curvilinear SBP-SAT, 60 m nodes ($(round(d.nodes / 1e6, digits=1)) M), Float32 on the GPU; receivers every " *
                 "100 m (in map coordinates) along the two ridge lines read from the paper's map (NW-SE: x = -0.885 y; WSW-ENE: " *
                 "y = 0.39 x; the distance is along the line); vertical plane waves P + 1.7 (SV_NS + SV_EW), Ormsby source; each " *
                 "subfigure normalized to its maximum; time axis shifted by -1.2 s as in Figs. 2 and 4. Map: the raw 10 m DEM " *
                 "(the simulation uses it smoothed with σ = 40 m).")
    println("saved ", compare_page(paper, ours, "compare_figS4.png"; top="Limberger et al. (2025), Fig. S4 (Salvus)",
                                   bottom="SeismicWavePC, curvilinear SBP, 60 m grid", notes))
end
