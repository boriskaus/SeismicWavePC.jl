# Fig. 7 of Limberger et al. (2025) from the cached results of matterhorn_fig7.jl, drawn in the style of the paper (same
# layout, axes and linear colour scales, scenario A / B labels), and a side-by-side comparison with the paper's figure
# (CC BY 4.0) if its image is given.
#
#   julia --project=examples/matterhorn examples/matterhorn/fig7_paper_style.jl [dx_km] [paper_fig7.png|none] [z0|halfspace] [plus|minus] [main|opposite]
#
# plus/minus: the relative polarity of the P and SV waves (P ± 1.7 SV; not stated in the paper); main/opposite: the
# names of the output files (the polarity that matches the paper best is the main figure).
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
include("compare.jl")
const DX7 = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.04
const PAPER = length(ARGS) >= 2 && ARGS[2] != "none" ? ARGS[2] : nothing
const NA7, SIG7 = 20, 0.1                                        # as in matterhorn_fig7.jl
const ANGLE_AT = length(ARGS) >= 3 ? ARGS[3] : "z0"              # 45° at z = 0 (as the paper) or in the half-space
const A45 = ANGLE_AT == "z0" ? asind(sind(45) * vp_of(Z_BOT - 1.0) / vp_of(0.0)) : 45.0
const POL = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :plus
const ROLE = length(ARGS) >= 5 ? ARGS[5] : "main"
tag = "$(round(Int, 1e3DX7))m_s$(round(Int, 1e3SIG7))"

# the region of the stored peaks (as in matterhorn_fig7.jl)
mesh, model, sec = section_model(DX7, NA7; σ=SIG7)
X, Z = mesh.x.val[:, 1, :], mesh.z.val[:, 1, :]
I = findall(x -> abs(x) <= 4.3, X[:, end])
k0 = findfirst(k -> minimum(Z[I, k]) >= 1.0, 1:size(Z, 2))
Xr, Zr = 1e3 .* X[I, k0:end], 1e3 .* Z[I, k0:end]
nI, nK = size(Xr)
top = nK
isum = argmin(abs.(Xr[:, top]))

cases = [(0.0, 90.0, 270.0, "Vertically incoming wave"), (A45, 90.0, 90.0, "45° from north"), (A45, 270.0, 270.0, "45° from south")]
R = [ANGLE_AT == "z0" ? getfield(deserialize(joinpath(@__DIR__, "..", "data", "matterhorn_fig7_$(round(Int, a))_$(round(Int, b))_$(tag)_pm.jls")), POL) :
     deserialize(joinpath(@__DIR__, "..", "data", "matterhorn_fig7_$(round(Int, a))_$(round(Int, b))_$tag.jls")) for (a, _, b, _) in cases]

const XOFF = 6000.0                                              # the paper's x: summit at 6000 m, south to north
verts = [vec(Xr) .+ XOFF vec(Zr)]
idx(i, k) = (k - 1) * nI + i
faces = vcat([[idx(i, k) idx(i + 1, k) idx(i + 1, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...,
             [[idx(i, k) idx(i + 1, k + 1) idx(i, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...)

fig = Figure(fontsize=34, backgroundcolor=:white)
pgv = (0.1, 1.0); dqr = (1.0, 40.0)
for (j, ((a, dir, _, title), r)) in enumerate(zip(cases, R))
    s = 0.01 / r.vh[isum, top]                                   # scenario A: summit horizontal PGV 1 cm/s
    for (row, (F, crange, cmap, arrowcol)) in enumerate(((100s .* r.vh, pgv, :batlowW, :white), (100s .* r.vv, pgv, :batlowW, :white),
                                                         (1e-3s .* r.dq, dqr, :vik, :black)))
        ax = Axis(fig[row, j], title=row == 1 ? title : "", titlesize=38, xlabel=row >= 2 ? "South-to-north direction (m)" : "",
                  ylabel=j == 1 ? "Elevation (m)" : "", xticks=5000:500:7000, yticks=3000:500:4500,
                  xticklabelsvisible=row >= 2, yticklabelsvisible=j == 1, aspect=DataAspect(),  # equal scales, as the paper
                  width=560, height=560 * 1700 / 2800)
        mesh!(ax, verts, faces, color=clamp.(vec(F), crange...), colormap=cmap, colorrange=crange, shading=NoShading)
        lines!(ax, Xr[:, top] .+ XOFF, Zr[:, top], color=:black, linewidth=1.2)
        xlims!(ax, 4700, 7500); ylims!(ax, 2800, 4500)
        text!(ax, 4760, 4480, text="($(('A':'I')[3(row-1)+j]))", align=(:left, :top), fontsize=34)
        if row != 2
            if a == 0                                            # arrows as in the paper
                arrows2d!(ax, collect(5450:150:6650), fill(3080.0, 9), zeros(9), fill(130.0, 9), color=arrowcol)
                text!(ax, 6050, 3060, text="Incidence angle 0°", align=(:center, :top), color=arrowcol, fontsize=28)
            else
                sgn = dir == 90.0 ? -1 : 1                       # from the north: travelling south (−x) and up
                base = sgn < 0 ? (6600.0, 2880.0) : (5400.0, 2880.0)
                p = (1.0, -sgn * 1.0) ./ sqrt(2)                 # along the wave front
                xs = [base[1] + t * 110 * p[1] for t in 0:4]; zs = [base[2] + (sgn < 0 ? t : 4 - t) * 110 * abs(p[2]) for t in 0:4]
                lines!(ax, [xs[1], xs[end]], [zs[1], zs[end]], color=arrowcol, linewidth=1.5)
                arrows2d!(ax, xs, zs, fill(sgn * 110.0, 5), fill(110.0, 5), color=arrowcol)
                text!(ax, base[1] + sgn * 40, 2830, text="45°", align=(:center, :bottom), color=arrowcol, fontsize=28)
            end
        end
    end
end
Colorbar(fig[1:2, 4], colormap=:batlowW, colorrange=pgv, ticks=([0.1, 1.0], ["0.1 / 3", "1 / 30"]), label="PGV in cm/s (scenario A / B)")
Colorbar(fig[3, 4], colormap=:vik, colorrange=dqr, ticks=([1, 10, 20, 30, 40], ["1 / 30", "10 / 300", "20 / 600", "30 / 900", "40 / 1200"]),
         label="dQ in kPa (A / B)")
Label(fig[1, 5], "Horizontal", rotation=-π / 2, fontsize=38, tellheight=false)
Label(fig[2, 5], "Vertical", rotation=-π / 2, fontsize=38, tellheight=false)
Label(fig[3, 5], "dQ", rotation=-π / 2, fontsize=38, tellheight=false)
rowgap!(fig.layout, 8); colgap!(fig.layout, 8)
resize_to_layout!(fig)
ours = joinpath(@__DIR__, "matterhorn_fig7_paperstyle_$(tag == "40m_s100" ? "40m" : tag)$(ANGLE_AT == "z0" ? "" : "_halfspace")$(ROLE == "main" ? "" : "_opposite").png")
save(ours, fig)
println("saved $ours")

# the comparison page: the paper's figure above ours, with a note on the resolution of both (from the paper's text and
# our runs)
const NOTES = wrap("Resolution. Paper (Salvus, spectral elements of order 4): ≥ 2 elements per minimum wavelength (λmin ≈ 504 m: vs 1765 m/s " *
    "at 3.5 Hz), elements < 200 m in 3D; in 2D the mesh lines of its Fig. S2 are ≈ 60 m apart (≈ 240 m elements if they are GLL " *
    "sub-cells); DEM \"slightly\" smoothed (amount not given); no convergence test reported. Ours (curvilinear SBP-SAT, 7th-order " *
    "upwind operators): 40 m node spacing (12.6 nodes per λmin), DEM smoothed with σ = 100 m (summit of the section at 4.30 km). " *
    "Converged: a 30 m run changes the surface dQ maxima by 1-3 kPa (≤ 10 %; 30 m check at 45° in the half-space). With σ = 40 m the " *
    "stresses at the sharp north-side corner changed with the grid (60 vs 40 m: up to 2×), hence σ = 100 m. Oblique waves: 45° at " *
    "z = 0 as in the paper (56.7° in the half-space below −2 km). P and SV waves summed as P $(POL == :plus ? "+" : "−") 1.7 SV (the " *
    "relative polarity is not stated in the paper).")
if PAPER !== nothing
    file = joinpath(@__DIR__, "compare_fig7$(ANGLE_AT == "z0" ? "" : "_halfspace")$(ROLE == "main" ? "" : "_opposite").png")
    file = compare_page(PAPER, ours, file; top="Limberger et al. (2025), Fig. 7 (Salvus)",
                 bottom="SeismicWavePC, curvilinear SBP, $(round(Int, 1e3DX7)) m grid", notes=NOTES)
    println("saved $file")
end
