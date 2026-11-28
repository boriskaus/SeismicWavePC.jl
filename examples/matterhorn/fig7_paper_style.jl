# Fig. 7 of Limberger et al. (2025) from the cached results of matterhorn_fig7.jl, drawn in the style of the paper (same
# layout, axes and linear colour scales, scenario A / B labels), and a side-by-side comparison with the paper's figure
# (CC BY 4.0) if its image is given.
#
#   julia --project=examples/matterhorn examples/matterhorn/fig7_paper_style.jl [dx_km] [paper_fig7.png]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Serialization, Printf
include("model.jl")
const DX7 = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.04
const PAPER = length(ARGS) >= 2 ? ARGS[2] : nothing
const NA7, SIG7 = 20, 0.1                                        # as in matterhorn_fig7.jl
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

cases = [(0.0, 90.0, 270.0, "Vertically incoming wave"), (45.0, 90.0, 90.0, "45° from north"), (45.0, 270.0, 270.0, "45° from south")]
R = [deserialize(joinpath(@__DIR__, "..", "data", "matterhorn_fig7_$(round(Int, a))_$(round(Int, b))_$tag.jls")) for (a, _, b, _) in cases]

const XOFF = 6000.0                                              # the paper's x: summit at 6000 m, south to north
verts = [vec(Xr) .+ XOFF vec(Zr)]
idx(i, k) = (k - 1) * nI + i
faces = vcat([[idx(i, k) idx(i + 1, k) idx(i + 1, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...,
             [[idx(i, k) idx(i + 1, k + 1) idx(i, k + 1)] for i in 1:nI-1 for k in 1:nK-1]...)

fig = Figure(size=(1500, 1180), fontsize=17, backgroundcolor=:white)
pgv = (0.1, 1.0); dqr = (1.0, 40.0)
for (j, ((a, dir, _, title), r)) in enumerate(zip(cases, R))
    s = 0.01 / r.vh[isum, top]                                   # scenario A: summit horizontal PGV 1 cm/s
    for (row, (F, crange, cmap, arrowcol)) in enumerate(((100s .* r.vh, pgv, :batlowW, :white), (100s .* r.vv, pgv, :batlowW, :white),
                                                         (1e-3s .* r.dq, dqr, :vik, :black)))
        ax = Axis(fig[row, j], title=row == 1 ? title : "", titlesize=21, xlabel=row >= 2 ? "South-to-north direction (m)" : "",
                  ylabel=j == 1 ? "Elevation (m)" : "", xticks=5000:500:7000, yticks=3000:500:4500,
                  xticklabelsvisible=row >= 2, yticklabelsvisible=j == 1)
        mesh!(ax, verts, faces, color=clamp.(vec(F), crange...), colormap=cmap, colorrange=crange, shading=NoShading)
        lines!(ax, Xr[:, top] .+ XOFF, Zr[:, top], color=:black, linewidth=1.2)
        xlims!(ax, 4700, 7500); ylims!(ax, 2800, 4500)
        text!(ax, 4760, 4480, text="($(('A':'I')[3(row-1)+j]))", align=(:left, :top), fontsize=19)
        if row != 2
            if a == 0                                            # arrows as in the paper
                arrows2d!(ax, collect(5450:150:6650), fill(3080.0, 9), zeros(9), fill(130.0, 9), color=arrowcol)
                text!(ax, 6050, 3060, text="Incidence angle 0°", align=(:center, :top), color=arrowcol, fontsize=14)
            else
                sgn = dir == 90.0 ? -1 : 1                       # from the north: travelling south (−x) and up
                base = sgn < 0 ? (6600.0, 2880.0) : (5400.0, 2880.0)
                p = (1.0, -sgn * 1.0) ./ sqrt(2)                 # along the wave front
                xs = [base[1] + t * 110 * p[1] for t in 0:4]; zs = [base[2] + (sgn < 0 ? t : 4 - t) * 110 * abs(p[2]) for t in 0:4]
                lines!(ax, [xs[1], xs[end]], [zs[1], zs[end]], color=arrowcol, linewidth=1.5)
                arrows2d!(ax, xs, zs, fill(sgn * 110.0, 5), fill(110.0, 5), color=arrowcol)
                text!(ax, base[1] + sgn * 40, 2830, text="45°", align=(:center, :bottom), color=arrowcol, fontsize=14)
            end
        end
    end
end
Colorbar(fig[1:2, 4], colormap=:batlowW, colorrange=pgv, ticks=([0.1, 1.0], ["0.1 / 3", "1 / 30"]), label="PGV in cm/s (scenario A / B)")
Colorbar(fig[3, 4], colormap=:vik, colorrange=dqr, ticks=([1, 10, 20, 30, 40], ["1 / 30", "10 / 300", "20 / 600", "30 / 900", "40 / 1200"]),
         label="dQ in kPa (A / B)")
Label(fig[1, 5], "Horizontal", rotation=-π / 2, fontsize=21, tellheight=false)
Label(fig[2, 5], "Vertical", rotation=-π / 2, fontsize=21, tellheight=false)
Label(fig[3, 5], "dQ", rotation=-π / 2, fontsize=21, tellheight=false)
rowgap!(fig.layout, 8); colgap!(fig.layout, 8)
ours = joinpath(@__DIR__, "matterhorn_fig7_paperstyle_$(tag == "40m_s100" ? "40m" : tag).png")
save(ours, fig)
println("saved $ours")

# side by side with the paper's figure
if PAPER !== nothing
    P = rotr90(load(PAPER)); O = rotr90(load(ours))
    h = 1100
    wp, wo = round(Int, h * size(P, 1) / size(P, 2)), round(Int, h * size(O, 1) / size(O, 2))
    comp = Figure(size=(wp + wo + 60, h + 140), backgroundcolor=:white, figure_padding=(20, 20, 10, 10))
    for (c, (img, w, head)) in enumerate(((P, wp, "Limberger et al. (2025), Fig. 7 (Salvus)"),
                                          (O, wo, "SeismicWavePC, curvilinear SBP, $(round(Int, 1e3DX7)) m grid")))
        ax = Axis(comp[2, c], aspect=DataAspect(), width=w, height=h)
        hidedecorations!(ax); hidespines!(ax)
        image!(ax, img)
        Label(comp[1, c], head, fontsize=26, font=:bold)
    end
    Label(comp[3, 1:2], "Left: Limberger, Rümpker, Kruse & Duretz (2025), Sci. Rep. 15, 23914, doi:10.1038/s41598-025-08218-5, CC BY 4.0.",
          fontsize=16, halign=:left)
    resize_to_layout!(comp)
    file = joinpath(@__DIR__, "matterhorn_fig7_comparison.png")
    save(file, comp)
    println("saved $file")
end
