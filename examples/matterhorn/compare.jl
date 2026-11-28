# Comparison pages: the paper's figure on top, ours below at the same width (A4 portrait: 2000 px ≈ 17 cm), headers, the
# CC BY 4.0 attribution of the paper's figure and notes at the bottom. Used by the figure scripts of this example.
using CairoMakie, GeophysicalModelGenerator

# The pages contain the paper's figures: they are written to a private folder outside the repository (environment variable
# MATTERHORN_PRIVATE; default examples/data, git-ignored), never into the repository, which holds only our own material
# (an open package). The paper's figures are expected in its subfolder paper_figs.
const PRIVATE = get(ENV, "MATTERHORN_PRIVATE", joinpath(@__DIR__, "..", "data"))
const COMPARE_DIR = joinpath(PRIVATE, "comparisons")
paper_fig(n) = joinpath(PRIVATE, "paper_figs", "fig$n.png")

const ATTRIBUTION = "Top: Limberger, Rümpker, Kruse & Duretz (2025), Sci. Rep. 15, 23914, doi:10.1038/s41598-025-08218-5, CC BY 4.0" *
                    " (cropped to the Matterhorn panels where marked)."

"""
    compare_page(paper, ours, file; top, bottom, notes=String[], width=2000, crop=nothing)

Stack the image file `paper` (optionally cropped: `crop = (rows, columns)` in its pixels) above the image file `ours`, both
scaled to `width` pixels, with the headers `top` and `bottom`, the attribution and `notes` (one line each, wrapped by the
caller) below. Saves `file` (its name) in `COMPARE_DIR`.
"""
function compare_page(paper, ours, file; top, bottom, notes=String[], width=2000, crop=nothing)
    file = joinpath(mkpath(COMPARE_DIR), basename(file))
    P = load(paper)
    crop === nothing || (P = P[crop...])
    O = load(ours)
    hp, ho = round(Int, width * size(P, 1) / size(P, 2)), round(Int, width * size(O, 1) / size(O, 2))
    fig = Figure(backgroundcolor=:white, figure_padding=(30, 30, 20, 20))
    for (r, (img, h, head)) in enumerate(((P, hp, top), (O, ho, bottom)))
        Label(fig[2r-1, 1], head, fontsize=34, font=:bold, halign=:left)
        ax = Axis(fig[2r, 1], aspect=DataAspect(), width=width, height=h)
        hidedecorations!(ax); hidespines!(ax)
        image!(ax, rotr90(img))
    end
    for (q, line) in enumerate([wrap(ATTRIBUTION); notes])
        Label(fig[4+q, 1], line, fontsize=29, halign=:left, justification=:left)
    end
    rowgap!(fig.layout, 6)
    resize_to_layout!(fig)
    save(file, fig)
    return file
end

"Wrap a long note into lines of at most `n` characters (at spaces)."
function wrap(text, n=125)
    lines, cur = String[], ""
    for w in split(text, ' ')
        if length(cur) + length(w) + 1 > n && !isempty(cur)
            push!(lines, cur); cur = String(w)
        else
            cur = isempty(cur) ? String(w) : cur * " " * w
        end
    end
    isempty(cur) || push!(lines, cur)
    return lines
end
