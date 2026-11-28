# The DEM of the Matterhorn region: swissALTIRegio (swisstopo, 10 m, LV95 / EPSG:2056; it also covers the Italian side),
# downloaded as 10 km tiles (ASCII xyz, from the swisstopo STAC API, data.geo.admin.ch) and cached in examples/data/.
# `matterhorn_dem()` returns (x, y, z) in km, with x east, y north and the origin at the summit (the highest point), and
# the LV95 coordinates and height of the summit [m].
using Downloads, Serialization

const DEM_DIR = joinpath(@__DIR__, "..", "data", "swissaltiregio")
tile_url(e, n) = "https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio_$(e)-$(lpad(n, 4, '0'))/" *
                 "swissaltiregio_$(e)-$(lpad(n, 4, '0'))_2056_5728.xyz.zip"

"The heights [m] of the 10 m tile with the lower-left corner (e, n) [km, LV95]: Dict((E, N) => z) of the cell centres."
function read_tile!(Z, e, n)
    zip = joinpath(DEM_DIR, "$(e)_$(n).xyz.zip")
    isfile(zip) || (mkpath(DEM_DIR); Downloads.download(tile_url(e, n), zip))
    xyz = joinpath(DEM_DIR, "$(e)_$(lpad(n, 4, '0')).xyz")
    isfile(xyz) || run(`unzip -o -q $zip -d $DEM_DIR`)
    for line in Iterators.drop(eachline(xyz), 1)
        a, b, c = split(line)
        Z[(parse(Int, a), parse(Int, b))] = parse(Float64, c)
    end
    return Z
end

function matterhorn_dem(; half=10.0, center=(2617050, 1091450), cache=joinpath(DEM_DIR, "..", "matterhorn_dem.jls"))
    if !isfile(cache)
        h = round(Int, 1000half)
        e1, e2 = center[1] - h, center[1] + h
        n1, n2 = center[2] - h, center[2] + h
        Z = Dict{Tuple{Int,Int},Float64}()
        for e in (fld(e1 - 5000, 10000) * 10 + 5):10:(fld(e2 - 5000, 10000) * 10 + 5),
            n in (fld(n1 - 4000, 10000) * 10 + 4):10:(fld(n2 - 4000, 10000) * 10 + 4)
            read_tile!(Z, e, n)
        end
        E = collect((cld(e1 - 5, 10) * 10 + 5):10:e2)            # cell centres end in 5
        N = collect((cld(n1 - 5, 10) * 10 + 5):10:n2)
        H = [get(Z, (a, b), NaN) for a in E, b in N]
        any(isnan, H) && error("DEM: $(count(isnan, H)) missing points in the window")
        serialize(cache, (; E, N, Z=H))
    end
    d = deserialize(cache)
    k = argmax(d.Z)
    e0, n0 = d.E[k[1]], d.N[k[2]]
    return (x=(d.E .- e0) ./ 1e3, y=(d.N .- n0) ./ 1e3, z=d.Z ./ 1e3, summit=(e0, n0, d.Z[k]))
end
