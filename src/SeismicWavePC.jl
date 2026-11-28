"""
    SeismicWavePC

Julia version of the OpenSWPC finite-difference seismic wave propagation code (Maeda et al., 2017),
for 2D (P-SV, SH) and 3D visco-elastic media given as GeophysicalModelGenerator `CartData`, or built per MPI process
(`DistributedModel`).
Runs on CPUs and GPUs (KernelAbstractions) and on multiple MPI processes (ImplicitGlobalGrid).
"""
module SeismicWavePC

using KernelAbstractions
using GeophysicalModelGenerator: CartData, GeoData, ProjectionPoint, convert2CartData, write_paraview, movie_paraview, xyz_grid
import ImplicitGlobalGrid as IGG
import MPI
using Printf
using Serialization: serialize, deserialize
using FFTW: fft, ifft, fftfreq, rfft, irfft, plan_irfft
using SparseArrays: sparse
using LinearAlgebra: LinearAlgebra
import Random

include("viscoelastic.jl")
include("sources.jl")
include("finite_fault.jl")
include("source_files.jl")
include("medium.jl")
include("kernels.jl")
include("source_window.jl")
include("interpolation.jl")
include("pml.jl")
include("planewave.jl")
include("injection.jl")
include("snapshots.jl")
include("simulation.jl")
include("derivatives.jl")
include("curvilinear.jl")
include("curvilinear3d.jl")
include("curvilinear_snapshots.jl")
include("files.jl")
include("layered.jl")
include("green.jl")
include("randommedium.jl")
include("misfits.jl")
include("gradient.jl")
include("adjoint.jl")
include("born.jl")
include("curvilinear_adjoint.jl")
include("curvilinear_teleseismic.jl")
include("anisotropy.jl")
include("inversion.jl")
include("analytic.jl")

export MomentTensorSource, DoubleCoupleSource, ForceSource, finite_fault, read_sources, read_srf, read_cmtsolution, ExplosionSource, PlaneWave, TeleseismicWave, Station, Seismograms, simulate, moment_rate, SampledSTF, BerlageSTF, KlauderSTF, CPU, DistributedModel
export curvilinear_mesh
export ReciprocalGreen, reciprocal_green, seismogram, Snapshots, random_medium, perturb, read_random_medium, misfit_gradient, born,
       L2Misfit, TravelTimeMisfit, EnvelopeMisfit, PhaseMisfit, TFPhaseMisfit, Filtered, bandpass, geographic_weights, garvin, tf_misfits, ModelParameters, initial_parameters, to_model, WaveformProblem, objective,
       objective_and_gradient, fg!
export read_netcdf, write_sac, read_sac, write_su, read_su, read_lhm, read_lgm, read_grd, flatten_earth, flatten_depth, read_snapshots, SnapshotSeries

end
