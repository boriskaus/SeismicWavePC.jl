# Curvilinear solver: method and related work

The curvilinear solver was written for SeismicWavePC; unlike the staggered-grid solver it is not a port of an existing
code. It follows no single paper. It combines established ingredients of summation-by-parts (SBP) finite differences with
simultaneous approximation terms (SAT); this page lists them and relates the solver to published SBP-SAT methods for seismic
waves. How to use the solver: [Topography on a curvilinear grid](@ref).

## The scheme

- **Equations.** The first-order velocity–stress form of the visco-elastic wave equation (generalized Zener body and VTI
  anisotropy as in the staggered solver; TTI by rotating the strain rate into the frame of the tilted axis and the stress
  back), transformed to a terrain-following mesh with vertical columns: x = ξ (and y = ψ),
  z = Z(ξ, η). With J = ∂Z/∂η and Jx = −∂Z/∂ξ the momentum equation is ρJ ∂t vᵢ = Dξ(J σᵢₓ) + Dη(Jx σᵢₓ + σᵢ_z), and the
  strain rates use ∂x = Dξ + (Jx/J) Dη, ∂z = Dη / J (3D: the same with ψ and Jy).
- **SBP operators.** Upwind *dual-pairing* operators D₊, D₋ with H D₊ + (H D₋)ᵀ = B (Mattsson, 2017), with the
  boundary-optimized closures of Mattsson, Niemelä & Winters (2027), order 7, from SummationByPartsOperators.jl (Ranocha,
  2021). The momentum equation uses D₋, the strain rates D₊; this pairing gives a discrete energy estimate, as for the
  dual-pairing elastic solvers of Duru, Fung & Williams (2022). In tests on a hill model the boundary-optimized closures gave
  the smallest errors near the free surface among the operators compared (Mattsson, 2017, orders 5 and 7; the
  dispersion-relation-preserving operators of Williams & Duru, 2024); central SBP operators (Mattsson & Nordström, 2004)
  let point sources excite spurious grid-scale modes.
- **Boundaries and interfaces (SAT).** Boundary and interface conditions are imposed weakly with penalty terms (Carpenter,
  Gottlieb & Abarbanel, 1994; reviews: Svärd & Nordström, 2014; Del Rey Fernández, Hicken & Zingg, 2014). The free surface
  removes the traction. The absorbing sides and bottom use impedance (characteristic) conditions, integrated implicitly
  (trapezoidal rule), so they do not limit the time step. Material interfaces are grid lines between blocks: half the
  traction mismatch enters the momentum equations and half the velocity mismatch the strain rates of both sides
  (welded). At fluid–solid interfaces only the normal velocity and normal traction are coupled and the tangential tractions
  are removed (slip); Lotto & Dunham (2015) couple a compressible ocean to an elastic solid with SBP-SAT as well.
- **Stability and time stepping.** With these terms the semi-discrete energy E = ½ Σ H (ρJ |v|² + J σ:C⁻¹σ) is conserved
  (or dissipated at the absorbing boundaries). The time integration is a leapfrog scheme; its time step follows from the
  spectral radius of the spatial operator (power iteration).
- **Absorbing layers.** `absorber = :pml` applies the complex-frequency-shifted PML of OpenSWPC (auxiliary differential
  equations, Zhang & Shen, 2010) to the ξ and η parts of the SBP operators. `absorber = :mpml` is the multiaxial PML
  (Meza-Fajardo & Papageorgiou, 2008) in its complex-frequency-shifted form, as formulated by Zhang, Zhang & Chen (2014) for
  elastic waves on curvilinear grids. Unlike the interior scheme, the layers are not covered by an energy estimate; for
  provably stable PMLs with SBP-SAT see Duru & Kreiss (2014) and Duru, Kozdon & Kreiss (2015).
- **Sources and stations.** Point sources and receivers use a discrete delta: a Kaiser-windowed sinc (Hicks, 2002) where the
  grid is uniform around the point, Lagrange weights near boundaries; stations with `surface = true` sit exactly on the
  topography.
- **Gradients.** The SBP identity H A = −(H J G)ᵀ between the momentum operator A and the strain-rate operator G (SAT terms
  included) gives the exact discrete adjoint by applying the forward operators, so misfit gradients and Born modelling are
  exact for the discrete scheme. Wiratama, Duru & Kim (2026) develop a space-time dual-pairing SBP framework for forward
  and adjoint wave equations.

## Related work

SBP-SAT methods for seismic waves on curvilinear, boundary-conforming meshes have a substantial literature:

- **Duru & Dunham (2016)** solve the first-order elastic wave equation with SBP operators on curvilinear meshes, with weakly
  imposed boundary and interface conditions and semi-discrete energy estimates, for dynamic ruptures on nonplanar faults.
- **Duru, Fung & Williams (2022)** use dual-pairing (upwind) SBP operators for large-scale elastic wave simulations in 3D
  complex geometries, the closest relative of the operators used here; Duru, Williams & Fung (2022, preprint) add
  dispersion-relation-preserving operators that avoid spurious high-frequency modes, for nonlinear ruptures.
- **Petersson & Sjögreen (2015)** discretize the anisotropic elastic wave equation in second-order form on curvilinear grids
  with SBP operators (the code SW4).
- **Almquist & Dunham (2021)** treat the general anisotropic elastic wave equation on curvilinear multiblock grids in
  second-order form, with weakly enforced boundary and interface conditions.
- **Kozdon, Dunham & Nordström (2012)** impose boundary and (frictional) interface conditions for elastodynamics weakly with
  SBP operators, and show the benefit of characteristic variables.
- **Lotto & Dunham (2015)** couple a compressible ocean to an elastic solid with SBP-SAT.
- **Duru & Kreiss (2014)** and **Duru, Kozdon & Kreiss (2015)** analyze the stability of PMLs for the elastic wave
  equation, with SBP-SAT discretizations.
- **O'Reilly et al. (2022)** use SBP operators on *staggered* curvilinear grids with topography (4th order, provably stable),
  for the GPU code AWP-ODC.
- **de la Puente et al. (2014)** take a different route: a fully staggered (Lebedev) grid deformed to the topography, with
  mimetic operators at the free surface (no energy estimate; 4× the memory of a standard staggered grid). Their test,
  Garvin's problem on tilted surfaces, is in [Validation](@ref).

The accuracy of the curvilinear solver on topography, against the staggered grid, is shown in [Validation](@ref)
(free surface on topography).

## References

Almquist, M., and Dunham, E. M. (2021). Elastic wave propagation in anisotropic solids using energy-stable finite
differences with weakly enforced boundary and interface conditions. *Journal of Computational Physics*, 424, 109842.
doi:10.1016/j.jcp.2020.109842

Carpenter, M. H., Gottlieb, D., and Abarbanel, S. (1994). Time-stable boundary conditions for finite-difference schemes
solving hyperbolic systems: methodology and application to high-order compact schemes. *Journal of Computational
Physics*, 111(2), 220–236. doi:10.1006/jcph.1994.1057

Del Rey Fernández, D. C., Hicken, J. E., and Zingg, D. W. (2014). Review of summation-by-parts operators with simultaneous
approximation terms for the numerical solution of partial differential equations. *Computers & Fluids*, 95, 171–196.
doi:10.1016/j.compfluid.2014.02.016

Duru, K., and Dunham, E. M. (2016). Dynamic earthquake rupture simulations on nonplanar faults embedded in 3D
geometrically complex, heterogeneous elastic solids. *Journal of Computational Physics*, 305, 185–207.
doi:10.1016/j.jcp.2015.10.021

Duru, K., Fung, F., and Williams, C. (2022). Dual-pairing summation by parts finite difference methods for large scale
elastic wave simulations in 3D complex geometries. *Journal of Computational Physics*, 454, 110966.
doi:10.1016/j.jcp.2022.110966

Duru, K., Kozdon, J. E., and Kreiss, G. (2015). Boundary conditions and stability of a perfectly matched layer for the
elastic wave equation in first order form. *Journal of Computational Physics*, 303, 372–395.
doi:10.1016/j.jcp.2015.09.048

Duru, K., and Kreiss, G. (2014). Boundary waves and stability of the perfectly matched layer for the two space
dimensional elastic wave equation in second order form. *SIAM Journal on Numerical Analysis*, 52(6), 2883–2904.
doi:10.1137/13093563x

de la Puente, J., Ferrer, M., Hanzich, M., Castillo, J. E., and Cela, J. M. (2014). Mimetic seismic wave modeling
including topography on deformed staggered grids. *Geophysics*, 79(3), T125–T141. doi:10.1190/geo2013-0371.1

Duru, K., Williams, C., and Fung, F. (2022). Accurate simulations of nonlinear dynamic shear ruptures on pre-existing
faults in 3D elastic solids with dual-pairing SBP methods. Preprint, arXiv:2207.07891.

Hicks, G. J. (2002). Arbitrary source and receiver positioning in finite-difference schemes using Kaiser windowed sinc
functions. *Geophysics*, 67(1), 156–166. doi:10.1190/1.1451454

Kozdon, J. E., Dunham, E. M., and Nordström, J. (2012). Interaction of waves with frictional interfaces using
summation-by-parts difference operators: weak enforcement of nonlinear boundary conditions. *Journal of Scientific
Computing*, 50(2), 341–367. doi:10.1007/s10915-011-9485-3

Lotto, G. C., and Dunham, E. M. (2015). High-order finite difference modeling of tsunami generation in a compressible
ocean from offshore earthquakes. *Computational Geosciences*, 19(2), 327–340. doi:10.1007/s10596-015-9472-0

Mattsson, K. (2017). Diagonal-norm upwind SBP operators. *Journal of Computational Physics*, 335, 283–310.
doi:10.1016/j.jcp.2017.01.042

Mattsson, K., Niemelä, O., and Winters, A. R. (2027). Boundary-optimized closures for diagonal-norm upwind SBP operators.
*Journal of Computational Physics*, 568, 115399. doi:10.1016/j.jcp.2026.115399

Mattsson, K., and Nordström, J. (2004). Summation by parts operators for finite difference approximations of second
derivatives. *Journal of Computational Physics*, 199(2), 503–540. doi:10.1016/j.jcp.2004.03.001

Meza-Fajardo, K. C., and Papageorgiou, A. S. (2008). A nonconvolutional, split-field, perfectly matched layer for wave
propagation in isotropic and anisotropic elastic media: stability analysis. *Bulletin of the Seismological Society of
America*, 98(4), 1811–1836. doi:10.1785/0120070223

O'Reilly, O., Yeh, T.-Y., Olsen, K. B., Hu, Z., Breuer, A., Roten, D., and Goulet, C. A. (2022). A high-order
finite-difference method on staggered curvilinear grids for seismic wave propagation applications with topography.
*Bulletin of the Seismological Society of America*, 112(1), 3–22. doi:10.1785/0120210096

Petersson, N. A., and Sjögreen, B. (2015). Wave propagation in anisotropic elastic materials and curvilinear coordinates
using a summation-by-parts finite difference method. *Journal of Computational Physics*, 299, 820–841.
doi:10.1016/j.jcp.2015.07.023

Ranocha, H. (2021). SummationByPartsOperators.jl: A Julia library of provably stable semidiscretization techniques with
mimetic properties. *Journal of Open Source Software*, 6(64), 3454. doi:10.21105/joss.03454

Svärd, M., and Nordström, J. (2014). Review of summation-by-parts schemes for initial–boundary-value problems. *Journal of
Computational Physics*, 268, 17–38. doi:10.1016/j.jcp.2014.02.031

Williams, C., and Duru, K. (2024). Full-spectrum dispersion relation preserving summation-by-parts operators. *SIAM
Journal on Numerical Analysis*, 62(4), 1565–1588. doi:10.1137/23m1586471

Wiratama, K., Duru, K., and Kim, Y. (2026). A space-time dual-pairing summation-by-parts framework for forward and
adjoint wave equations. Preprint, arXiv:2603.17014.

Zhang, W., and Shen, Y. (2010). Unsplit complex frequency-shifted PML implementation using auxiliary differential
equations for seismic wave modeling. *Geophysics*, 75(4), T141–T154. doi:10.1190/1.3463431

Zhang, Z., Zhang, W., and Chen, X. (2014). Complex frequency-shifted multi-axial perfectly matched layer for elastic wave
modelling on curvilinear grids. *Geophysical Journal International*, 198(1), 140–153. doi:10.1093/gji/ggu124
