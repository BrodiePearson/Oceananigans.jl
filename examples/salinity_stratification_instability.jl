# # Langmuir turbulence example
#
# This example implements the Langmuir turbulence simulation reported in section
# 4 of
#
# > [McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of Fluid Mechanics (1997)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/langmuir-turbulence-in-the-ocean/638FD0E368140E5972144348DB930A38).
#
# This example demonstrates
#
#   * How to run large eddy simulations with surface wave effects via the
#     Craik-Leibovich approximation.
#
#   * How to specify time- and horizontally-averaged output.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

using Oceananigans
using Oceananigans.Units: minute, minutes, hours

using SeawaterPolynomials.TEOS10

FT = Float64

# ## Model set-up
#
# To build the model, we specify the grid, Stokes drift, boundary conditions, and
# Coriolis parameter.
#
# ### Domain and numerical grid specification
#
# We create a grid with modest resolution. The grid extent is similar, but not
# exactly the same as that in McWilliams et al. (1997).

grid = RegularRectilinearGrid(size=(32, 32, 24), extent=(128, 128, 48))

# ### The Stokes Drift profile
#
# The surface wave Stokes drift profile prescribed in McWilliams et al. (1997)
# corresponds to a 'monochromatic' (that is, single-frequency) wave field.
#
# A monochromatic wave field is characterized by its wavelength and amplitude
# (half the distance from wave crest to wave trough), which determine the wave
# frequency and the vertical scale of the Stokes drift profile.

using Oceananigans.BuoyancyModels: g_Earth

 amplitude = 0.8 # m
wavelength = 60 # m
wavenumber = 2π / wavelength # m⁻¹
 frequency = sqrt(g_Earth * wavenumber) # s⁻¹

## The vertical scale over which the Stokes drift of a monochromatic surface wave
## decays away from the surface is `1/2wavenumber`, or
const vertical_scale = wavelength / 4π

## Stokes drift velocity at the surface
const Uˢ = amplitude^2 * wavenumber * frequency * 0.0 # m s⁻¹

# The `const` declarations ensure that Stokes drift functions compile on the GPU.
# To run this example on the GPU, write `architecture = GPU()` in the constructor
# for `IncompressibleModel` below.
#
# The Stokes drift profile is

uˢ(z) = Uˢ * exp(z / vertical_scale)

# which we'll need for the initial condition.
#
# !!! info "The Craik-Leibovich equations in Oceananigans"
#     Oceananigans implements the Craik-Leibovich approximation for surface wave effects
#     using the _Lagrangian-mean_ velocity field as its prognostic momentum variable.
#     In other words, `model.velocities.u` is the Lagrangian-mean ``x``-velocity beneath surface
#     waves. This differs from models that use the _Eulerian-mean_ velocity field
#     as a prognostic variable, but has the advantage that ``u`` accounts for the total advection
#     of tracers and momentum, and that ``u = v = w = 0`` is a steady solution even when Coriolis
#     forces are present. See the
#     [physics documentation](https://clima.github.io/OceananigansDocumentation/stable/physics/surface_gravity_waves/)
#     for more information.
#
# The vertical derivative of the Stokes drift is

∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

# Finally, we note that the time-derivative of the Stokes drift must be provided
# if the Stokes drift changes in time. In this example, the Stokes drift is constant
# and thus the time-derivative of the Stokes drift is 0.

# ### Boundary conditions
#
# At the surface at ``z=0``, McWilliams et al. (1997) impose a wind stress
# on ``u``,

Qᵘ = -3.72e-5 # m² s⁻², surface kinematic momentum flux

u_boundary_conditions = UVelocityBoundaryConditions(grid, top = FluxBoundaryCondition(Qᵘ))

# McWilliams et al. (1997) impose a linear buoyancy gradient `N²` at the bottom
# along with a weak, destabilizing flux of buoyancy at the surface to faciliate
# spin-up from rest.

Qᵀ = 0.0 #1e-5 # W m⁻²
Qˢ = -1e-7

#N²_T = α * 1.936e-5 # s⁻², initial and bottom buoyancy gradient

# Define approximate expansion.contraction coefficients using the linear EOS values
α = 0.000167
β = 0.00078
g = 9.80665

N² = 1.936e-5 # s⁻², initial and bottom buoyancy gradient
Tᶻ_from_N² = -0.5*N²/α/g # Slightly Unstable
Sᶻ_from_N² = -1.5*N²/β/g # Very stable

T_boundary_conditions = TracerBoundaryConditions(grid, top = FluxBoundaryCondition(Qᵀ),
                                                        bottom = GradientBoundaryCondition(Tᶻ_from_N²))
nothing # hide

S_boundary_conditions = TracerBoundaryConditions(grid, top = FluxBoundaryCondition(Qˢ),
                                                    bottom = GradientBoundaryCondition(Sᶻ_from_N²))
nothing # hide

# !!! info "The flux convention in Oceananigans"
#     Note that Oceananigans uses "positive upward" conventions for all fluxes. In consequence,
#     a negative flux at the surface drives positive velocities, and a positive flux of
#     buoyancy drives cooling.

# ### Coriolis parameter
#
# McWilliams et al. (1997) use

coriolis = FPlane(f=1e-4) # s⁻¹

# which is typical for mid-latitudes on Earth.

# ## Model instantiation
#
# We are ready to build the model. We use a fifth-order Weighted Essentially
# Non-Oscillatory (WENO) advection scheme and the `AnisotropicMinimumDissipation`
# model for large eddy simulation. Because our Stokes drift does not vary in ``x, y``,
# we use `UniformStokesDrift`, which expects Stokes drift functions of ``z, t`` only.

model = IncompressibleModel(
           architecture = CPU(),
              advection = WENO5(),
            timestepper = :RungeKutta3,
                   grid = grid,
                #tracers = :b,
               #buoyancy = BuoyancyTracer(),
               buoyancy = SeawaterBuoyancy(gravitational_acceleration = g_Earth,
                           equation_of_state = TEOS10EquationOfState(FT),
                           constant_temperature = false, constant_salinity = false),
               coriolis = coriolis,
                closure = AnisotropicMinimumDissipation(),
           stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
    boundary_conditions = (u=u_boundary_conditions, T=T_boundary_conditions,
                            S=S_boundary_conditions),
)

# ## Initial conditions
#
# We make use of random noise concentrated in the upper 4 meters
# for buoyancy and velocity initial conditions,

Ξ(z) = randn() * exp(z / 4)
nothing # hide

# Create temperature and salinity profiles that are well-mixed near the surface,
# and increase/decrease below a certain depth (15 m)
# First create Heaviside and Interbal functions
function heaviside(t)
   0.5 * (sign(t) + 1)
end
function interval(t, a, b)
   heaviside(t-a) - heaviside(t-b)
end

#α = model.buoyancy.model.equation_of_state.α
#β = model.buoyancy.model.equation_of_state.β
#g = model.buoyancy.model.gravitational_acceleration

# Stable T (should be combined with Unstable or constant S below)
#Tᵢ(x, y, z) =  29.0 - heaviside(-z-15)*sqrt(abs(z+15))*β/(α * 10)
# Constant S (should be combined with Stable or Constant T above)
#Sᵢ(x, y, z) = 29.0

initial_mixed_layer_depth = 15 # m
T_stratification(z) = z < - initial_mixed_layer_depth ? Tᶻ_from_N² * z : Tᶻ_from_N²  * (-initial_mixed_layer_depth)
S_stratification(z) = z < - initial_mixed_layer_depth ? Sᶻ_from_N² * z : Sᶻ_from_N²  * (-initial_mixed_layer_depth)

# Constant T (should be combined with Stable or Constant S below)
#Tᵢ(x, y, z) = 29.0
#Tᵢ(x, y, z) =  29.0 + 0.5*heaviside(-z-15)*sqrt(abs(z+15))*β/(α * 10)
# Stable S (should be combined with Unstable or constant T above)
#Sᵢ(x, y, z) = 29.0 + heaviside(-z-15)*sqrt(abs(z+15))/10

#Tᵢ(x, y, z) = -1.5 #- 0.5*z*3/40 # *β/α
#Sᵢ(x, y, z) = 29.0 - z *3/40

Sᵢ(x, y, z) = S_stratification(z) + 29.0 + 1e-1 * Ξ(z) * Sᶻ_from_N² * model.grid.Lz
Tᵢ(x, y, z) = T_stratification(z) + -1.5 + 1e-1 * Ξ(z) * Tᶻ_from_N² * model.grid.Lz


# Unstable T (should be combined with Stable S below)
#Tᵢ(x, y, z) =  29.0 + heaviside(-z-15)*sqrt(abs(z+15))*β/(α * 10)
# Unstable S (should be combined with Stable T above)
#Sᵢ(x, y, z) = 29.0 - heaviside(-z-15)*sqrt(abs(z+15))/10
nothing # hide

# The velocity initial condition in McWilliams et al. (1997) is zero *Eulerian-mean* velocity.
# This means that we must add the Stokes drift profile to the Lagrangian-mean ``u`` velocity field
# modeled by Oceananigans.jl. We also add noise scaled by the friction velocity to ``u`` and ``w``.

uᵢ(x, y, z) = uˢ(z) + sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

wᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, T=Tᵢ, S=Sᵢ)

# ## Setting up the simulation
#
# We use the `TimeStepWizard` for adaptive time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 1.0,

wizard = TimeStepWizard(cfl=1.0, Δt=45.0, max_change=1.1, max_Δt=1minute)

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of ``u, v, w`` and the current wall clock time.

using Printf

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model
    u, v, w = model.velocities

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

# Now we create the simulation,

simulation = Simulation(model, iteration_interval = 100,
                                               Δt = wizard,
                                        stop_time = 12hours,
                                         progress = print_progress)

# ## Output
#
# ### A field writer
#
# We set up an output writer for the simulation that saves all velocity fields,
# tracer fields, and the subgrid turbulent diffusivity.

output_interval = 30minutes

fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, fields_to_output,
                     schedule = TimeInterval(output_interval),
                     prefix = "shear_turbulence_fields",
                     force = true)

# ### An "averages" writer
#
# We also set up output of time- and horizontally-averaged velocity field and
# momentum fluxes,

u, v, w = model.velocities
T, S = model.tracers

U = AveragedField(u, dims=(1, 2))
V = AveragedField(v, dims=(1, 2))
#B = AveragedField(model.tracers.b, dims=(1, 2))

wu = AveragedField(w * u, dims=(1, 2))
wv = AveragedField(w * v, dims=(1, 2))
ww = AveragedField(w * w, dims=(1, 2))

wT = AveragedField(w * T, dims=(1, 2))
wS = AveragedField(w * S, dims=(1, 2))

simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (u=U, v=V, wu=wu, wv=wv, ww=ww, wT=wT, wS=wS),
                     schedule = AveragedTimeInterval(output_interval, window=2minutes),
                     prefix = "shear_turbulence_averages",
                     force = true)

# ## Running the simulation
#
# This part is easy,

run!(simulation)

# # Making a neat movie
#
# We look at the results by plotting vertical slices of ``u`` and ``w``, and a horizontal
# slice of ``w`` to look for Langmuir cells.

k = searchsortedfirst(grid.zF[:], -8)
nothing # hide

# Making the coordinate arrays takes a few lines of code,

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
xT, yT, zT = nodes(model.tracers.T)
xS, yS, zS = nodes(model.tracers.S)
nothing # hide

using JLD2, Plots
using Statistics

fields_file = jldopen(simulation.output_writers[:fields].filepath)
averages_file = jldopen(simulation.output_writers[:averages].filepath)
iterations = parse.(Int, keys(fields_file["timeseries/t"]))

# Create a plot showing the initial and end profiles of T and S
T_init = fields_file["timeseries/T/0"]
S_init = fields_file["timeseries/S/0"]
end_time = last(iterations)
Tₑ = fields_file["timeseries/T/$end_time"]
Sₑ = fields_file["timeseries/S/$end_time"]
wₑ = fields_file["timeseries/w/$end_time"]
wT_snapshot = averages_file["timeseries/wT/$end_time"][1, 1, :]
wS_snapshot = averages_file["timeseries/wS/$end_time"][1, 1, :]
ww_snapshot = averages_file["timeseries/ww/$end_time"][1, 1, :]
uw_snapshot = averages_file["timeseries/wu/$end_time"][1, 1, :]
vw_snapshot = averages_file["timeseries/wv/$end_time"][1, 1, :]
mean_Tₑ = mean(mean(Tₑ, dims=1), dims=2)
mean_T_init = mean(mean(T_init, dims=1), dims=2)
mean_Sₑ = mean(mean(Sₑ, dims=1), dims=2)
mean_S_init = mean(mean(S_init, dims=1), dims=2)
mean_wₑ² = mean(mean(wₑ.^2, dims=1), dims=2)
mean_b_init = g * ( α * mean_T_init[1,1,:] - β * mean_S_init[1,1,:] )
mean_bₑ = g * ( α * mean_Tₑ[1,1,:] - β * mean_Sₑ[1,1,:] )

plot_T_evolution = plot([mean_T_init[1,1,:], mean_Tₑ[1,1,:]], zT;
                        xlims = (-1.6, -1.1),
                        #xlims = (27.4, 27.6),
                        ylims = (-grid.Lz, 0),
                         xlabel = "Pot. Temp. (C)",
                         ylabel = "z (m)",
                         label = ["t = 0" "t = 48 hours"],
                         legend = :bottomright)

plot_S_evolution = plot([mean_S_init[1,1,:], mean_Sₑ[1,1,:]], zT;
                        xlims = (28.95, 29.2),
                        ylims = (-grid.Lz, 0),
                        xlabel = "S (g/kg)",
                        ylabel = "z (m)",
                        label = ["t = 0" "t = 48 hours"],
                        legend = :bottomleft)

plot_b_evolution = plot( [ mean_b_init , mean_bₑ], zT;
                        xlims = (-0.226, -0.224),
                        ylims = (-grid.Lz, 0),
                        xlabel = "buoyancy perturbation (m² s⁻³)",
                        ylabel = "z (m)",
                        label = ["t = 0" "t = 48 hours"],
                        legend = :bottomright)

plot_w² = plot([mean_wₑ²[1,1,:], ww_snapshot], zw;
                        xlims = (0, 2e-4),
                        ylims = (-grid.Lz, 0),
                        xlabel = "w² (m² s⁻²)",
                        ylabel = "z (m)",
                        label = ["snapshot" "average"],
                        legend = :bottomright)

fluxes_plot = plot([uw_snapshot, vw_snapshot], zw,
                       label = ["uw" "vw"],
                       legend = :bottom,
                       xlabel = "Momentum fluxes (m² s⁻²)",
                       ylabel = "z (m)")

tracer_fluxes_plot = plot([wT_snapshot, wS_snapshot], zw,
                      label = ["wT", "wS"],
                      legend = :bottom,
                      xlabel = "Tracer fluxes (m² s⁻²)",
                      ylabel = "z (m)")

l = @layout [a b c; d e f]
plot_changes = plot(plot_T_evolution, plot_S_evolution, plot_b_evolution, plot_w², fluxes_plot, tracer_fluxes_plot, layout=l, size=(1200, 800),
    title = ["T(z, t=end) (m/s)" "S(z, t=end) (m/s)" "b(z, t=end) (m/s)" "w²(z, t=end) (m/s)"])

png(plot_changes, string("examples/salinity_instability_Us0_", round(Uˢ, sigdigits=1), ".png"))

# Finally, we're ready to animate.

close(fields_file)
close(averages_file)
