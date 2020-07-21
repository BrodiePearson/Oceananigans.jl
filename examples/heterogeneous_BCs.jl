# # Langmuir turbulence example
#
# This example implements the Langmuir turbulence simulation reported in section
# 4 of
#
# [McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of Fluid Mechanics (1997)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/langmuir-turbulence-in-the-ocean/638FD0E368140E5972144348DB930A38).
#
# This example demonstrates:
#
#   * how to run large eddy simulations with surface wave effects
#     via the Craik-Leibovich approximation

using Oceananigans

# ## Model set-up
#
# To build the model, we specify the grid, Stokes drift, boundary conditions, and
# Coriolis parameter.
#
# ### Domain specification and Grid construction
#
# We create a grid with modest resolution. The grid extent is similar, but not
# exactly the same as that in McWilliams et al. (1997).

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(32, 32, 48), extent=(128, 128, 96))

# ### The Stokes Drift profile
#
# The surface wave Stokes drift profile used in McWilliams et al. (1997)
# corresponds to a 'monochromatic' (that is, single-frequency) wave field with

const wavenumber = 2π / 60 # m⁻¹
nothing # hide

# and

const amplitude = 0.8 # m
nothing # hide

# The `const` declarations ensure that Stokes drift functions compile on the GPU.
# To run this example on the GPU, write `architecture = GPU()` in the constructor
# for `IncompressibleModel` below.
#
# The Stokes drift at the surface for a monochromatic, deep water wave is

using Oceananigans.Buoyancy: g_Earth

const Uˢ = amplitude^2 * wavenumber * sqrt(g_Earth * wavenumber) # m s⁻¹

# The Stokes drift profile is then,

uˢ(z) = Uˢ * exp(2wavenumber * z)
nothing # hide

# which we need for the initial condition.
#
# Note that `Oceananigans.jl` implements the Lagrangian-mean form of the Craik-Leibovich
# equations. This means that our model takes the *vertical derivative* as an input,
# rather than the Stokes drift profile itself.
#
# The vertical derivative of the Stokes drift is

∂z_uˢ(z, t) = 2wavenumber * Uˢ * exp(2wavenumber * z)
nothing # hide

# Finally, we note that the time-derivative of the Stokes drift must be provided
# if the Stokes drift changes in time. In this example, the Stokes drift is constant
# and thus the time-derivative of the Stokes drift is 0.

# ### Boundary conditions
#
# At the surface at $z=0$, McWilliams et al. (1997) impose wind stress,

Qᵘ = -3.72e-5 # m² s⁻²
nothing # hide

N² = 1.936e-5 # s⁻²
nothing # hide

# Physical constants.
ρ₀ = 1027  # Density of seawater [kg/m³]
uᶠ = sqrt(-Qᵘ)
Laᵗ = sqrt(uᶠ/uˢ(0))

# add a strong, heterogeneous cooling at the surface
# First create Heaviside and Interbal functions
@inline function heaviside(t)
   0.5 * (sign(t) + 1)
end
@inline function interval(t, a, b)
    # Interval is +1 if t is between a and b, and a<b
   heaviside(t-a) - heaviside(t-b)
end

#@inline Qᵇ_map(y) = 2.307e-7*interval(y, 70, 40) # m³ s⁻²

@inline Qᵇ(x, y, t) = 2.307e-6*interval(y, 40, 80)
nothing # hide


#const cooling_flux = 5e-7 # m² s⁻³

# which corresponds to an upward heat flux of ≈ 1000 W m⁻².
# We cool just long enough to deepen the boundary layer to 100 m.

#target_depth = 100 # m

## Duration of convection to produce a boundary layer with `target_depth`.
## The "3" is empirical.
#const t_convection = target_depth^2 * N² / (3 * cooling_flux)

#@inline Qᵇ(x, y, t) = ifelse(t < t_convection, cooling_flux, 0.0)
#nothing

# Oceananigans uses "positive upward" conventions for all fluxes. In consequence,
# a negative flux at the surface drives positive velocities, and a positive flux of
# buoyancy drives cooling.
#
# The initial condition and bottom boundary condition for buoyancy
# impose a linear stratification with buoyancy frequency


# To summarize, we impose a surface flux on $u$,

using Oceananigans.BoundaryConditions

u_boundary_conditions = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))
nothing # hide

# and a surface flux and bottom linear gradient on buoyancy, $b$,

b_boundary_conditions = TracerBoundaryConditions(grid, top = TracerBoundaryCondition(Flux, :z, Qᵇ),
                                                       bottom = BoundaryCondition(Gradient, N²))
nothing # hide

# ### Coriolis parameter
#
# McWilliams et al. (1997) use

f = 1e-4 # s⁻¹
nothing # hide

# which is typical for mid-latitudes on Earth.

# ## Model instantiation
#
# Finally, we are ready to build the model. We use the AnisotropicMinimumDissipation
# model for large eddy simulation. Because our Stokes drift does not vary in $x, y$,
# we use `UniformStokesDrift`, which expects Stokes drift functions of $z, t$ only.

using Oceananigans.Buoyancy: BuoyancyTracer
using Oceananigans.SurfaceWaves: UniformStokesDrift

model = IncompressibleModel(        architecture = CPU(),
                                            grid = grid,
                                         tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=f),
                                         closure = AnisotropicMinimumDissipation(),
                                   surface_waves = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                             boundary_conditions = (u=u_boundary_conditions,
                                                    b=b_boundary_conditions),
                            )

# ## Initial conditions
#
# We make use of random noise concentrated in the upper 4 meters
# for buoyancy and velocity initial conditions,

Ξ(z) = randn() * exp(z / 4)
nothing # hide

# Set constant temperature profile
η(z) =  N²*z
nothing # hide
# Add a Gaussian maximum between 15m and 25m depth (peaking at 20m)
# Add a second Gaussian maximum between 40 and 50 m depth (triple amplitude of other peak)

# Add constant stratification below 50m


# Impose a subsurface buoyancy maximum (this will convect as linear EOS),

bᵢ(x, y, z) = η(z)
nothing # hide

# The velocity initial condition is zero *Eulerian* velocity. This means that we
# must add the Stokes drift profile to the $u$ velocity field. We also add noise scaled
# by the friction velocity to $u$ and $w$.

uᵢ(x, y, z) = uˢ(z) + sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

wᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

# ## Setting up the simulation
#
# We use the `TimeStepWizard` for adaptive time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2,

wizard = TimeStepWizard(cfl=0.2, Δt=5.0, max_change=1.1, max_Δt=10.0)
nothing # hide

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of $u, v, w$ and the current wall clock time.

using Oceananigans.Diagnostics, Printf

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   umax(), vmax(), wmax(),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

# Now we create the simulation,

using Oceananigans.Utils: hour # correpsonds to "1 hour", in units of seconds

simulation = Simulation(model, progress_frequency = 100,
                                               Δt = wizard,
                                        stop_time = 4hour,
                                         progress = print_progress)

# ## Output
#
# We set up an output writer for the simulation that saves all velocity fields,
# tracer fields, and the subgrid turbulent diffusivity every 2 minutes.

using Oceananigans.OutputWriters
using Oceananigans.Utils: minute

field_outputs = FieldOutputs(merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,)))

simulation.output_writers[:fields] = JLD2OutputWriter(model, field_outputs,
                                                      interval = 2minute,
                                                        prefix = "langmuir_turbulence",
                                                         force = true)
nothing # hide

# ## Running the simulation
#
# This part is easy,

run!(simulation)

# # Making a neat movie
#
# We look at the results by plotting vertical slices of $u$ and $w$, and a horizontal
# slice of $w$ to look for Langmuir cells.

k = searchsortedfirst(grid.zF[:], -8)
nothing # hide

# Making the coordinate arrays takes a few lines of code,

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
xb, yb, zb = nodes(model.tracers.b)

xw, yw, zw = xw[:], yw[:], zw[:]
xu, yu, zu = xu[:], yu[:], zu[:]
xb, yb, zb = xb[:], yb[:], zb[:]
nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim)
    levels = range(-clim, stop=clim, length=10)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=10), [cmax])
    end

    return levels
end
nothing # hide

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from file, omitting halo regions
    w = file["timeseries/w/$iter"][2:end-1, 2:end-1, 2:end-1]
    u = file["timeseries/u/$iter"][2:end-1, 2:end-1, 2:end-1]
    b = file["timeseries/b/$iter"][2:end-1, 2:end-1, 2:end-1]

    ## Extract slices
    wxy = w[:, :, k]
    wxz = w[:, 1, :]
    uxz = u[:, 1, :]
    bz = b[1, 1, :]

    wlim = 0.02
    ulim = 0.05
    blim = 0.002
    wlevels = nice_divergent_levels(w, wlim)
    ulevels = nice_divergent_levels(w, ulim)

    wxy_plot = contourf(xw, yw, wxy';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (0, grid.Ly),
                             xlabel = "x (m)",
                             ylabel = "y (m)")

    wxz_plot = contourf(xw, zw, wxz';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    uxz_plot = contourf(xu, zu, uxz';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-ulim, ulim),
                             levels = ulevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    bz_plot = plot(bz, zb;
                            xlims = (-blim, 0.002),
                            ylims = (-grid.Lz, 0),
                             xlabel = "b (?)",
                             ylabel = "z (m)",
                             legend = false)

    plot(wxy_plot, wxz_plot, uxz_plot, bz_plot, layout=(1, 4), size=(1300, 400),
         title = ["w(x, y, z=-8, t) (m/s)" "w(x, y=0, z, t) (m/s)" "u(x, y = 0, z, t) (m/s)" "b(x=0, y = 0, z, t) (m/s)"])

    iter == iterations[end] && close(file)
end

mp4(anim, "seaice_lead_y.mp4", fps = 5) # hide
