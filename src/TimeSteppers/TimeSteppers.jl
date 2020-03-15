module TimeSteppers

export
    AdamsBashforthTimeStepper,
    time_step!,
    compute_w_from_continuity!

using GPUifyLoops: @launch, @loop, @unroll

import Oceananigans: TimeStepper

using Oceananigans.Architectures: @hascuda
@hascuda using CUDAnative, CuArrays

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.SurfaceWaves
using Oceananigans.BoundaryConditions
using Oceananigans.Solvers
using Oceananigans.Models
using Oceananigans.Utils

using Oceananigans.TurbulenceClosures:
    calculate_diffusivities!, ∂ⱼ_2ν_Σ₁ⱼ, ∂ⱼ_2ν_Σ₂ⱼ, ∂ⱼ_2ν_Σ₃ⱼ, ∇_κ_∇c

"""
    AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

"""
    TimeStepper(name, args...)

Returns a timestepper with name `name`, instantiated with `args...`.

Example
=======

julia> stepper = TimeStepper(:AdamsBashforth, Float64, CPU(), grid, tracernames)
"""
function TimeStepper(name::Symbol, args...)
    fullname = Symbol(name, :TimeStepper)
    return eval(Expr(:call, fullname, args...))
end

# Fallback
TimeStepper(stepper::AbstractTimeStepper, args...) = stepper

"""Returns the arguments passed to boundary conditions functions."""
boundary_condition_function_arguments(model) =
    (model.clock.time, model.clock.iteration, datatuple(model.velocities),
     datatuple(model.tracers), model.parameters)

include("time_step_precomputations.jl")
include("calculate_tendencies.jl")
include("calculate_pressure_correction.jl")
include("velocity_and_tracer_tendencies.jl")
include("time_stepping_kernels.jl")
include("adams_bashforth.jl")

end # module
