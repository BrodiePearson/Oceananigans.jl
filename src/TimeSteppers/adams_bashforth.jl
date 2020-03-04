"""
    AdamsBashforthTimeStepper(float_type, arch, grid, tracers, χ=0.125;
                              Gⁿ = TendencyFields(arch, grid, tracers),
                              G⁻ = TendencyFields(arch, grid, tracers))

Return an AdamsBashforthTimeStepper object with tendency fields on `arch` and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
struct AdamsBashforthTimeStepper{T, TG, P} <: AbstractTimeStepper
     χ :: T
    Gⁿ :: TG
    G⁻ :: TG
    predictor_velocities :: P
end

function AdamsBashforthTimeStepper(float_type, arch, grid, velocities, tracers, χ=0.125;
                                   Gⁿ = TendencyFields(arch, grid, tracers),
                                   G⁻ = TendencyFields(arch, grid, tracers)
                                   )

    u★ = Field{Face, Cell, Cell}(data(velocities.u), grid, velocities.u.boundary_conditions)
    v★ = Field{Cell, Face, Cell}(data(velocities.v), grid, velocities.v.boundary_conditions)
    w★ = Field{Face, Cell, Face}(data(velocities.w), grid, velocities.w.boundary_conditions)

    predictor_velocities = (u=u★, v=v★, w=w★)

    return AdamsBashforthTimeStepper{float_type, typeof(Gⁿ), 
                                     typeof(predictor_velocities)}(χ, Gⁿ, G⁻, predictor_velocities)
                                    
end


#####
##### Time steppping
#####

"""
    time_step!(model::IncompressibleModel{<:AdamsBashforthTimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::IncompressibleModel{<:AdamsBashforthTimeStepper}, Δt; euler=false)
    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Convert NamedTuples of Fields to NamedTuples of OffsetArrays
    velocities, tracers, pressures, diffusivities, Gⁿ, G⁻, predictor_velocities =
        datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities,
                   model.timestepper.Gⁿ, model.timestepper.G⁻, model.timestepper.predictor_velocities)

    ab2_store_previous_source_terms!(G⁻, model.architecture, model.grid, Gⁿ)

    # Calculate tendencies:
    calculate_explicit_substep!(Gⁿ, velocities, tracers, pressures, diffusivities, model)

    # Step forward tracers
    ab2_update_tracers!(tracers, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    # Fractional step. Note that predictor velocities share memory with velocities, which 
    # permits in-place updates of both.
    ab2_update_predictor_velocities!(predictor_velocities, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    calculate_pressure_correction!(pressures.pNHS, Δt, predictor_velocities, model)

    fractional_step_velocities!(velocities, model.architecture, model.grid, Δt, pressures.pNHS)

    compute_w_from_continuity!(model)

    tick!(model.clock, Δt)

    return nothing
end

#####
##### Adams-Bashforth-specific kernels
#####

""" Store previous source terms before updating them. """
function ab2_store_previous_source_terms!(G⁻, arch, grid, Gⁿ)

    # Velocity fields
    @launch(device(arch), config=launch_config(grid, :xyz),
            ab2_store_previous_velocity_source_terms!(G⁻, grid, Gⁿ))

    # Tracer fields
    for i in 4:length(G⁻)
        @inbounds Gc⁻ = G⁻[i]
        @inbounds Gcⁿ = Gⁿ[i]
        @launch(device(arch), config=launch_config(grid, :xyz),
                ab2_store_previous_tracer_source_term!(Gc⁻, grid, Gcⁿ))
    end

    return nothing
end

""" Store previous source terms for `u`, `v`, and `w` before updating them. """
function ab2_store_previous_velocity_source_terms!(G⁻, grid, Gⁿ)
    @loop_xyz i j k grid begin
        @inbounds G⁻.u[i, j, k] = Gⁿ.u[i, j, k]
        @inbounds G⁻.v[i, j, k] = Gⁿ.v[i, j, k]
        @inbounds G⁻.w[i, j, k] = Gⁿ.w[i, j, k]
    end
    return nothing
end

""" Store previous source terms for a tracer before updating them. """
function ab2_store_previous_tracer_source_term!(Gc⁻, grid, Gcⁿ)
    @loop_xyz i j k grid begin
        @inbounds Gc⁻[i, j, k] = Gcⁿ[i, j, k]
    end
    return nothing
end

# Predictor velocity stuff
  
"""
Evaluate the right-hand-side terms for velocity fields and tracer fields
at time step n+½ using a weighted 2nd-order Adams-Bashforth method.
"""
function ab2_update_predictor_velocities!(U★, arch, grid, Δt, χ, Gⁿ, G⁻)
    @launch(device(arch), config=launch_config(grid, :xyz),
            _ab2_update_predictor_velocities!(U★, grid, Δt, χ, Gⁿ, G⁻))
    return nothing
end

""" Update predictor velocity field. """
function _ab2_update_predictor_velocities!(U★, grid::AbstractGrid{FT}, Δt, χ, Gⁿ, G⁻) where FT
    @loop_xyz i j k grid begin
        @inbounds begin
            U★.u[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.u[i, j, k] 
                                    - (FT(0.5) + χ) * G⁻.u[i, j, k] )

            U★.v[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.v[i, j, k] 
                                    - (FT(0.5) + χ) * G⁻.v[i, j, k] )

            U★.w[i, j, k] += Δt * (   (FT(1.5) + χ) * Gⁿ.w[i, j, k] 
                                    - (FT(0.5) + χ) * G⁻.w[i, j, k] )

    end
    return nothing
end

# Fractional step

"""
Update tracer via

    `c^{n+1} = c^n + Δt [ (1.5 + χ) Gc^{n} - (0.5 + χ) Gc^{-}`

"""
function ab2_update_tracer!(c, grid::AbstractGrid{FT}, Δt, χ, Gcⁿ, Gc⁻) where FT
    @loop_xyz i j k grid begin
        @inbounds c[i, j, k] += Δt * ( (FT(1.5) + χ) * Gcⁿ[i, j, k] - (FT(0.5) + χ) * Gc⁻[i, j, k] )
    end
    return nothing
end

"Update the solution variables (velocities and tracers)."
function ab2_update_tracers!(C, arch, grid, Δt, χ, Gⁿ, G⁻)
    for i in 1:length(C)
        @inbounds c = C[i]
        @inbounds Gcⁿ = Gⁿ[i+3]
        @inbounds Gc⁻ = G⁻[i+3]
        @launch device(arch) config=launch_config(grid, :xyz) ab2_update_tracer!(c, grid, Δt, χ, Gcⁿ, Gc⁻)
    end

    return nothing
end
