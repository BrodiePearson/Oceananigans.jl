"""
    AnisotropicBiharmonicDiffusivity{FT, KH, KV}

Parameters for anisotropic biharmonic diffusivity models.
"""
struct AnisotropicBiharmonicDiffusivity{FT, KH, KV} <: AbstractTensorDiffusivity
    νh :: FT
    νv :: FT
    κh :: KH
    κv :: KV
    function AnisotropicBiharmonicDiffusivity{FT}(νh, νv, κh, κv) where FT
        return new{FT, typeof(κh), typeof(κv)}(νh, νv, convert_diffusivity(FT, κh), convert_diffusivity(FT, κv))
    end
end

"""
    AnisotropicBiharmonicDiffusivity(; νh, νv, κh, κv)

Returns parameters for a fourth-order, anisotropic biharmonic diffusivity closure with
constant horizontal and vertical biharmonic viscosities `νh`, `νv` and constant horizontal
and vertical tracer biharmonic diffusivities `κh`, `κv`.
`κh` and `κv` may be `NamedTuple`s with fields corresponding
to each tracer, or a single number to be a applied to all tracers.
The tracer flux divergence associated with an anisotropic biharmonic diffusivity is, for example

```math
    ∂ᵢ κᵢⱼ ∂ⱼc = (κh ∇⁴h + κv ∂⁴z) c
```

"""
AnisotropicBiharmonicDiffusivity(FT=Float64; νh=0, νv=0, κh=0, κv=0) =
    AnisotropicBiharmonicDiffusivity{FT}(νh, νv, κh, κv)

function with_tracers(tracers, closure::AnisotropicBiharmonicDiffusivity{FT}) where FT
    κh = tracer_diffusivities(tracers, closure.κh)
    κv = tracer_diffusivities(tracers, closure.κv)
    return AnisotropicBiharmonicDiffusivity{FT}(closure.νh, closure.νv, κh, κv)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicBiharmonicDiffusivity, args...) = nothing

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity, U, args...) = (
    - closure.νh * ∇⁴hᶠᶜᵃ(i, j, k, grid, U.u)
    - closure.νv * ∂⁴zᵃᵃᶜ(i, j, k, grid, U.u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity, U, args...) = (
    - closure.νh * ∇⁴hᶜᶠᵃ(i, j, k, grid, U.v)
    - closure.νv * ∂⁴zᵃᵃᶜ(i, j, k, grid, U.v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity, U, args...) = (
    - closure.νh * ∇⁴hᶜᶜᵃ(i, j, k, grid, U.w)
    - closure.νv * ∂⁴zᵃᵃᶠ(i, j, k, grid, U.w)
    )

@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κh = closure.κh[tracer_index]
    @inbounds κv = closure.κv[tracer_index]

    return (- κh * ∇⁴hᶜᶜᵃ(i, j, k, grid, c)
            - κv * ∂⁴zᵃᵃᶜ(i, j, k, grid, c)
           )
end
