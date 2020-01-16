using Oceananigans.Solvers: solve_poisson_equation!

function ∇²!(grid, f, ∇²f)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
            end
        end
    end
end

function pressure_solver_instantiates(FT, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(100, 100, 100))
    solver = PressureSolver(CPU(), grid, HorizontallyPeriodicBCs(), planner_flag)
    return true  # Just making sure the PressureSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    fbcs = HorizontallyPeriodicBCs()
    solver = PressureSolver(arch, grid, fbcs)

    RHS = CellField(FT, arch, grid)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)
    solver.storage .= interior(RHS)
    solve_poisson_equation!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ, ∇²ϕ)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)

    interior(∇²ϕ) ≈ interior(RHS_orig)
end

function poisson_pnn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    fbcs = ChannelBCs()
    solver = PressureSolver(arch, grid, fbcs)

    RHS = CellField(FT, arch, grid)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)

    solver.storage .= interior(RHS)

    solve_poisson_equation!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ, ∇²ϕ)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)

    interior(∇²ϕ) ≈ interior(RHS_orig)
end

function poisson_ppn_planned_div_free_gpu(FT, Nx, Ny, Nz)
    arch = GPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    fbcs = HorizontallyPeriodicBCs()
    solver = PressureSolver(arch, grid, fbcs)

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)
    RHS = CuArray(RHS)

    RHS_orig = copy(RHS)

    solver.storage .= RHS

    # Performing the permutation [a, b, c, d, e, f] -> [a, c, e, f, d, b]
    # in the z-direction in preparation to calculate the DCT in the Poisson
    # solver.
    solver.storage .= cat(solver.storage[:, :, 1:2:Nz], solver.storage[:, :, Nz:-2:2]; dims=3)

    solve_poisson_equation!(solver, grid)

    # Undoing the permutation made above to complete the IDCT.
    solver.storage .= CuArray(reshape(permutedims(cat(solver.storage[:, :, 1:Int(Nz/2)],
                                                      solver.storage[:, :, end:-1:Int(Nz/2)+1]; dims=4), (1, 2, 4, 3)), Nx, Ny, Nz))

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ.data, ∇²ϕ.data)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)
    interior(∇²ϕ) ≈ RHS_orig
end

function poisson_pnn_planned_div_free_gpu(FT, Nx, Ny, Nz)
    arch = GPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    fbcs = ChannelBCs()
    solver = PressureSolver(arch, grid, fbcs)

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)
    RHS = CuArray(RHS)

    RHS_orig = copy(RHS)

    storage = solver.storage.storage1
    storage .= RHS

    storage .= cat(storage[:, :, 1:2:Nz], storage[:, :, Nz:-2:2]; dims=3)
    storage .= cat(storage[:, 1:2:Ny, :], storage[:, Ny:-2:2, :]; dims=2)

    solve_poisson_equation!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    # Indices used when we need views to permuted arrays where the odd indices
    # are iterated over first followed by the even indices.
    p_y_inds = [1:2:Ny..., Ny:-2:2...] |> CuArray
    p_z_inds = [1:2:Nz..., Nz:-2:2...] |> CuArray

    ϕ_p = view(interior(ϕ), 1:Nx, p_y_inds, p_z_inds)

    @. ϕ_p = real(storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ.data, ∇²ϕ.data)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)
    interior(∇²ϕ) ≈ RHS_orig
end

"""
    poisson_ppn_recover_sine_cosine_solution(FT, Nx, Ny, Nz, Lx, Ly, Lz, mx, my, mz)

Test that the Poisson solver can recover an analytic solution. In this test, we
are trying to see if the solver can recover the solution

    ``\\Psi(x, y, z) = cos(\\pi m_z z / L_z) sin(2\\pi m_y y / L_y) sin(2\\pi m_x x / L_x)``

by giving it the source term or right hand side (RHS), which is

    ``f(x, y, z) = \\nabla^2 \\Psi(x, y, z) =
    -((\\pi m_z / L_z)^2 + (2\\pi m_y / L_y)^2 + (2\\pi m_x/L_x)^2) \\Psi(x, y, z)``.
"""
function poisson_ppn_recover_sine_cosine_solution(FT, Nx, Ny, Nz, Lx, Ly, Lz, mx, my, mz)
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    solver = PressureSolver(CPU(), grid, HorizontallyPeriodicBCs())

    xC, yC, zC = grid.xC, grid.yC, grid.zC
    xC = reshape(xC, (Nx, 1, 1))
    yC = reshape(yC, (1, Ny, 1))
    zC = reshape(zC, (1, 1, Nz))

    Ψ(x, y, z) = cos(π*mz*z/Lz) * sin(2π*my*y/Ly) * sin(2π*mx*x/Lx)
    f(x, y, z) = -((mz*π/Lz)^2 + (2π*my/Ly)^2 + (2π*mx/Lx)^2) * Ψ(x, y, z)

    @. solver.storage = f(xC, yC, zC)
    solve_poisson_equation!(solver, grid)
    ϕ = real.(solver.storage)

    error = norm(ϕ - Ψ.(xC, yC, zC)) / √(Nx*Ny*Nz)

    @info "Error (ℓ²-norm), $FT, N=($Nx, $Ny, $Nz), m=($mx, $my, $mz): $error"

    isapprox(ϕ, Ψ.(xC, yC, zC); rtol=5e-2)
end

function vertically_stretched_poisson_solver_correct_answer(arch, Nx, Ny, zF)
    Lx, Ly, Lz = 1, 1, zF[end]
    Δx, Δy = Lx/Nx, Ly/Ny

    #####
    ##### Vertically stretched operators
    #####

    @inline δx_caa(i, j, k, f) = @inbounds f[i+1, j, k] - f[i, j, k]
    @inline δy_aca(i, j, k, f) = @inbounds f[i, j+1, k] - f[i, j, k]
    @inline δz_aac(i, j, k, f) = @inbounds f[i, j, k+1] - f[i, j, k]

    @inline ∂x_caa(i, j, k, Δx,  f) = δx_caa(i, j, k, f) / Δx
    @inline ∂y_aca(i, j, k, Δy,  f) = δy_aca(i, j, k, f) / Δy
    @inline ∂z_aac(i, j, k, ΔzF, f) = δz_aac(i, j, k, f) / ΔzF[k]

    @inline ∂x²(i, j, k, Δx, f)       = (∂x_caa(i, j, k, Δx, f)  - ∂x_caa(i-1, j, k, Δx, f))  / Δx
    @inline ∂y²(i, j, k, Δy, f)       = (∂y_aca(i, j, k, Δy, f)  - ∂y_aca(i, j-1, k, Δy, f))  / Δy
    @inline ∂z²(i, j, k, ΔzF, ΔzC, f) = (∂z_aac(i, j, k, ΔzF, f) - ∂z_aac(i, j, k-1, ΔzF, f)) / ΔzC[k]

    @inline div_ccc(i, j, k, Δx, Δy, ΔzF, u, v, w) = ∂x_caa(i, j, k, Δx, u) + ∂y_aca(i, j, k, Δy, v) + ∂z_aac(i, j, k, ΔzF, w)

    @inline ∇²(i, j, k, Δx, Δy, ΔzF, ΔzC, f) = ∂x²(i, j, k, Δx, f) + ∂y²(i, j, k, Δy, f) + ∂z²(i, j, k, ΔzF, ΔzC, f)

    #####
    ##### Generate "fake" vertically stretched grid
    #####

    function grid(zF)
        Nz = length(zF) - 1
        ΔzF = [zF[k+1] - zF[k] for k in 1:Nz]
        zC = [(zF[k] + zF[k+1]) / 2 for k in 1:Nz]
        ΔzC = [zC[k+1] - zC[k] for k in 1:Nz-1]
        return zF, zC, ΔzF, ΔzC
    end

    Nz = length(zF) - 1
    zF, zC, ΔzF, ΔzC = grid(zF)

    # Need some halo regions.
    ΔzF = OffsetArray([ΔzF[1], ΔzF...], 0:Nz)
    ΔzC = [ΔzC..., ΔzC[end]]

    # Useful for broadcasting z operations
    ΔzC = reshape(ΔzC, (1, 1, Nz))

    # Temporary hack: Useful for reusing fill_halo_regions! and BatchedTridiagonalSolver
    # which only need Nx, Ny, Nz.
    fake_grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))

    #####
    ##### Generate batched tridiagonal system coefficients and solver
    #####

    function λi(Nx, Δx)
        is = reshape(1:Nx, Nx, 1, 1)
        @. (2sin((is-1)*π/Nx) / Δx)^2
    end

    function λj(Ny, Δy)
        js = reshape(1:Ny, 1, Ny, 1)
        @. (2sin((js-1)*π/Ny) / Δy)^2
    end

    kx² = λi(Nx, Δx)
    ky² = λj(Ny, Δy)

    # Lower and upper diagonals are the same
    ld = [1/ΔzF[k] for k in 1:Nz-1]
    ud = copy(ld)

    # Diagonal (different for each i,j)
    @inline δ(k, ΔzF, ΔzC, kx², ky²) = - (1/ΔzF[k-1] + 1/ΔzF[k]) - ΔzC[k] * (kx² + ky²)

    d = zeros(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny
        d[i, j, 1] = -1/ΔzF[1] - ΔzC[1] * (kx²[i] + ky²[j])
        d[i, j, 2:Nz-1] .= [δ(k, ΔzF, ΔzC, kx²[i], ky²[j]) for k in 2:Nz-1]
        d[i, j, Nz] = -1/ΔzF[Nz-1] - ΔzC[Nz] * (kx²[i] + ky²[j])
    end

    #####
    ##### Random right hand side
    #####

    # Random right hand side
    Ru = CellField(Float64, arch, fake_grid)
    Rv = CellField(Float64, arch, fake_grid)
    Rw = CellField(Float64, arch, fake_grid)

    interior(Ru) .= rand(Nx, Ny, Nz)
    interior(Rv) .= rand(Nx, Ny, Nz)
    interior(Rw) .= zeros(Nx, Ny, Nz)
    U = (u=Ru, v=Rv, w=Rw)

    uv_bcs = HorizontallyPeriodicBCs()
    w_bcs = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())

    fill_halo_regions!(Ru.data, uv_bcs, arch, fake_grid)
    fill_halo_regions!(Rv.data, uv_bcs, arch, fake_grid)

    _compute_w_from_continuity!(U, fake_grid)

    fill_halo_regions!(Rw.data,  w_bcs, arch, fake_grid)

    R = zeros(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        R[i, j, k] = div_ccc(i, j, k, Δx, Δy, ΔzF, Ru.data, Rv.data, Rw.data)
    end

    # @show sum(R)  # should be zero by construction.

    F = zeros(Nx, Ny, Nz)
    F = ΔzC .* R  # RHS needs to be multiplied by ΔzC

    #####
    ##### Solve system
    #####

    F̃ = fft(F, [1, 2])

    btsolver = BatchedTridiagonalSolver(arch, dl=ld, d=d, du=ud, f=F̃, grid=fake_grid)

    ϕ̃ = zeros(Complex{Float64}, Nx, Ny, Nz)
    solve_batched_tridiagonal_system!(ϕ̃, arch, btsolver)

    ϕ = CellField(Float64, arch, fake_grid)
    interior(ϕ) .= real.(ifft(ϕ̃, [1, 2]))
    ϕ.data .= ϕ.data .- mean(interior(ϕ))

    #####
    ##### Compute Laplacian of solution ϕ to test that it's correct
    #####

    # Gotta fill halo regions
    fbcs = HorizontallyPeriodicBCs()
    fill_halo_regions!(ϕ.data, fbcs, arch, fake_grid)

    ∇²ϕ = CellField(Float64, arch, fake_grid)

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ∇²ϕ.data[i, j, k] = ∇²(i, j, k, Δx, Δy, ΔzF, ΔzC, ϕ.data)
    end

    return interior(∇²ϕ) ≈ R
end

@testset "Pressure solvers" begin
    @info "Testing pressure solvers..."

    @testset "Pressure solver instantiation" begin
        @info "  Testing pressure solver instantiation..."

        for FT in float_types
            @test pressure_solver_instantiates(FT, 32, 32, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT, 1,  32, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT, 32,  1, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT,  1,  1, 32, FFTW.ESTIMATE)
        end
    end

    @testset "Divergence-free solution [CPU]" begin
        @info "  Testing divergence-free solution [CPU]..."

        for N in [7, 10, 16, 20]
            for FT in float_types
                for planner_flag in (FFTW.ESTIMATE, FFTW.MEASURE)
                    @test poisson_ppn_planned_div_free_cpu(FT, N, N, N, planner_flag)
                    @test poisson_ppn_planned_div_free_cpu(FT, 1, N, N, planner_flag)
                    @test poisson_ppn_planned_div_free_cpu(FT, N, 1, N, planner_flag)
                    @test poisson_ppn_planned_div_free_cpu(FT, 1, 1, N, planner_flag)
                end
            end
        end

        Ns = [5, 11, 20, 32]
        for Nx in Ns, Ny in Ns, Nz in Ns, FT in float_types
            @test poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, FFTW.ESTIMATE)
            @test poisson_pnn_planned_div_free_cpu(FT, Nx, Ny, Nz, FFTW.ESTIMATE)
        end
    end

    @testset "Divergence-free solution [GPU]" begin
        @info "  Testing divergence-free solution [GPU]..."
        @hascuda begin
            for FT in [Float64]
                @test poisson_ppn_planned_div_free_gpu(FT, 16, 16, 16)
                @test poisson_ppn_planned_div_free_gpu(FT, 32, 32, 32)
                @test poisson_ppn_planned_div_free_gpu(FT, 32, 32, 16)
                @test poisson_ppn_planned_div_free_gpu(FT, 16, 32, 24)

                @test poisson_pnn_planned_div_free_gpu(FT, 16, 16, 16)
                @test poisson_pnn_planned_div_free_gpu(FT, 32, 32, 32)
                @test poisson_pnn_planned_div_free_gpu(FT, 32, 32, 16)
                @test poisson_pnn_planned_div_free_gpu(FT, 16, 32, 24)
            end
        end
    end

    @testset "Analytic solution reconstruction" begin
        @info "  Testing analytic solution reconstruction..."
        for N in [32, 48, 64], m in [1, 2, 3]
            @test poisson_ppn_recover_sine_cosine_solution(Float64, N, N, N, 100, 100, 100, m, m, m)
        end
    end

    for arch in [CPU()]
        @testset "Vertically stretched Poisson solver [FACR, $arch]" begin
            @info "  Testing vertically stretched Poisson solver [FACR, $arch]..."

            Nx = Ny = 8
            zF = [1, 2, 4, 7, 11, 16, 22, 29, 37]
            @test vertically_stretched_poisson_solver_correct_answer(arch, Nx, Ny, zF)
        end
    end
end
