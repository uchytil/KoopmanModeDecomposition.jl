using KoopmanModeDecomposition
using LinearAlgebra
using Test

@testset "KoopmanModeDecomposition.jl" begin
    @testset "1. Observables & Delays" begin
        X = reshape(1.0:20.0, 2, 10) # 2 variables, 10 timesteps

        # Identity
        @test size(Identity()(X)) == (2, 10)
        @test size(Identity(1)(X)) == (1, 10)

        # Max Delay Logic
        @test KoopmanModeDecomposition.max_delay(Identity()) == 0
        @test KoopmanModeDecomposition.max_delay(Delay(3) ∘ Identity()) == 3
        @test KoopmanModeDecomposition.max_delay(Delay(1:4) ∘ Identity()) == 4

        # Delay Composition Math
        d_combo = Delay(2) ∘ Delay(3)
        @test d_combo.τ == 5

        d_block = Delay(1:2) ∘ Delay(1:2)
        @test d_block.τ == [2, 3, 3, 4] # Cartesian outer sum!

        # Evaluation Shapes
        dict = [Identity(); Delay(2) ∘ Identity()]
        Y = dict(X)
        @test size(Y) == (4, 8) # 2 (id) + 2 (delay) rows, 10 - 2 max_delay cols
    end

    @testset "2. Monomials" begin
        # Use a deterministic matrix so we can evaluate exact math
        X = [2.0 3.0;   # Variable 1 (x₁)
            4.0 5.0]   # Variable 2 (x₂)

        # Exact degree 2 of 2 variables has 3 terms: x₁², x₁*x₂, x₂²
        m1 = Monomials(2)
        Y1 = m1(X, 1:2)

        @test size(Y1) == (3, 2)

        # The expected exact values column-by-column
        expected_Y1 = [
            4.0 9.0;   # x₁²
            8.0 15.0;   # x₁ * x₂
            16.0 25.0    # x₂²
        ]

        # Check that our framework correctly computes the exact math
        @test isapprox(Y1, expected_Y1, atol=1e-10)

        # Degree range 1:2 should include degree-1 and degree-2 terms
        m2 = Monomials(1:2)
        Y2 = m2(X, 1:2)

        expected_Y2 = [
            2.0 3.0;    # x₁
            4.0 5.0;    # x₂
            4.0 9.0;    # x₁²
            8.0 15.0;   # x₁ * x₂
            16.0 25.0   # x₂²
        ]

        @test size(Y2) == (5, 2)
        @test isapprox(Y2, expected_Y2, atol=1e-10)

        # Explicit degree list should be supported
        m3 = Monomials([1, 2, 8])
        Y3 = m3(X, 1:2)
        @test size(Y3) == (14, 2) # (1+1) + (2+1) + (8+1) terms for 2 variables

        # Degree 0 is the constant monomial
        m4 = Monomials([0, 2])
        Y4 = m4(X, 1:2)

        expected_Y4 = [
            1.0 1.0;
            4.0 9.0;
            8.0 15.0;
            16.0 25.0
        ]

        @test size(Y4) == (4, 2)
        @test isapprox(Y4, expected_Y4, atol=1e-10)

        # Degree 0 alone should produce only the constant monomial
        m5 = Monomials(0)
        Y5 = m5(X, 1:2)
        @test size(Y5) == (1, 2)
        @test isapprox(Y5, ones(1, 2), atol=1e-10)

        # Duplicate degrees should be interpreted literally (no deduplication)
        m6 = Monomials([2, 1, 2])
        Y6 = m6(X, 1:2)
        expected_Y6 = vcat(expected_Y1, expected_Y2[1:2, :], expected_Y1)
        @test size(Y6) == (8, 2)
        @test isapprox(Y6, expected_Y6, atol=1e-10)

        # Invalid degree specifications should throw
        @test_throws ArgumentError Monomials(-1)
        @test_throws ArgumentError Monomials(Int[])
        @test_throws ArgumentError Monomials(-1:1)
        @test_throws MethodError Monomials(2, drop_constant=true)
    end

    @testset "3. Solvers (Linear Recovery)" begin
        # Create a perfectly linear system: x_{t+1} = A * x_t
        A_true = [0.9 0.1; -0.1 0.8]
        X = zeros(2, 100)
        X[:, 1] = [1.0, 0.0]
        for t in 1:99
            X[:, t+1] = A_true * X[:, t]
        end

        dict = Identity()

        # Test PseudoInverse
        model_pinv = fit(PseudoInverse(), dict, X)
        @test isapprox(model_pinv.operator, A_true, atol=1e-5)

        # Test TruncatedSVD
        model_svd = fit(TruncatedSVD(atol=1e-8), dict, X)
        @test isapprox(model_svd.operator, A_true, atol=1e-5)
    end


    @testset "4. Solvers (Multiple Trajectories)" begin
        # Create a perfectly linear system: x_{t+1} = A * x_t
        A_true = [0.9 0.1; -0.1 0.8]


        x₁s = LinRange(-1.0, 1.0, 5)
        x₂s = LinRange(-1.0, 1.0, 5)

        Xs = Matrix{Float64}[]

        for x₁ in x₁s, x₂ in x₂s
            X = zeros(2, 100)
            X[:, 1] = [x₁, x₂]
            for t in 1:99
                X[:, t+1] = A_true * X[:, t]
            end
            push!(Xs, X)
        end


        dict = Identity()

        # Test PseudoInverse
        model_pinv = fit(PseudoInverse(), dict, Xs)
        @test isapprox(model_pinv.operator, A_true, atol=1e-5)

        # Test TruncatedSVD
        model_svd = fit(TruncatedSVD(atol=1e-8), dict, Xs)
        @test isapprox(model_svd.operator, A_true, atol=1e-5)
    end


    @testset "5. Models & Inference" begin
        X = rand(2, 20)

        # Test No-Delay Prediction (Can take a single vector)
        model_nodelay = fit(PseudoInverse(), Identity(), X)
        x0 = [1.0, 1.0]
        Y_traj1 = predict(model_nodelay, x0, 5)
        @test size(Y_traj1) == (2, 1 + 5) # 1 initial + 5 future

        # Test Delay Prediction
        dict_delay = [Identity(); Delay(2) ∘ Identity()]
        model_delay = fit(PseudoInverse(), dict_delay, X)

        # Passing just 1 snapshot should throw an error (needs 3)
        @test_throws ErrorException predict(model_delay, X[:, 1:1], 5)

        # Passing 3 snapshots should work
        Y_traj2 = predict(model_delay, X[:, 1:3], 5)
        @test size(Y_traj2) == (4, 1 + 5) # 4 lifted rows, 1 valid start + 5 future
    end
end
