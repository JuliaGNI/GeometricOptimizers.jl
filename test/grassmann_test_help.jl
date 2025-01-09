function grassmann_test_help(result::Bool, N::Integer, n::Integer)
    if N > n
        @test result
    elseif N == n 
        @test !result
    else
        error("N has to be greater than n")
    end
end