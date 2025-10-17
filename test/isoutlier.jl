# test_isoutlier.jl
using Statistics
# using NaNMath
using DataFrames


@testset "Outlier Detection Tests" begin
    @testset "Z-Score Method" begin
        # Basic functionality test
        data = [1.0, 2, 3, 100, 2, 3]
        result = outlier_zscore(data)
        @test result == [false, false, false, false, false, false]

        # Test with specific threshold
        result_thr2 = outlier_zscore(data, 2)
        @test result_thr2 == [false, false, false, true, false, false]

        # Test with lower threshold to catch more outliers
        data2 = [1.0, 2, 3, 10, 2, 3]
        result_thr1 = outlier_zscore(data2, 1)
        @test result_thr1 == [false, false, false, true, false, false]

        # Test with missing values
        data_missing = [1.0, 2, 3, missing, 100, 2, 3]
        result_missing = outlier_zscore(data_missing)
        @test (result_missing |> skipmissing |> collect) == [false, false, false, missing, false, false, false] |> skipmissing |> collect
        @test result_missing[4] |> ismissing

        # Test with all same values
        data_same = [5.0, 5, 5, 5]
        result_same = outlier_zscore(data_same)
        # When all values are the same, std=0, so z-scores are NaN
        # NaN comparison always returns false
        @test all(.!result_same)

        # Test with empty vector
        @test isempty(outlier_zscore(Float64[]))

        # Test curried function
        outlier_detector = outlier_zscore(2.5)
        @test outlier_detector(data) == outlier_zscore(data, 2.5)

        # Test with DataFrame
        df = DataFrame(value=[1.0, 2, 3, 100, 2, 3])
        transform!(df, :value => outlier_zscore(3) => :isoutlier)
        @test df.isoutlier == [false, false, false, false, false, false]
    end

    @testset "Z-Score Method: Rule - `NaN` is never an outlier" begin
        # NaN should be identified as false (not outlier)
        data_with_nan = [1.0, 2, 3, NaN, 100.0]
        result_nan = outlier_zscore(data_with_nan)
        @test result_nan[4] == false  # NaN at position 4 should be false

        # All NaN values
        data_all_nan = [NaN, NaN, NaN]
        result_all_nan = outlier_zscore(data_all_nan)
        @test all(.!result_all_nan)  # All should be false

        # Mix of NaN and outliers with low threshold
        data_nan_outlier = [1.0, 2, 3, NaN, 50, NaN, 2]
        result_mixed = outlier_zscore(data_nan_outlier, 1)
        @test result_mixed[4] == false  # First NaN should be false
        @test result_mixed[6] == false  # Second NaN should be false

        # NaN with zero std (all other values the same)
        data_nan_const = [5.0, 5, NaN, 5]
        result_nan_const = outlier_zscore(data_nan_const)
        @test result_nan_const[3] == false  # NaN should be false even when std=0
    end

    @testset "Z-Score Method: Rule - `missing` propagates" begin
        # Missing values should propagate as missing in the output
        data_with_missing = Vector{Union{Float64,Missing}}([1.0, 2, missing, 3, 100.0])
        result_missing = outlier_zscore(data_with_missing)
        @test isequal(result_missing[3], missing)  # Missing should remain missing

        # Multiple missing values
        data_multi_missing = Vector{Union{Float64,Missing}}([missing, 1.0, 2, 3, missing, 100.0])
        result_multi = outlier_zscore(data_multi_missing)
        @test isequal(result_multi[1], missing)
        @test isequal(result_multi[5], missing)

        # All missing
        data_all_missing = [missing, missing, missing]
        result_all_missing = outlier_zscore(data_all_missing)
        @test all(ismissing.(result_all_missing))

        # Complex case with missing, NaN, and outliers
        data_complex = Vector{Union{Float64,Missing}}([1.0, 2, missing, NaN, 100.0, 3])
        result_complex = outlier_zscore(data_complex, 2)
        @test isequal(result_complex[3], missing)  # missing propagates
        @test result_complex[4] == false  # NaN is false
        # Check structure: [false, false, missing, false, true/false depending on threshold, false]
        @test ismissing(result_complex[3])
        @test result_complex[4] == false
    end

    @testset "Z-Score Method: Rule - `Inf` and `-Inf` handling" begin
        # Inf should be flagged as outlier (z-score will be Inf)
        data_with_inf = [1.0, 2, 3, 4, Inf]
        result_inf = outlier_zscore(data_with_inf)
        @test result_inf[5] == true  # Inf should be outlier

        # -Inf should be flagged as outlier
        data_with_ninf = [-Inf, 1.0, 2, 3, 4]
        result_ninf = outlier_zscore(data_with_ninf)
        @test result_ninf[1] == true  # -Inf should be outlier

        # Both Inf and -Inf
        data_both_inf = [-Inf, 1.0, 2, 3, Inf]
        result_both = outlier_zscore(data_both_inf)
        @test result_both[1] == true
        @test result_both[5] == true

        # Mix of Inf, NaN, and missing
        data_special_mix = Vector{Union{Float64,Missing}}([1.0, Inf, NaN, missing, -Inf])
        result_special = outlier_zscore(data_special_mix)
        @test result_special[2] == true   # Inf is outlier
        @test result_special[3] == false  # NaN is false
        @test isequal(result_special[4], missing)  # missing propagates
        @test result_special[5] == true   # -Inf is outlier
    end

    @testset "Z-Score Method: Edge cases with special values" begin
        # Single element
        @test outlier_zscore([10.0]) == [false]
        @test isequal(outlier_zscore([missing]), [missing])
        @test outlier_zscore([NaN]) == [false]
        @test outlier_zscore([Inf]) == [true]  # Single Inf is an outlier

        # Empty vector
        @test isempty(outlier_zscore(Float64[]))
        @test isempty(outlier_zscore(Union{Float64,Missing}[]))
    end

    @testset "IQR Method" begin
        # Basic functionality test
        data = [1, 2, 3, 4, 5, 6, 20]
        result = outlier_iqr(data)
        # Q1 = 2, Q3 = 6, IQR = 4, lower = 2 - 1.5*4 = -4, upper = 6 + 1.5*4 = 12
        # So only 20 should be flagged as outlier
        @test result == [false, false, false, false, false, false, true]

        # Test with custom coefficient
        result_c3 = outlier_iqr(data, 3)
        # Lower = 2 - 3*4 = -10, upper = 6 + 3*4 = 18
        # With higher c, fewer outliers (20 still an outlier)
        @test result_c3 == [false, false, false, false, false, false, true]

        # Test with more extremes
        data2 = [1, 2, 3, 4, 5, 6, 20, -15]
        result2 = outlier_iqr(data2)
        # Same Q1, Q3, IQR, but now both 20 and -15 should be outliers
        @test result2 == [false, false, false, false, false, false, true, true]

        # Test with missing values
        data_missing = [1, 2, missing, 3, 4, 5, 6, 20]
        result_missing = outlier_iqr(data_missing)
        # Missing should be treated as non-outlier
        @test (result_missing |> skipmissing |> collect) == ([false, false, missing, false, false, false, false, true] |> skipmissing |> collect)
        @test result_missing[3] |> ismissing

        # Test with all same values
        data_same = [5, 5, 5, 5]
        result_same = outlier_iqr(data_same)
        # When all values are the same, IQR=0, should handle this special case
        @test all(.!result_same)
    end

    @testset "IQR Method: Rule - `NaN` is never an outlier" begin
        # NaN should be identified as false (not outlier)
        data_with_nan = [1, 2, 3, NaN, 100.0]
        result_nan = outlier_iqr(data_with_nan)
        @test result_nan[4] == false  # NaN at position 4 should be false

        # All NaN values
        data_all_nan = [NaN, NaN, NaN]
        result_all_nan = outlier_iqr(data_all_nan)
        @test all(.!result_all_nan)  # All should be false

        # Mix of NaN and outliers
        data_nan_outlier = [1, 2, 3, NaN, 50, NaN, 2]
        result_mixed = outlier_iqr(data_nan_outlier)
        @test result_mixed[4] == false  # First NaN should be false
        @test result_mixed[6] == false  # Second NaN should be false
    end

    @testset "IQR Method: Rule - `missing` propagates" begin
        # Missing values should propagate as missing in the output
        data_with_missing = Vector{Union{Float64,Missing}}([1, 2, missing, 3, 100.0])
        result_missing = outlier_iqr(data_with_missing)
        @test isequal(result_missing[3], missing)  # Missing should remain missing

        # Multiple missing values
        data_multi_missing = Vector{Union{Float64,Missing}}([missing, 1, 2, 3, missing, 100.0])
        result_multi = outlier_iqr(data_multi_missing)
        @test isequal(result_multi[1], missing)
        @test isequal(result_multi[5], missing)

        # All missing
        data_all_missing = [missing, missing, missing]
        result_all_missing = outlier_iqr(data_all_missing)
        @test all(ismissing.(result_all_missing))

        # Complex case with missing, NaN, and outliers
        data_complex = Vector{Union{Float64,Missing}}([1, 2, missing, NaN, 100.0, 3])
        result_complex = outlier_iqr(data_complex)
        @test isequal(result_complex[3], missing)  # missing propagates
        @test result_complex[4] == false  # NaN is false
        @test isequal(result_complex, [false, false, missing, false, true, false])
    end

    @testset "IQR Method: Rule - `Inf` and `-Inf` handling" begin
        # Inf should be flagged as outlier
        data_with_inf = [1, 2, 3, 4, Inf]
        result_inf = outlier_iqr(data_with_inf)
        @test result_inf[5] == true  # Inf should be outlier

        # -Inf should be flagged as outlier
        data_with_ninf = [-Inf, 1, 2, 3, 4]
        result_ninf = outlier_iqr(data_with_ninf)
        @test result_ninf[1] == true  # -Inf should be outlier

        # Both Inf and -Inf
        data_both_inf = [-Inf, 1, 2, 3, Inf]
        result_both = outlier_iqr(data_both_inf)
        @test result_both[1] == true
        @test result_both[5] == true
    end
end# test_isoutlier.jl


@testset "outlier_iqr vs implementation details" begin
    # This test verifies the implementation matches the expected behavior
    data = [1, 2, 3, 4, 5, 20, 30]

    # Manual calculation of IQR detection
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    expected = [(x < lower_bound || x > upper_bound) for x in data]

    @test outlier_iqr(data) == expected

end

@testset "Test Inf." begin
    # This test verifies the implementation matches the expected behavior
    # Inf is always flagged as outlier, and 30 is also outside IQR bounds
    data = [1, 2, 3, 4, 5, 20, Inf, 30]
    @test outlier_iqr(data) == [false, false, false, false, false, false, true, true]

    # This test case no longer throws an error with the new implementation
    # Inf is filtered out before quantile calculation, so NaN issue is avoided
    data = [1, 2, 5, 999, Inf]
    result = outlier_iqr(data)
    @test result[5] == true  # Inf is always an outlier
    # The other values depend on IQR calculation from finite values

end


# Assume the function to be tested, `outlier_mad`, is defined in your project.

@testset "outlier_mad function tests" begin

    #================================================#
    ## 1. Basic Functionality (Sanity Check)
    #================================================#
    @testset "Basic functionality on finite numbers" begin
        # Case with a clear outlier
        data1 = [1, 2, 3, 2, 1, 2, 3, 100.0]
        @test outlier_mad(data1, 3.5) == [false, false, false, false, false, false, false, true]

        # Case with a clear negative outlier
        data2 = [-100.0, 1, 2, 3, 2, 1, 2, 3]
        @test outlier_mad(data2, 3.5) == [true, false, false, false, false, false, false, false]

        # Case with no outliers
        data3 = [1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1.0]
        @test outlier_mad(data3, 3.5) == falses(length(data3))

        # Case with multiple symmetric outliers
        data4 = [-100, 1, 2, 3, 2, 1, 2, 3, 100.0]
        @test outlier_mad(data4, 3.5) == [true, false, false, false, false, false, false, false, true]
    end

    #================================================#
    ## 2. Tests for Special Values as per Specification
    #================================================#
    @testset "Edge cases" begin
        # Empty vector should return an empty boolean vector
        @test outlier_mad(Float64[], 3.5) == Bool[]
        @test isequal(outlier_mad(Float64[]), Bool[])
        @test isequal(outlier_mad(Union{Float64,Missing}[]), Union{Bool,Missing}[])

        # Single element vector has no outliers
        @test outlier_mad([10.0], 3.5) == [false]
        @test outlier_mad([10.0]) == [false]
        @test isequal(outlier_mad([missing]), [missing])
        @test outlier_mad([NaN]) == [false]
        @test outlier_mad([Inf]) == [true]

        # Vector with all identical elements (MAD = 0)
        # This is a critical test for division-by-zero errors.
        data_const = [5.0, 5.0, 5.0, 5.0, 5.0]
        @test outlier_mad(data_const, 3.5) == [false, false, false, false, false]

        # Vector with two alternating values
        data_alt = [1.0, 10.0, 1.0, 10.0, 1.0, 10.0]
        @test outlier_mad(data_alt, 1.0) == falses(length(data_alt))

        # Case where MAD is 0 but special values are also present.
        data_const_special = Vector{Union{Float64,Missing}}([5.0, 5.0, Inf, NaN, missing])
        expected_const_special = Vector{Union{Bool,Missing}}([false, false, true, false, missing])
        @test isequal(outlier_mad(data_const_special, 3.5), expected_const_special)

    end

    @testset "Rule: `Inf` or `-Inf` is always an outlier" begin
        # When an Inf exists, MAD is Inf, so all finite points are non-outliers.
        data_inf = [1, 2, 3, Inf, 5, 6.0]
        @test outlier_mad(data_inf, 3.5) == [false, false, false, true, false, false]

        data_ninf = [-Inf, 1, 2, 3, 5, 6.0]
        @test outlier_mad(data_ninf, 3.5) == [true, false, false, false, false, false]

        # When an Inf is present with another extreme finite value.
        # The Inf is an outlier. The MAD is calculated on the remaining finite data,
        # which will detect the other outlier as well.
        data_inf_and_outlier = [1, 2, 3, 200.0, Inf]
        @test outlier_mad(data_inf_and_outlier, 3.5) == [false, false, false, true, true]
    end

    @testset "Rule: `NaN` is never an outlier" begin
        # `NaN` is ignored for MAD calculation, and its own result is `false`.
        data_nan_outlier = [1, 2, 3, NaN, 100.0]
        @test outlier_mad(data_nan_outlier, 3.5) == [false, false, false, false, true]

        # If all data points are `NaN`, all results are `false`.
        data_all_nan = [NaN, NaN, NaN]
        @test outlier_mad(data_all_nan, 3.5) == [false, false, false]
    end

    @testset "Rule: `missing` propagates" begin
        # The test vector must be able to hold `missing`.
        data_with_missing = Vector{Union{Float64,Missing}}([1, 2, 100.0, missing])
        expected_output = Vector{Union{Bool,Missing}}([false, false, true, missing])

        # IMPORTANT: `==` fails with `missing`. `isequal` is required.
        @test isequal(outlier_mad(data_with_missing, 3.5), expected_output)

        # If all data points are `missing`, all results are `missing`.
        data_all_missing = [missing, missing, missing]
        expected_all_missing = [missing, missing, missing]
        @test isequal(outlier_mad(data_all_missing, 3.5), expected_all_missing)
    end


    #================================================#
    ## 3. Data Types
    #================================================#
    @testset "Data types" begin
        # Should work with integer vectors
        data_int = [1, 2, 3, 2, 1, 100]
        @test outlier_mad(data_int, 3.5) == [false, false, false, false, false, true]

        # Should work with mixed float and integer vectors
        data_mixed = [1, 2.0, 3, 2.0, 1, 100.0]
        @test outlier_mad(data_mixed, 3.5) == [false, false, false, false, false, true]
    end

    #================================================#
    ## 4. Special Values (NaN, Inf, missing)
    #================================================#
    @testset "Special floating point and missing values" begin
        # NaN values should be ignored and not flagged as outliers
        data_nan = [1, 2, 3, NaN, 100.0]
        @test outlier_mad(data_nan, 3.5) == [false, false, false, false, true]

        # Inf should always be an outlier (unless all data is Inf)
        data_inf = [1, 2, 3, 2, 1, Inf]
        @test outlier_mad(data_inf, 3.5) == [false, false, false, false, false, true]

        # -Inf should also always be an outlier
        data_ninf = [-Inf, 1, 2, 3, 2, 1]
        @test outlier_mad(data_ninf, 3.5) == [true, false, false, false, false, false]

        # `missing` values should be ignored and not flagged as outliers
        data_missing = [1, 2, 3, missing, 100.0]
        @test isequal(outlier_mad(data_missing, 3.5), [false, false, false, missing, true])

        # A complex mix of special values
        data_mix_special = [1, 2, -Inf, 3, NaN, 100.0, missing, Inf]
        @test isequal(outlier_mad(data_mix_special, 3.5), [false, false, true, false, false, true, missing, true])

        # This test combines all rules into a single, complex vector.
        data_complex = Vector{Union{Float64,Missing}}([0, 1, 2, 100.0, -100.0, Inf, -Inf, NaN, missing])

        # Expected logic:
        # - `Inf` and `-Inf` -> `true`
        # - `NaN` -> `false`
        # - `missing` -> `missing`
        # - Outliers are detected in the remaining finite set [0, 1, 2, 100.0, -100.0]
        expected_complex = Vector{Union{Bool,Missing}}([false, false, false, true, true, true, true, false, missing])

        @test isequal(outlier_mad(data_complex, 3.5), expected_complex)
    end

    #================================================#
    ## 5. Threshold Parameter
    #================================================#
    @testset "Threshold parameter" begin
        # M-score for the value `15.0` is ~5.17 in this dataset
        data = [1, 2, 3, 4, 5, 15.0]

        # Test the default threshold (7.0 as per the prompt)
        @test outlier_mad(data) == [false, false, false, false, false, false] # 5.17 < 7.0

        # Test with a specific threshold that makes it an outlier
        @test outlier_mad(data, 5.0) == [false, false, false, false, false, true] # 5.17 > 5.0

        # Test with a threshold that makes it NOT an outlier
        @test outlier_mad(data, 6.0) == [false, false, false, false, false, false] # 5.17 < 6.0
    end

end
