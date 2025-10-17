const expected_behaviors = """
# Expected behaviors:

- if a value is `missing`, the output outlier identifier should be `missing`.
- if a value is `NaN`, it should be identified as `false` (not outlier).

"""


NaNMath.mean(x::Base.SkipMissing) = NaNMath.mean(x |> collect)
NaNMath.std(x::Base.SkipMissing) = NaNMath.std(x |> collect)
NaNMath.median(x::Base.SkipMissing) = NaNMath.median(x |> collect)
NaNMath.median(x::AbstractVector{<:Integer}) = float(x) |> NaNMath.median



function outlier_zscore(data::AbstractVector, threshold=3)
    values = collect(data)

    # Handle edge case: all missing
    clean = skipmissing(values)

    # Filter out NaN and Inf for mean/std calculation
    finite_vals = filter(x -> isfinite(x), clean)

    # If no finite values, return defaults
    if isempty(finite_vals)
        return _default_outlier_result(values)
    end

    mean_val = NaNMath.mean(finite_vals)
    std_dev = NaNMath.std(finite_vals)

    # Build result using predicate function
    predicate(x) = abs((x - mean_val) / std_dev) > threshold
    return _build_outlier_result(values, predicate)
end

"""
# Example

```
transform!(df::DataFrame, :value => outlier_zscore(5) => :isoutlier) # define outlier by a threshold of 5 standard deviation.
```

$expected_behaviors
"""
outlier_zscore(thr::Real) = data -> outlier_zscore(data, thr)

"""
Given `data::AbstractVector`, `outlier_iqr(data; c=1.5)` returns `outlier_indices` of the same
length as `data` indicating the values in `data` lies outside the range of `[Q1, Q3] ± c * IQR`.

$expected_behaviors
"""
function outlier_iqr(data, c=1.5)
    values = collect(data)
    clean = skipmissing(values)

    # Filter out NaN and Inf for quantile calculation
    finite_vals = filter(x -> isfinite(x), clean)

    # If no finite values, handle special values only
    if isempty(finite_vals)
        return _default_outlier_result(values)
    end

    # Calculate Q1 and Q3 from finite values
    q1 = quantile(finite_vals, 0.25)
    q3 = quantile(finite_vals, 0.75)

    # Calculate the Interquartile Range (IQR)
    iqr_value = q3 - q1

    # Define the outlier fences
    lower_bound = q1 - c * iqr_value
    upper_bound = q3 + c * iqr_value

    # Build result using predicate function
    predicate(x) = x < lower_bound || x > upper_bound
    return _build_outlier_result(values, predicate)
end

outlier_iqr(thr::Real) = data -> outlier_iqr(data, thr)

"""
In a perfect normal distribution, 50% of the data lies between approximately -0.67448975 and +0.67448975 standard deviations from the mean.
"""
const std50perc = 0.6744897501960817

"""
Given `data::AbstractVector`, `outlier_mad(data; threshold=7.0, constant=1.4826)` marks values
whose robust z-score based on the Median Absolute Deviation exceeds `threshold`.

Let `med = median(data)` over non-missing entries and `MAD = median(|x_i - med|)`.
Each finite value flags as an outlier when `constant * |(x_i - med)| / MAD > threshold`,
where constant = $std50perc being the standard deviation where 50% of data lies in between, assuming a perfect normal distribution.

Noted that infinite values are excluded in the calculation of `MAD` and `med`.

$expected_behaviors
- When MAD is `Inf`, for any finite data point xᵢ its robust Z-score will be 0 (less than any positive threshold), and thus classified as non-outlier (`false`).
- When a value `x_i` in `data` is infinite, it will be classified as outlier (`true`), regardless of MAD calculation.
- `NaN` will be ignored in calculating `MAD`. However, if `data` are all `NaN`, no data point will be flagged as an outlier (all `false`).
"""
function outlier_mad(data::AbstractVector, threshold=7.0; constant=std50perc)
    values = collect(data)
    clean = skipmissing(values)

    # Filter out Inf values for MAD calculation (they're outliers by rule)
    finite_vals = filter(x -> isfinite(x), clean)

    # If no finite values, return defaults
    if isempty(finite_vals)
        return _default_outlier_result(values)
    end

    med = NaNMath.median(finite_vals)
    deviations = abs.(finite_vals .- med) # this step might produce NaN.
    mad_val = NaNMath.median(deviations)

    thr = float(threshold)

    # Build result using predicate function
    predicate(x) = constant * abs((x - med) / mad_val) > thr
    return _build_outlier_result(values, predicate)
end

outlier_mad(thr::Real; constant=std50perc) = data -> outlier_mad(data, thr; constant=constant)






"""
Helper function to create default result vector for special values only.
Returns `missing` for missing, `false` for NaN, `true` for Inf/-Inf, `false` otherwise.
"""
function _default_outlier_result(values)
    result = Vector{Union{Missing,Bool}}(undef, length(values))
    for (i, x) in enumerate(values)
        if ismissing(x)
            result[i] = missing
        elseif isnan(x)
            result[i] = false
        elseif isinf(x)
            result[i] = true
        else
            result[i] = false
        end
    end
    return result
end

"""
Helper function to build outlier result vector given a predicate function.
Handles special values (missing, NaN, Inf) according to expected_behaviors,
and applies `predicate(x)` to finite values.
"""
function _build_outlier_result(values, predicate::Function)
    result = Vector{Union{Missing,Bool}}(undef, length(values))
    for (i, x) in enumerate(values)
        if ismissing(x)
            result[i] = missing
        elseif isnan(x)
            result[i] = false
        elseif isinf(x)
            result[i] = true
        else
            result[i] = predicate(x)
        end
    end
    return result
end


const outlier_handling_methods = Dict(
    "zscore" => outlier_zscore,
    "iqr" => outlier_iqr,
    "mad" => outlier_mad,
)
