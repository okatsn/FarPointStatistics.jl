module FarPointStatistics

using NaNMath, Statistics
include("isoutlier.jl")
export outlier_zscore, outlier_iqr, outlier_mad



end
