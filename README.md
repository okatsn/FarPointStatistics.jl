# FarPointStatistics

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://okatsn.github.io/FarPointStatistics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://okatsn.github.io/FarPointStatistics.jl/dev/)
[![Build Status](https://github.com/okatsn/FarPointStatistics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/okatsn/FarPointStatistics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/okatsn/FarPointStatistics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/okatsn/FarPointStatistics.jl)

<!-- Don't have any of your custom contents above; they won't occur if there is no citation. -->

## Documentation Badge is here:

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://okatsn.github.io/FarPointStatistics.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://okatsn.github.io/FarPointStatistics.jl/dev)

> See [Documenter.jl: Documentation Versions](https://documenter.juliadocs.org/dev/man/hosting/#Documentation-Versions)

## Introduction

This is a julia package created using `okatsn`'s preference, and this package is expected to be registered to [okatsn/OkRegistry](https://github.com/okatsn/OkRegistry) for CIs to work properly.

FarPointStatistics.jl provides functions to identify outliers based on common statistical methods.
It handles special values like missing and NaN correctly.

## Examples

```julia
using FarPointStatistics
using DataFrames

# Example data with outliers, missing, and NaN
data = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 10.0, 1.1, 0.9, missing, NaN, -9.0];

# Create a DataFrame
df = DataFrame(value = data);

# Use transform! to add outlier flags
# Functions are curried, so you can pass the threshold first.
transform!(df, :value => outlier_zscore(3) => :is_outlier_z)
transform!(df, :value => outlier_iqr(1.5) => :is_outlier_iqr)
transform!(df, :value => outlier_mad(7) => :is_outlier_mad)
```


## Checklist

- [x] Create an empty repository (namely, `https://github.com/okatsn/FarPointStatistics.jl.git`) on github, and push the local to origin. See [connecting to remote](#tips-for-connecting-to-remote).
- [x] Add `ACCESS_OKREGISTRY` secret in the settings of this repository on Github, or delete both `register.yml` and `TagBot.yml` in `/.github/workflows/`. See [Auto-Registration](#auto-registration).
- [ ] To keep `Manifest.toml` being tracked, delete the lines in `.gitignore`.
- [ ] You might like to register `v0.0.0` in order to `pkg> dev FarPointStatistics` in your environment.




This package is create on 2025-10-17.
