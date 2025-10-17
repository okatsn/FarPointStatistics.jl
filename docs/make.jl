using FarPointStatistics
using Documenter
# using DocumenterCitations
# # 1. Uncomment this line and the CitationBibliography line
# # 2. add docs/src/refs.bib
# # 3. Cite something in refs.bib and add ```@bibliography ``` (in index.md, for example)
# # Please refer https://juliadocs.org/DocumenterCitations.jl/stable/


DocMeta.setdocmeta!(FarPointStatistics, :DocTestSetup, :(using FarPointStatistics); recursive=true)

makedocs(;
    modules=[FarPointStatistics],
    authors="okatsn <okatsn@gmail.com> and contributors",
    repo="https://github.com/okatsn/FarPointStatistics.jl/blob/{commit}{path}#{line}",
    sitename="FarPointStatistics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://okatsn.github.io/FarPointStatistics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    # plugins=[
    #     CitationBibliography(joinpath(@__DIR__, "src", "refs.bib")),
    # ],
)

deploydocs(;
    repo="github.com/okatsn/FarPointStatistics.jl",
    devbranch="main",
)
