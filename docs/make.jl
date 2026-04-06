using KoopmanModeDecomposition
using Documenter

DocMeta.setdocmeta!(KoopmanModeDecomposition, :DocTestSetup, :(using KoopmanModeDecomposition); recursive=true)

makedocs(;
    modules=[KoopmanModeDecomposition],
    authors="Adam Uchytil",
    sitename="KoopmanModeDecomposition.jl",
    format=Documenter.HTML(;
        canonical="https://uchytil.github.io/KoopmanModeDecomposition.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/uchytil/KoopmanModeDecomposition.jl",
    devbranch="main",
)
