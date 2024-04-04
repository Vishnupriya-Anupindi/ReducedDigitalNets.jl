using ReducedDigitalNets
using Documenter

DocMeta.setdocmeta!(ReducedDigitalNets, :DocTestSetup, :(using ReducedDigitalNets); recursive=true)

makedocs(;
    modules=[ReducedDigitalNets],
    authors="Vishnupriya Anupindi and contributors",
    sitename="ReducedDigitalNets.jl",
    format=Documenter.HTML(;
        canonical="https://Vishnupriya-Anupindi.github.io/ReducedDigitalNets.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Vishnupriya-Anupindi/ReducedDigitalNets.jl",
    devbranch="main",
)
