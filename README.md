# Experiments on the strong lottery ticket hypothesis for convolutional neural networks

This is the code used to produce the empyrical results reported in [our paper](https://openreview.net/forum?id=Vjki79-619-). 


## Running

All commands referred should be executed in the project's root directory.

A working installation of [Gurobi Optimizer](https://www.gurobi.com/products/gurobi-optimizer/) is necessary to run the experiments.
To install the other dependencies, start Julia and run
```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

To perform the experiments and generate the plots, run
```
julia --project=. main.jl
```
The script will look for cached solutions in `saves/`.
This search is based on file's names, so renaming them (or using different parameters) will generate solutions from scratch (which can take many hours).
