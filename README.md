### This repository is ispired from an old version of [`Tenkai.jl`](https://github.com/Arpit-Babbar/Tenkai.jl). Significant modifications and adaptations have been made to extend its functionality. 

To run the codes, first make sure you have julia installed in your system.

## Step 1: Set the parameters
In the `run` files in `Examples` directory modify the parameters if needed. For examples set `degree` as 1,2,3, or 4; `final_time`; `boundary_condition` etc.

## Step 2: Activate project environment in Julia
```shell
julia --project=.
```
or by starting plain `julia` REPL and then entering `import Pkg; Pkg.activate(".")`. 

## Step 3: Install all dependencies (only needed the first time)
```julia
julia> import Pkg; Pkg.instantiate()
```

For the first time, to precompile parts of code to local drive, it is also recommended that you run

```julia
julia> using SSFR
```

## Step 4: Run the code
Assuming you have modified the parameters in `run_RHD1D_riemann1.jl` file, run it as
```julia
julia> include("Examples/1d/run_RHD1D_riemann1.jl")
```

## Visualization

For 1-D, you can see `png`, `.txt` and interactive HTML files of the final solution in `output` directory.

For 2-D, plot the solution using visit as

```shell
visit -o output/sol*.vtr
```


Note that, if you have a 4 core CPU, you can use 4 threads by starting REPL as

```shell
julia --project=. --threads=4
```

# Refer us!

If you use these codes for your research work, please cite us as

```bibtex
@article{basak2025bound,
  title={Bound Preserving Lax-Wendroff Flux Reconstruction Method for Special Relativistic Hydrodynamics},
  author={Basak, Sujoy and Babbar, Arpit and Kumar, Harish and Chandrashekar, Praveen},
  journal={Journal of Computational Physics},
  pages={113815},
  year={2025},
  publisher={Elsevier}
}
```
