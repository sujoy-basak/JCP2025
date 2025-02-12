using UnPack
using SSFR
using Printf

grid_size_array = zeros(Int,11)
grid_size_array[1] = 50 #8
for i in 2:11
    grid_size_array[i] = grid_size_array[i-1]+50
    # grid_size_array[i] = 2 * grid_size_array[i-1]
end

@show grid_size_array

level = length(grid_size_array)

Error_l1, Error_l2, Error_linf = zeros(level), zeros(level), zeros(level)
Order_l1, Order_l2, Order_linf = zeros(level-1), zeros(level-1), zeros(level-1)

examples_dir_ = "$(SSFR.src_dir)/../Examples"
sol = include("$examples_dir_/1d/run_RHD1D_isentropic.jl")
# sol = include("$examples_dir_/1d/run_RHD1D_smooth.jl")

@unpack errors = sol
Error_l1[1], Error_l2[1], Error_linf[1] = errors["l1_error"], errors["l2_error"], errors["linf_error"]


for k=2:level
    global grid_size = grid_size_array[k]
    local param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = false,
                   cfl_safety_factor = cfl_safety_factor)
    local sol = SSFR.solve(equation, problem, scheme, param);
    @unpack errors = sol
    Error_l1[k], Error_l2[k], Error_linf[k] = errors["l1_error"], errors["l2_error"], errors["linf_error"]
    Order_l1[k-1] = log(Error_l1[k-1]/Error_l1[k])/log(grid_size_array[k]/grid_size_array[k-1])
    Order_l2[k-1] = log(Error_l2[k-1]/Error_l2[k])/log(grid_size_array[k]/grid_size_array[k-1])
    Order_linf[k-1] = log(Error_linf[k-1]/Error_linf[k])/log(grid_size_array[k]/grid_size_array[k-1])
    println("l1_convergence rate = ", Order_l1[1])
    println("l2_convergence rate = ", Order_l2[k-1])
    println("linf_convergence rate = ", Order_linf[k-1])
end

    @printf("Grids      l1 Error          l1 Order                 l2 Error                l2 Order                 linf Error                    linf Order\n")
    @printf("%2.0f   &    %2.5e     &      -             &          %2.5e       &        -           &          %2.5e           &          - \n",grid_size_array[1],  Error_l1[1], Error_l2[1], Error_linf[1] )
for k=2:level
    @printf("%2.0f   &    %2.5e     &      %2.5f       &      %2.5e      &       %2.5f      &       %2.5e        &          %2.5f\n",grid_size_array[k],  Error_l1[k], Order_l1[k-1], Error_l2[k], Order_l2[k-1], Error_linf[k], Order_linf[k-1] )
end

# println("l1 Errors are $Error_l1, l2 Errors are $Error_l2, linf Errors are $Error_linf")