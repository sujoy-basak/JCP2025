using SSFR
using StaticArrays

Eq = SSFR.EqRHD2D
#------------------------------------------------------------------------------
xmin, xmax = -20.0, 20.0
ymin, ymax = -20.0, 20.0

boundary_condition = (periodic, periodic, periodic, periodic)
γ = 5/3
final_time = 80.0

zero_bv(x,y,t) = SVector(0.0,0.0,0.0,0.0)

initial_value, exact_solution =  Eq.RHD_isentropic_vortex, Eq.exact_RHD_isentropic_vortex

boundary_value = zero_bv

degree = 3
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "no" #"yes"
bflux = evaluate

nx, ny = 20, 20
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 0
save_time_interval = 0.0 #final_time / 3.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.9 #0.95


indicator_model = "model1"

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)

# limiter = setup_limiter_blend(
#                         blend_type = fo_blend(equation),
#                         indicating_variables = Eq.rho_lorentz_p_indicator!,
#                         #indicating_variables = Eq.rho_p_indicator!,
#                         reconstruction_variables = conservative_reconstruction,
#                         indicator_model = indicator_model,
#                         constant_node_factor = 1.0,
#                         amax = 1.0,
#                         #debug_blend = true,
#                         pure_fv = false #pure_fv
#                       )

limiter = setup_limiter_none()

scheme = Scheme(solver, degree, solution_points, correction_function,
                   numerical_flux, bound_limit, limiter, bflux)
                   


param = Parameters(grid_size, cfl, bounds, save_iter_interval,
                   save_time_interval, compute_error_interval,
                   animate = animate,
                   cfl_safety_factor = cfl_safety_factor)


#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme, equation,
                                          ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);

print(sol["errors"])

return sol;