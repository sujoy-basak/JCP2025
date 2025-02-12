using StaticArrays
using SSFR
using Plots

# Submodules
Eq = SSFR.EqRHD1D
plotlyjs() # Set backend

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 1.0

degree = 3
γ = 5/3
final_time = 0.35
nx = 200
boundary_condition = (neumann, neumann)
initial_value, exact_solution = Eq.RHD_density_pert, Eq.exact_RHD_density_pert


dummy_bv(x,t) = 0.0
boundary_value = dummy_bv

solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

 
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in RHD

tvbM = 800.0

save_iter_interval = 0
save_time_interval = 0.0 #final_time/5.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

# blend parameters
indicator_model = "model1"
debug_blend =  true #false
cfl_safety_factor = 0.95
pure_fv = false

equation = Eq.get_equation(γ)
#------------------------------------------------------------------------------

grid_size = nx
domain = [xmin, xmax]
problem = Problem(domain, initial_value, boundary_value,
                  boundary_condition, final_time, exact_solution)

# limiter = setup_limiter_blend(
#                               blend_type = fo_blend(equation),
#                               indicating_variables = Eq.rho_lorentz2_p_indicator!,
#                               #indicating_variables = Eq.conservative_indicator!,
#                               reconstruction_variables = conservative_reconstruction,
#                               indicator_model = indicator_model,
#                               constant_node_factor = 1.0,
#                               amax = 1.0,
#                               debug_blend = debug_blend,
#                               pure_fv = pure_fv
#                             )

#
                            
#limiter = setup_limiter_none()
limiter = setup_limiter_tvb(equation; tvbM = tvbM)
scheme = Scheme(solver, degree, solution_points, correction_function,
                numerical_flux, bound_limit, limiter, bflux)
param = Parameters(
                    grid_size, cfl, bounds, save_iter_interval,
                    save_time_interval, compute_error_interval;
                    animate = animate,
                    cfl_safety_factor = cfl_safety_factor,
                  )
#------------------------------------------------------------------------------
problem, scheme, param = ParseCommandLine(problem, param, scheme,
                                          equation, ARGS)
#------------------------------------------------------------------------------
sol = SSFR.solve(equation, problem, scheme, param);