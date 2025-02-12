using SSFR
using StaticArrays

Eq = SSFR.EqRHD2D
#------------------------------------------------------------------------------
xmin, xmax = 0.0, 325.0
ymin, ymax = -45.0, 45.0

boundary_condition =  (dirichlet, dirichlet, reflect, reflect)
γ = 5/3
final_time = 500.0

function newriemann_bv(x,y,t)
  γ = 5/3
  if x < 265
     rho = 1.0
     vx  = 0.0
     vy  = 0.0
     p   = 0.05
  else
     rho = 1.865225080631180
     vx  = -0.196781107378299
     vy  = 0.0
     p   = 0.15
  end
  prim = [rho, vx, vy, p]
  equation = Eq.get_equation(γ)
  U = Eq.prim2con(equation, prim)
  return U
end

initial_value, exact_solution =  Eq.RHD_newriemann2, Eq.exact_RHD_newriemann2
boundary_value = newriemann_bv

degree = 4
solver = "lwfr"
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
bound_limit = "yes"
bflux = evaluate

nx, ny = 325, 90
cfl = 0.0
bounds = ([-Inf],[Inf]) # Not used in Euler
tvbM = 10.0
save_iter_interval = 0 #29
save_time_interval = final_time / 5.0
animate = true # Factor on save_iter_interval or save_time_interval
compute_error_interval = 0

cfl_safety_factor = 0.95


indicator_model = "model1" #

#------------------------------------------------------------------------------
grid_size = [nx, ny]
domain = [xmin, xmax, ymin, ymax]
equation = Eq.get_equation(γ)
problem = Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)




limiter = setup_limiter_blend(
                        blend_type = fo_blend(equation),
                        indicating_variables = Eq.rho_lorentz_p_indicator!,
                        #indicating_variables = Eq.rho_p_indicator!,
                        reconstruction_variables = conservative_reconstruction,
                        indicator_model = indicator_model,
                        constant_node_factor = 1.0,
                        amax = 1.0,
                        pure_fv = false
                      )

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