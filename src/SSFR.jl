module SSFR

src_dir = @__DIR__ # Directory of file
data_dir = "$src_dir/../data/"
# base_dir = "$src_dir/../base"
eq_dir = "$src_dir/equations"
kernels_dir = "$src_dir/kernels"
grid_dir = "$src_dir/grids"
utils_dir = "$src_dir/../utils"
solvers_dir = "$src_dir/solvers"
fr_dir = "$solvers_dir/FR"
lwfr_dir = "$solvers_dir/LW"

export lwfr_dir

examples_dir() = "$src_dir/../Examples"
test_dir = "$src_dir/../test"

export examples_dir # TODO - Remove this export

# TODO - Move to a file "allmodules.jl"
include("$fr_dir/Basis.jl")
include("$grid_dir/CartesianGrids.jl")
include("$eq_dir/Equations.jl")

using .Equations: nvariables, eachvariable, AbstractEquations

export nvariables, eachvariable, AbstractEquations

using .Basis: Vandermonde_lag

include("$fr_dir/FR.jl")

(
using .FR: Problem, Scheme, Parameters, ParseCommandLine, solve, PlotData,
           get_filename, minmod, @threaded, periodic, dirichlet, neumann, dirichlet_neumann, hllc_bc,
           conservative2conservative_reconstruction!,
           conservative2primitive_reconstruction!,
           primitive2conservative_reconstruction!,
           conservative2characteristic_reconstruction!,
           characteristic2conservative_reconstruction!,
           setup_limiter_none,
           reflect, extrapolate, evaluate,
           get_node_vars,get_node_vars_mutable, set_node_vars!,
           add_to_node_vars!, subtract_from_node_vars!,
           multiply_add_to_node_vars!, multiply_add_set_node_vars!,
           get_first_node_vars, get_second_node_vars,
           comp_wise_mutiply_node_vars!,
           setup_limiter_blend,
           setup_limiter_hierarchical,
           ParseCommandLine
)

( # General API to be used in Equation files
export
       get_filename, minmod, @threaded, Vandermonde_lag,
       Problem, Scheme, Parameters,
       ParseCommandLine,
       setup_limiter_none,
       setup_limiter_blend,
       setup_limiter_hierarchical,
       periodic, dirichlet, neumann, reflect, dirichlet_neumann, hllc_bc,
       extrapolate, evaluate,
       update_ghost_values_periodic!,
       get_node_vars,get_node_vars_mutable, set_node_vars!,
       get_first_node_vars, get_second_node_vars,
       add_to_node_vars!, subtract_from_node_vars!,
       multiply_add_to_node_vars!, multiply_add_set_node_vars!,
       comp_wise_mutiply_node_vars!
)

( # Methods that the user has to define in Equation module
import .FR: flux, prim2con, prim2con!, con2prim, con2prim!, eigmatrix, check_admissibility
)

( # Skeleton Methods that will be defined and extended in FR1D, FR2D
import .FR: update_ghost_values_periodic!,
            update_ghost_values_u1!,
            update_ghost_values_fn_blend!,
            modal_smoothness_indicator,
            modal_smoothness_indicator_gassner,
            set_initial_condition!,
            compute_cell_average!,
            get_cfl,
            compute_time_step,
            compute_face_residual!,
            apply_bound_limiter!,
            apply_bound_limiter_extreme!,
            apply_tvb_limiter!,
            apply_tvb_limiterβ!,
            setup_limiter_tvb,
            setup_limiter_tvbβ,
            Blend,
            set_blend_dt!,
            fo_blend,
            mh_blend,
            zhang_shu_flux_fix,
            limit_slope,
            no_upwinding_x,
            is_admissible,
            is_admissible_extreme,
            conservative_indicator!,
            apply_hierarchical_limiter!,
            Hierarchical,
            setup_arrays_lwfr,
            solve_lwfr,
            compute_error,
            initialize_plot,
            write_soln!,
            create_aux_cache,
            write_poly,
            write_soln!,
            post_process_soln
)

# TODO - This situation disallows us from doing an allmodules.jl thing

include("$fr_dir/FR1D.jl")

( # Methods defined for 1-D in FR1D
import .FR1D: update_ghost_values_periodic!,
              update_ghost_values_u1!,
              update_ghost_values_fn_blend!,
              modal_smoothness_indicator,
              modal_smoothness_indicator_gassner,
              set_initial_condition!,
              compute_cell_average!,
              get_cfl,
              compute_time_step,
              compute_face_residual!,
              apply_bound_limiter!,
              apply_tvb_limiter!,
              apply_tvb_limiterβ!,
              setup_limiter_tvb,
              setup_limiter_tvbβ,
              # blending limiter methods
              Blend,
              set_blend_dt!,
              fo_blend,
              mh_blend,
              zhang_shu_flux_fix,
              limit_slope,
              no_upwinding_x,
              is_admissible,
              is_admissible_extreme,
              apply_hierarchical_limiter!,
              Hierarchical,
              compute_error,
              initialize_plot,
              write_soln!,
              create_aux_cache,
              write_poly,
              write_soln!,
              post_process_soln
)

include("$fr_dir/FR2D.jl")

( # Methods defined for 2-D in FR2D
import .FR2D: update_ghost_values_periodic!,
              update_ghost_values_u1!,
              update_ghost_values_fn_blend!,
              modal_smoothness_indicator,
              # modal_smoothness_indicator_gassner, # yet to be implemented
              set_initial_condition!,
              compute_cell_average!,
              get_cfl,
              compute_time_step,
              compute_face_residual!,
              apply_bound_limiter!,
              apply_bound_limiter_extreme!,
              apply_tvb_limiter!,
              apply_tvb_limiterβ!,
              setup_limiter_tvb,
              setup_limiter_tvbβ,
              Blend,
              set_blend_dt!,
              fo_blend,
              mh_blend,
              blending_flux_factors,
              zhang_shu_flux_fix,
              limit_slope,
              no_upwinding_x,
              is_admissible,
              is_admissible_extreme,
              apply_hierarchical_limiter!,
              Hierarchical, # not yet implemented
              compute_error,
              initialize_plot,
              write_soln!,
              create_aux_cache,
              write_poly,
              write_soln!,
              post_process_soln
)


# 1D methods t

# Pack blending methods into containers for user API

# KLUDGE - Move reconstruction named tuples to FR.jl

conservative_reconstruction = (;
                               conservative2recon! = conservative2conservative_reconstruction!,
                               recon2conservative! = conservative2conservative_reconstruction!,
                               recon_string = "conservative"
                              )

primitive_reconstruction = (;
                            conservative2recon! = conservative2primitive_reconstruction!,
                            recon2conservative! = primitive2conservative_reconstruction!,
                            recon_string = "primitive"
                           )

characteristic_reconstruction = (;
                                 conservative2recon! = conservative2characteristic_reconstruction!,
                                 recon2conservative! = characteristic2conservative_reconstruction!,
                                 recon_string = "characteristic"
                                )

(
export conservative_reconstruction, primitive_reconstruction,
       characteristic_reconstruction, conservative_indicator!,
       fo_blend, mh_blend
)

( # FR API
export
       setup_limiter_tvb,
       setup_limiter_tvbβ,
       ParseCommandLine
)

include("$lwfr_dir/LWFR.jl")

( # Skeleton methods to be defined and extended in LWFR1D, LWFR2D
import .LWFR: setup_arrays_lwfr,
              compute_cell_residual_1!, compute_cell_residual_2!,
              compute_cell_residual_3!, compute_cell_residual_4!,
              update_ghost_values_lwfr!,
              eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
              extrap_bflux!
)

include("$lwfr_dir/LWFR1D.jl")

( # methods extended to LWFR1D
import .LWFR1D: setup_arrays_lwfr,
                compute_cell_residual_1!, compute_cell_residual_2!,
                compute_cell_residual_3!, compute_cell_residual_4!,
                update_ghost_values_lwfr!,
                eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
                extrap_bflux!, flux
)

include("$lwfr_dir/LWFR2D.jl")

( # methods extended to LWFR2D
import .LWFR2D: setup_arrays_lwfr,
                compute_cell_residual_1!, compute_cell_residual_2!,
                compute_cell_residual_3!, compute_cell_residual_4!,
                update_ghost_values_lwfr!,
                eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
                extrap_bflux!, flux
)

# ( # LWFR API exported
# export setup_arrays_lwfr,
#        compute_cell_residual_1!, compute_cell_residual_2!,
#        compute_cell_residual_3!, compute_cell_residual_4!,
#        update_ghost_values_lwfr!,
#        eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
#        extrap_bflux!
# )

## Example equation files

# 1D
include("$eq_dir/EqRHD1D.jl")

# 2D
include("$eq_dir/EqRHD2D.jl")

# Utils

include("$utils_dir/Utils.jl")

end # module
