module LWFR1D

( # methods extended in this module
import ..SSFR: setup_arrays_lwfr,
               compute_cell_residual_1!, compute_cell_residual_2!,
               compute_cell_residual_3!, compute_cell_residual_4!,
               update_ghost_values_lwfr!,
               eval_bflux1!, eval_bflux2!, eval_bflux3!, eval_bflux4!,
               extrap_bflux!, flux, is_admissible, is_admissible_extreme
)

(
using ..SSFR: periodic, dirichlet, neumann, reflect, extrapolate, evaluate,
              update_ghost_values_periodic!,
              update_ghost_values_fn_blend!,
              get_node_vars, get_node_vars_mutable, set_node_vars!,
              add_to_node_vars!, subtract_from_node_vars!,
              multiply_add_to_node_vars!, multiply_add_set_node_vars!,
              comp_wise_mutiply_node_vars!
)

using UnPack
using TimerOutputs
using Polyester
using MuladdMacro
using OffsetArrays
using StaticArrays
using LinearAlgebra: dot

using ..FR: @threaded, alloc_for_threads
using ..Equations: AbstractEquations, nvariables, eachvariable
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#-------------------------------------------------------------------------------
# Allocate solution arrays needed by LWFR in 1d
#-------------------------------------------------------------------------------
function setup_arrays_lwfr(grid, scheme, eq::AbstractEquations{1})
   gArray(nvar,nx) = OffsetArray(zeros(nvar,nx+2),
                                 OffsetArrays.Origin(1,0))
   gArray(nvar,n1,nx) = OffsetArray(zeros(nvar,n1,nx+2),
                                    OffsetArrays.Origin(1,1,0))
   # Allocate memory
   @unpack degree = scheme
   nd   = degree + 1
   nx   = grid.size
   nvar = nvariables(eq)
   u1  = gArray(nvar, nd, nx)
   ua  = gArray(nvar, nx)
   res = gArray(nvar, nd, nx)
   Fb  = gArray(nvar, 2, nx)
   Ub  = gArray(nvar, 2, nx)

   if degree == 1
      cell_data_size = 6
      eval_data_size = 6
      # eval_data_exp_size = 3 #2
   elseif degree == 2
      cell_data_size = 8 #10 #8
      eval_data_size = 6
      # eval_data_exp_size = 3 #2
   elseif degree == 3
      cell_data_size = 13 #17 #13
      eval_data_size = 16
      # eval_data_exp_size = 3#2
   elseif degree == 4
      cell_data_size = 15 #19 #15
      eval_data_size = 18
      # eval_data_exp_size = 3#2
   else
      @assert false "Degree not implemented"
   end

   MArr = MArray{Tuple{nvariables(eq), nd}, Float64}
   cell_data = alloc_for_threads(MArr, cell_data_size)

   MArr = MArray{Tuple{nvariables(eq), 1}, Float64}
   eval_data = alloc_for_threads(MArr, eval_data_size)

   # MVec = MVector{nvariables(eq), Float64}
   # eval_data_exp = alloc_for_threads(MVec, eval_data_exp_size)

   cache = (; u1, ua, res, Fb, Ub, cell_data, eval_data) #, eval_data_exp)
   return cache
end

#-------------------------------------------------------------------------------
# Fill ghost values
#-------------------------------------------------------------------------------
function update_ghost_values_lwfr!(problem, scheme, eq, grid, aux, op, cache,
                                   t, dt)
   @timeit aux.timer "Update ghost values" begin
   @unpack Fb, Ub = cache
   update_ghost_values_periodic!(eq, problem, Fb, Ub)

   if problem.periodic_x
      return nothing
   end

   nx = grid.size
   @unpack degree, xg, wg = op
   nd = degree + 1
   dx, xf = grid.dx, grid.xf
   nvar = eq.nvar
   @unpack boundary_value, boundary_condition = problem
   left, right = boundary_condition
   refresh!(u) = fill!(u,0.0)

   ub, fb = zeros(nvar), zeros(nvar)

   # For Dirichlet bc, use upwind flux at faces by assigning both physical
   # and ghost cells through the bc.
   if left == dirichlet
      x = xf[1]
      for l=1:nd
         tq = t + xg[l]*dt
         ubvalue = boundary_value(x,tq)

         fbvalue = flux(x, ubvalue, eq)
         for n=1:nvar
            ub[n] += ubvalue[n] * wg[l]
            fb[n] += fbvalue[n] * wg[l]
         end
      end
      for n=1:nvar
         Ub[n, 1, 1] = Ub[n, 2, 0] = ub[n]
         Fb[n, 1, 1] = Fb[n, 2, 0] = fb[n]
      end
   elseif left == neumann
      for n=1:nvar
         Ub[n, 2, 0] = Ub[n, 1, 1]
         Fb[n, 2, 0] = Fb[n, 1, 1]
      end
   elseif left == reflect
      # velocity reflected back in opposite direction and density is same
      for n=1:nvar
         Ub[n, 2, 0] = Ub[n, 1, 1]
         Fb[n, 2, 0] = Fb[n, 1, 1]
      end
      Ub[2, 2, 0] = -Ub[2, 2, 0] # velocity reflected back
      Fb[1, 2, 0], Fb[3, 2, 0] = -Fb[1, 2, 0], -Fb[3, 2, 0] # vel multiple term
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   refresh!.((ub, fb))
   if right == dirichlet
      x  = xf[nx+1]
      for l=1:nd
         tq = t + xg[l]*dt
         ubvalue = boundary_value(x,tq)
         fbvalue = flux(x, ub, eq)
         for n=1:nvar
            ub[n] += ubvalue[n] * wg[l]
            fb[n] += fbvalue[n] * wg[l]
         end
      end
      for n=1:nvar
         Ub[n, 2, nx] = Ub[n, 1, nx+1] = ub[n]
         Fb[n, 2, nx] = Fb[n, 1, nx+1] = fb[n]
      end
   elseif right == neumann
      for n=1:nvar
         Ub[n, 1, nx+1] = Ub[n, 2, nx]
         Fb[n, 1, nx+1] = Fb[n, 2, nx]
      end
   elseif right == reflect
      # velocity reflected back in opposite direction and density is same
      for n=1:nvar
         Ub[n, 1, nx+1] = Ub[n, 2, nx]
         Fb[n, 1, nx+1] = Fb[n, 2, nx]
      end
      Ub[2, 1, nx+1] = -Ub[2, 1, nx+1] # velocity reflected back
      Fb[1, 1, nx+1], Fb[3, 1, nx+1] = (-Fb[1, 1, nx+1],
                                        -Fb[3, 1, nx+1]) # vel multiple term

   else
      println("Incorrect bc specified at right.")
      @assert false
   end

   if scheme.limiter.name == "blend"
      update_ghost_values_fn_blend!(eq, problem, grid, aux)
   end

   end # timer
   return nothing
end

#------------------------------------------------------------------------------
# Boundary flux functions as place holders
#------------------------------------------------------------------------------
function eval_bflux1!(eq::AbstractEquations{1}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
   nothing
end

function eval_bflux2!(eq::AbstractEquations{1}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
   nothing
end

function eval_bflux3!(eq::AbstractEquations{1}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
   nothing
end

function eval_bflux4!(eq::AbstractEquations{1}, grid, cell_data, eval_data,
                      xg, Vl, Vr, F, G, Fb, aux)
   nothing
end
#------------------------------------------------------------------------------
# Compute cell residual for degree=1 case and for all real cells
#------------------------------------------------------------------------------

function compute_cell_residual_1!(eq::AbstractEquations{1}, grid, op, scheme,
                                  aux, t, dt, u1, res, Fb, Ub, cache)   ########## What is done here?
   @unpack xg, wg, Dm, D1, Vl, Vr = op
   nd         = length(xg)
   nx         = grid.size
   @unpack bflux_ind = scheme.bflux
   @unpack blend = aux
   refresh!(u) = fill!(u,0.0)
   # Pre-allocate local variables

   @unpack cell_data, eval_data = cache

   F, f, U, ut, up, um = cell_data[Threads.threadid()]
   refresh!.((res, Ub, Fb)) # Reset previously used variables to zero

   @inbounds for el_x in Base.OneTo(nx) # Loop over cells
      dx   = grid.dx[el_x]
      xc   = grid.xc[el_x]
      lamx = dt / dx
      refresh!(ut)

      # Solution points
      for i in Base.OneTo(nd)
         x_ = xc - 0.5 * dx + xg[i] * dx
         u_node = get_node_vars(u1, eq, i, el_x)
         # Compute flux at all solution points
         flux1 = flux(x_, u_node, eq)
         set_node_vars!(F, flux1, eq, i)
         set_node_vars!(f, flux1, eq, i)
         for ii in Base.OneTo(nd) # ut = -lamx * DmT * f
            multiply_add_to_node_vars!(ut, -lamx * Dm[ii,i], flux1, eq, ii)
         end
         set_node_vars!(um, u_node, eq, i)
         set_node_vars!(up, u_node, eq, i)
         set_node_vars!(U , u_node, eq, i)
      end
      for i in Base.OneTo(nd)
         ut_node = get_node_vars(ut, eq, i)
         add_to_node_vars!(up, ut_node, eq, i)
         subtract_from_node_vars!(um, ut_node, eq, i)
      end

      for i in Base.OneTo(nd)
         x_ = xc - 0.5 * dx + xg[i] * dx
         um_node = get_node_vars_mutable(um, eq, i)
         up_node = get_node_vars_mutable(up, eq, i)

         fm = flux(x_, um_node, eq)
         fp = flux(x_, up_node, eq)
         multiply_add_to_node_vars!(F,
                                     0.25, fp,
                                    -0.25, fm,
                                     eq , i )
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(U,
                                    0.5, ut_node,
                                    eq , i )
         F_node = get_node_vars(F, eq, i)
         for ix in Base.OneTo(nd)
            multiply_add_to_node_vars!(res, lamx * D1[ix,i], F_node, eq, ix, el_x)
         end
      end
      u = @view u1[:,:,el_x]
      r = @view res[:,:,el_x]
      blend.blend_cell_residual!(el_x, eq, scheme, aux, lamx, dt, dx,
                                 grid.xf[el_x], op, u1 , u, cache.ua, f, r)
      # Interpolate to faces
      for i in Base.OneTo(nd)
         U_node = get_node_vars(U, eq, i)
         multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, el_x)
         multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, el_x)
      end
      if bflux_ind == extrapolate
         for i in Base.OneTo(nd)
            Fl_node = get_node_vars(F, eq, i)
            Fr_node = get_node_vars(F, eq, i)
            multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, el_x)
            multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, el_x)
         end
      else
         ul, ur, upl, upr, uml, umr = eval_data[Threads.threadid()]
         refresh!.((ul,ur,upl,uml,umr,upr))
         xl, xr = grid.xf[el_x], grid.xf[el_x+1]
         for i in Base.OneTo(nd)
            u_node  = get_node_vars(u1, eq, i, el_x)
            up_node = get_node_vars(up, eq, i)
            um_node = get_node_vars(um, eq, i)
            multiply_add_to_node_vars!(ul , Vl[i], u_node , eq, 1)
            multiply_add_to_node_vars!(ur , Vr[i], u_node , eq, 1)
            multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, 1)
            multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, 1)
            multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, 1)
            multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, 1)
         end
         # IDEA - Try this in TVB limiter as well
         ul_node  = get_node_vars(ul,  eq, 1)
         ur_node  = get_node_vars(ur,  eq, 1)
         upl_node = get_node_vars(upl, eq, 1)
         upr_node = get_node_vars(upr, eq, 1)
         uml_node = get_node_vars(uml, eq, 1)
         umr_node = get_node_vars(umr, eq, 1)
         fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

         set_node_vars!(Fb, fl, eq, 1, el_x)
         set_node_vars!(Fb, fr, eq, 2, el_x)

         fml, fmr = flux(xl, uml_node, eq), flux(xr, umr_node, eq)

         fpl, fpr = flux(xl, upl_node, eq), flux(xr, upr_node, eq)
         multiply_add_to_node_vars!(Fb,
                                       0.25, fpl,
                                      -0.25, fml,
                                       eq , 1, el_x )
         multiply_add_to_node_vars!(Fb,
                                       0.25, fpr,
                                      -0.25, fmr,
                                       eq , 2, el_x )
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=2 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_2!(eq::AbstractEquations{1}, grid, op, scheme,
                                  aux, t, dt, u1, res, Fb, Ub, cache)
   @unpack xg, wg, Dm, D1, Vl, Vr = op
   nd         = length(xg)
   nx         = grid.size
   @unpack bflux_ind = scheme.bflux
   @unpack blend = aux
   refresh!(u) = fill!(u,0.0)

   @unpack cell_data, eval_data = cache

   F, U, f, ft, ut, utt, up, um = cell_data[Threads.threadid()]

   refresh!.([res, Ub, Fb])

   @inbounds for cell in Base.OneTo(nx) # Loop over cells
      dx = grid.dx[cell]
      xc = grid.xc[cell]
      lamx = dt/dx
      refresh!(ut); refresh!(utt); refresh!(ft)
      # Solution points
      for i=1:nd
         x_ = xc - 0.5 * dx + xg[i] * dx
         u_node = get_node_vars(u1, eq, i, cell)
         # Compute flux at all solution points
         flux1 = flux(x_, u_node, eq)
         set_node_vars!(F, flux1, eq, i)
         set_node_vars!(f, flux1, eq, i)
         for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(ut, -lamx * Dm[ix,i], flux1, eq, ix)
         end
         set_node_vars!(um, u_node, eq, i)
         set_node_vars!(up, u_node, eq, i)
         set_node_vars!(U , u_node, eq, i)
      end

      for i in Base.OneTo(nd)
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(up,  1.0, # KLUDGE - Avoid redundant multiplication
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(um, -1.0, # KLUDGE - Avoid redundant multiplication
                                    ut_node, eq, i)
      end

      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         um_node = get_node_vars_mutable(um, eq, i)
         up_node = get_node_vars_mutable(up, eq, i)

         fm = flux(x_, um_node, eq)
         fp = flux(x_, up_node, eq)
         multiply_add_to_node_vars!(ft, 0.5, fp, -0.5, fm, eq, i)
         ft_node = get_node_vars(ft, eq, i)
         multiply_add_to_node_vars!(F,
                                    0.5, ft_node,
                                    eq  , i )
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(U,
                                    0.5, ut_node,
                                    eq , i )
         for ix in Base.OneTo(nd) # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(utt, -lamx * Dm[ix,i], ft_node, eq,
                                       ix)
         end
      end
      # computes ftt, gtt and puts them in respective place; no need to store
     
      for i in Base.OneTo(nd) # Loop over solution points
         utt_node = get_node_vars(utt, eq, i)
         multiply_add_to_node_vars!(um,
                                    0.5, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(up,
                                    0.5, utt_node,
                                    eq , i )
      end

      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         utt_node = get_node_vars(utt, eq, i)

         up_node = get_node_vars(up, eq, i)
         um_node = get_node_vars(um, eq, i)

         fp = flux(x_, up_node, eq)
         fm = flux(x_, um_node, eq)
         f_node = get_node_vars(f, eq, i)
         multiply_add_to_node_vars!(F, # ftt = fp - 2.0*f + fm; F += 1/6*ftt
                                    -1.0/3.0, f_node,
                                     1.0/6.0, fp, fm,
                                     eq, i)
         multiply_add_to_node_vars!(U,  # U += 1/6*utt
                                   1.0/6.0, utt_node,
                                   eq, i)
         F_node = get_node_vars(F, eq, i)
         for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(res, lamx * D1[ix,i], F_node, eq,
                                       ix, cell)
         end
      end
      u = @view u1[:,:,cell]
      r = @view res[:,:,cell]
      blend.blend_cell_residual!(cell, eq, scheme, aux, lamx, dt, dx,
                                 grid.xf[cell], op, u1 , u, cache.ua, f, r)
      # Interpolate to faces
      # KLUDGE - Make this into a function
      for i in Base.OneTo(nd)
         U_node = get_node_vars(U, eq, i)
         multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, cell)
         multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, cell)
      end
      if bflux_ind == extrapolate
         # KLUDGE - Make this into a function
         for i in Base.OneTo(nd)
            Fl_node = get_node_vars(F, eq, i)
            Fr_node = get_node_vars(F, eq, i)
            multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, cell)
         end
      else
         ul, ur, upl, upr, uml, umr = eval_data[Threads.threadid()]
         refresh!.((ul,ur,upl,uml,umr,upr))
         xl, xr = grid.xf[cell], grid.xf[cell+1]
         for i in Base.OneTo(nd)
            u_node  = get_node_vars(u1, eq, i, cell)
            up_node = get_node_vars(up, eq, i)
            um_node = get_node_vars(um, eq, i)
            multiply_add_to_node_vars!(ul , Vl[i], u_node , eq, 1)
            multiply_add_to_node_vars!(ur , Vr[i], u_node , eq, 1)
            multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, 1)
            multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, 1)
            multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, 1)
            multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, 1)
         end
         ul_node  = get_node_vars(ul,  eq, 1)
         ur_node  = get_node_vars(ur,  eq, 1)
         upl_node = get_node_vars(upl, eq, 1)
         upr_node = get_node_vars(upr, eq, 1)
         uml_node = get_node_vars(uml, eq, 1)
         umr_node = get_node_vars(umr, eq, 1)
         fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

         set_node_vars!(Fb, fl, eq, 1, cell)
         set_node_vars!(Fb, fr, eq, 2, cell)

         fml, fmr = flux(xl, uml_node, eq), flux(xr, umr_node, eq)
         fpl, fpr = flux(xl, upl_node, eq), flux(xr, upr_node, eq)
         multiply_add_to_node_vars!(Fb, # ft = 0.5*(fp-fm); Fb+= 0.5*ft
                                     0.25, fpl,
                                    -0.25, fml,
                                     eq , 1, cell )
         multiply_add_to_node_vars!(Fb, # ft = 0.5*(fp-fm); Fb+= 0.5*ft
                                     0.25, fpr,
                                    -0.25, fmr,
                                     eq , 2, cell )
         multiply_add_to_node_vars!(Fb, # ftt = fp - 2.0*f + fm; F += 1/6*ftt
                                     -1.0/3.0, fl,
                                      1.0/6.0, fpl, fml,
                                      eq , 1, cell )
         multiply_add_to_node_vars!(Fb, ## ftt = fp - 2.0*f + fm; F += 1/6*ftt
                                     -1.0/3.0, fr,
                                      1.0/6.0, fpr, fmr,
                                      eq , 2, cell )
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=3 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_3!(eq::AbstractEquations{1}, grid, op, scheme,
                                  aux, t, dt, u1, res, Fb, Ub, cache)
   @unpack xg, wg, Dm, D1, Vl, Vr = op
   nd         = length(xg)
   nx         = grid.size
   @unpack bflux_ind = scheme.bflux
   @unpack blend = aux
   refresh!(u) = fill!(u,0.0)

   @unpack cell_data, eval_data = cache

   F, U, f, um, up, umm, upp, ft, ftt, fttt, ut, utt, uttt = cell_data[Threads.threadid()]

   refresh!.([res, Fb, Ub])

   @inbounds for cell=1:nx # Loop over cells
      dx = grid.dx[cell]
      xc = grid.xc[cell]
      lamx = dt / dx
      refresh!(ut); refresh!(utt); refresh!(uttt)
      refresh!(ft); refresh!(ftt), refresh!(fttt)
      # Solution points
      for i=1:nd
         x_ = xc - 0.5 * dx + xg[i] * dx
         u_node = get_node_vars(u1, eq, i, cell)
         # Compute flux at all solution points
         flux1 = flux(x_, u_node, eq)
         set_node_vars!(F, flux1, eq, i)
         set_node_vars!(f, flux1, eq, i)
         for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(ut, -lamx * Dm[ix,i], flux1, eq, ix)
         end
         set_node_vars!(um, u_node, eq, i)
         set_node_vars!(up, u_node, eq, i)
         set_node_vars!(umm, u_node, eq, i)
         set_node_vars!(upp, u_node, eq, i)
         set_node_vars!(U , u_node, eq, i)
      end

      # computes and stores ft, gt and puts them in respective place
      for i in Base.OneTo(nd) # Loop over solution points
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(um, -1.0, # KLUDGE - Avoid redundant multiplication
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(up,  1.0, # KLUDGE - Avoid redundant multiplication
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(umm, -2.0,
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(upp,  2.0,
                                    ut_node, eq, i)
      end

      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         um_node = get_node_vars(um, eq, i)
         up_node = get_node_vars(up, eq, i)

         fm = flux(x_, um_node, eq)
         fp = flux(x_, up_node, eq)
         umm_node = get_node_vars(umm, eq, i)
         upp_node = get_node_vars(upp, eq, i)

         fmm = flux(x_, umm_node, eq)
         fpp = flux(x_, upp_node, eq)
         # ft = 1/12 * (-fpp + 8*fp - 8*fm + fmm)
         multiply_add_to_node_vars!(ft,  1.0/12.0, -1.0, fpp, 8.0 , fp,
                                                   -8.0, fm , 1.0 , fmm,
                                    eq, i)
         # KLUDGE - Can you avoid redundant mutiplication by 1.0?
         ft_node = get_node_vars(ft, eq, i)
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(F, 0.5, ft_node, eq, i) # F += 0.5*ft
         multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i) # U += 0.5*ut

         for ix in Base.OneTo(nd) # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(utt, -lamx * Dm[ix,i], ft_node, eq,
                                       ix)
         end
      end
      # computes ftt, gtt and puts them in respective place and stores them
      for i in Base.OneTo(nd) # Loop over solution points
         utt_node = get_node_vars(utt, eq, i)

         multiply_add_to_node_vars!(um,
                                    0.5, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(up,
                                    0.5, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(U,  # U += 1/6*utt
                                   1.0/6.0, utt_node,
                                   eq, i)
      end
      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         up_node = get_node_vars(up, eq, i)
         um_node = get_node_vars(um, eq, i)

         fp = flux(x_, up_node, eq)
         fm = flux(x_, um_node, eq)
         f_node = get_node_vars(f, eq, i)
         # ftt = fp - 2.0*f + fm
         multiply_add_to_node_vars!(ftt, 1.0, fp, -2.0 , f_node, 1.0, fm, eq, i)
         ftt_node = get_node_vars(ftt, eq, i)
         # F += 1/6*ftt
         multiply_add_to_node_vars!(F,
                                    1.0/6.0, ftt_node,
                                    eq, i)
         for ii in Base.OneTo(nd) # uttt[n] = -lamx * Dm * ftt[n] for each n=1:nvar
            multiply_add_to_node_vars!(uttt, -lamx * Dm[ii,i], ftt_node, eq,
                                       ii)
         end
      end
      # computes fttt, gttt and puts them in respective place; no need to store
      for i=1:nd # Loop over solution points
         utt_node = get_node_vars(utt, eq, i)
         uttt_node = get_node_vars(uttt, eq, i)
         multiply_add_to_node_vars!(U, # U += 1.0/24.0 * uttt
                                    1.0/24.0, uttt_node,
                                    eq, i)

         multiply_add_to_node_vars!(um,
                                    -1.0/6.0, uttt_node, # um -= 1.0/6.0*uttt
                                    eq, i)
         multiply_add_to_node_vars!(up,
                                     1.0/6.0, uttt_node, # up += 1.0/6.0*uttt
                                     eq, i)
         multiply_add_to_node_vars!(umm, # umm += 2.0*utt - 4.0/3.0*uttt
                                     2.0, utt_node,
                                    -4.0/3.0, uttt_node,
                                    eq, i)
         multiply_add_to_node_vars!(upp, # upp += 2.0*utt + 4.0/3.0*uttt
                                     2.0, utt_node,
                                     4.0/3.0, uttt_node,
                                     eq, i)
      end

      for i=1:nd # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         um_node = get_node_vars(um, eq, i)
         fm = flux(x_, um_node, eq)
         up_node = get_node_vars(up, eq, i)
         fp = flux(x_, up_node, eq)
         umm_node = get_node_vars(umm, eq, i)
         fmm = flux(x_, umm_node, eq)
         upp_node = get_node_vars(upp, eq, i)
         fpp = flux(x_, upp_node, eq)
         multiply_add_to_node_vars!(fttt,  0.5,  1.0, fpp, -2.0 , fp,
                                                 2.0, fm , -1.0 , fmm,
                                    eq, i)
         fttt_node = get_node_vars(fttt, eq, i)
         multiply_add_to_node_vars!(F, # F += 1.0/24.0 * fttt_node
                                    1.0/24.0, fttt_node,
                                    eq, i)
         F_node = get_node_vars(F, eq, i)
         for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(res, lamx * D1[ix,i], F_node, eq,
                                       ix, cell)
         end
      end
      u = @view u1[:,:,cell]
      r = @view res[:,:,cell]

      blend.blend_cell_residual!(cell, eq, scheme, aux, lamx, dt, dx,
                                 grid.xf[cell], op, u1 , u, cache.ua, f, r)
      # Interpolate to faces
      for i in Base.OneTo(nd)
         U_node = get_node_vars(U, eq, i)
         multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, cell)
         multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, cell)
      end
      if bflux_ind == extrapolate
         for i in Base.OneTo(nd)
            Fl_node = get_node_vars(F, eq, i)
            Fr_node = get_node_vars(F, eq, i)
            multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, cell)
         end
      else
         ( ul, ur, upl, upr, uml, umr, uppl, uppr, umml, ummr, ftl,
           ftr, fttl, fttr, ftttl, ftttr) = eval_data[Threads.threadid()]
         refresh!.((ul,ur,upl,uml,umr,upr, uppl, uppr, umml, ummr))
         xl, xr = grid.xf[cell], grid.xf[cell+1]
         for i in Base.OneTo(nd)
            u_node   = get_node_vars(u1, eq, i, cell)
            up_node  = get_node_vars(up, eq, i)
            um_node  = get_node_vars(um, eq, i)
            upp_node = get_node_vars(upp, eq, i)
            umm_node = get_node_vars(umm, eq, i)
            multiply_add_to_node_vars!(ul , Vl[i], u_node , eq, 1)
            multiply_add_to_node_vars!(ur , Vr[i], u_node , eq, 1)

            multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, 1)
            multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, 1)
            multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, 1)
            multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, 1)

            multiply_add_to_node_vars!(uppl, Vl[i], upp_node, eq, 1)
            multiply_add_to_node_vars!(uppr, Vr[i], upp_node, eq, 1)
            multiply_add_to_node_vars!(umml, Vl[i], umm_node, eq, 1)
            multiply_add_to_node_vars!(ummr, Vr[i], umm_node, eq, 1)
         end
         ul_node  = get_node_vars(ul , eq, 1)
         ur_node  = get_node_vars(ur , eq, 1)
         upl_node = get_node_vars(upl, eq, 1)
         upr_node = get_node_vars(upr, eq, 1)
         uml_node = get_node_vars(uml, eq, 1)
         umr_node = get_node_vars(umr, eq, 1)

         uppl_node = get_node_vars(uppl, eq, 1)
         uppr_node = get_node_vars(uppr, eq, 1)
         umml_node = get_node_vars(umml, eq, 1)
         ummr_node = get_node_vars(ummr, eq, 1)

         fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

         set_node_vars!(Fb, fl, eq, 1, cell)
         set_node_vars!(Fb, fr, eq, 2, cell)

         fml, fmr = flux(xl, uml_node, eq), flux(xr, umr_node, eq)
         fpl, fpr = flux(xl, upl_node, eq), flux(xr, upr_node, eq)
         fmml, fmmr = flux(xl, umml_node, eq), flux(xr, ummr_node, eq)
         fppl, fppr = flux(xl, uppl_node, eq), flux(xr, uppr_node, eq)
         # ftl = 1.0/12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
         multiply_add_set_node_vars!(ftl, 1.0/12.0, -1.0, fppl, 8.0, fpl ,
                                                    -8.0, fml , 1.0, fmml,
                                     eq, 1)
         # ftr = 1.0/12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)
         multiply_add_set_node_vars!(ftr, 1.0/12.0, -1.0, fppr, 8.0, fpr,
                                                   -8.0, fmr , 1.0, fmmr,
                                    eq, 1)
         # fttl  = fpl - 2.0 * fl + fml
         multiply_add_set_node_vars!(fttl, 1.0, fpl, -2.0 , fl, 1.0, fml, eq, 1)
         # fttr  = fpr - 2.0 * fr + fmr
         multiply_add_set_node_vars!(fttr, 1.0, fpr, -2.0 , fr, 1.0, fmr, eq, 1)
         # ftttl = 0.5 * (fppl - 2.0 * fpl + 2.0 * fml - fmml)
         multiply_add_set_node_vars!(ftttl,  0.5, 1.0, fppl, -2.0 , fpl,
                                                 2.0, fml , -1.0 , fmml,
                                    eq, 1)
         # ftttr = 0.5 * (fppr - 2.0 * fpr + 2.0 * fmr - fmmr)
         multiply_add_set_node_vars!(ftttr,  0.5, 1.0, fppr, -2.0 , fpr,
                                                 2.0, fmr , -1.0 , fmmr,
                                    eq, 1)

         ftl_node  = get_node_vars(ftl,  eq, 1)
         ftr_node  = get_node_vars(ftr,  eq, 1)
         fttl_node  = get_node_vars(fttl,  eq, 1)
         fttr_node  = get_node_vars(fttr,  eq, 1)
         ftttl_node  = get_node_vars(ftttl,  eq, 1)
         ftttr_node  = get_node_vars(ftttr,  eq, 1)

         # F = f + 0.5 * ftr + (1.0/6.0) * fttr + (1.0/24.0) * ftttr
         multiply_add_to_node_vars!(Fb,
                                    0.5, ftl_node,
                                    1.0/6.0, fttl_node,
                                    1.0/24.0, ftttl_node,
                                    eq, 1, cell)
         multiply_add_to_node_vars!(Fb,
                                    0.5, ftr_node,
                                    1.0/6.0, fttr_node,
                                    1.0/24.0, ftttr_node,
                                    eq, 2, cell)
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=4 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_4!(eq::AbstractEquations{1}, grid, op, scheme,
                                  aux, t, dt, u1, res, Fb, Ub, cache)
   @unpack xg, Dm, D1, Vl, Vr = op
   nd         = length(xg)
   nx         = grid.size
   @unpack bflux_ind = scheme.bflux
   @unpack blend = aux
   refresh!(u) = fill!(u,0.0)

   @unpack cell_data, eval_data = cache

   (F, U, f, um, up, umm, upp, ft, ftt, fttt, ftttt, ut, utt, uttt,
    utttt) = cell_data[Threads.threadid()]

   refresh!.((res, Fb, Ub))

   @inbounds for cell=1:nx # Loop over cells
      dx = grid.dx[cell]
      xc = grid.xc[cell]
      lamx = dt/dx

      refresh!(ut); refresh!(utt); refresh!(uttt); refresh!(utttt)
      refresh!(ft); refresh!(ftt); refresh!(fttt); refresh!(ftttt)
      # Solution points
      for i=1:nd
         x_ = xc - 0.5 * dx + xg[i] * dx
         u_node = get_node_vars(u1, eq, i, cell)

         # Compute flux at all solution points
         flux1 = flux(x_, u_node, eq)
         set_node_vars!(F, flux1, eq, i)
         set_node_vars!(f, flux1, eq, i)
         for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(ut, -lamx * Dm[ix,i], flux1, eq, ix)
         end

         set_node_vars!(um, u_node, eq, i)
         set_node_vars!(up, u_node, eq, i)
         set_node_vars!(umm, u_node, eq, i)
         set_node_vars!(upp, u_node, eq, i)
         set_node_vars!(U , u_node, eq, i)
      end

      # computes and stores ft, gt and puts them in respective place
      for i in Base.OneTo(nd) # Loop over solution points
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(um, -1.0, # KLUDGE - Avoid redundant multiplication
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(up,  1.0, # KLUDGE - Avoid redundant multiplication
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(umm, -2.0,
                                    ut_node, eq, i)
         multiply_add_to_node_vars!(upp,  2.0,
                                    ut_node, eq, i)
      end
      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         um_node = get_node_vars(um, eq, i)
         up_node = get_node_vars(up, eq, i)
         fm = flux(x_, um_node, eq)
         fp = flux(x_, up_node, eq)
         umm_node = get_node_vars(umm, eq, i)
         upp_node = get_node_vars(upp, eq, i)
         fmm = flux(x_, umm_node, eq)
         fpp = flux(x_, upp_node, eq)
         # ft = 1/12 * (-fpp + 8*fp - 8*fm + fmm)
         multiply_add_to_node_vars!(ft,  1.0/12.0, -1.0, fpp, 8.0 , fp,
                                                   -8.0, fm , 1.0 , fmm,
                                    eq, i)
         # KLUDGE - Can you avoid redundant mutiplication by 1.0?
         ft_node = get_node_vars(ft, eq, i)
         ut_node = get_node_vars(ut, eq, i)
         multiply_add_to_node_vars!(F, 0.5, ft_node, eq, i) # F += 0.5*ft
         multiply_add_to_node_vars!(U, 0.5, ut_node, eq, i) # U += 0.5*ut

         for ix in Base.OneTo(nd) # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(utt, -lamx * Dm[ix,i], ft_node, eq,
                                       ix)
         end
      end

      # computes ftt, gtt and puts them in respective place and stores them
      for i in Base.OneTo(nd) # Loop over solution points
         utt_node = get_node_vars(utt, eq, i)
         multiply_add_to_node_vars!(um,
                                    0.5, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(up,
                                    0.5, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(umm,
                                    2.0, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(upp,
                                    2.0, utt_node,
                                    eq , i )
         multiply_add_to_node_vars!(U,  # U += 1/6*utt
                                    1.0/6.0, utt_node,
                                    eq, i)
      end
      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         up_node = get_node_vars(up, eq, i)
         um_node = get_node_vars(um, eq, i)
         upp_node = get_node_vars(upp, eq, i)
         umm_node = get_node_vars(umm, eq, i)

         fp = flux(x_, up_node, eq)
         fm = flux(x_, um_node, eq)
         fpp = flux(x_, upp_node, eq)
         fmm = flux(x_, umm_node, eq)
         f_node = get_node_vars(f, eq, i)
         # ftt = fp - 2.0*f + fm
         multiply_add_to_node_vars!(ftt,  1.0/12.0, -1.0 , fpp, 16.0 , fp,
                                                    -30.0, f_node,
                                                     16.0, fm , -1.0 , fmm,
                                    eq, i)
         ftt_node = get_node_vars(ftt, eq, i)
         # F += 1/6*ftt
         multiply_add_to_node_vars!(F,
                                    1.0/6.0, ftt_node,
                                    eq, i)
         for ix in Base.OneTo(nd) # uttt[n] = -lamx * Dm * ftt[n] for each n=1:nvar
            multiply_add_to_node_vars!(uttt, -lamx * Dm[ix,i], ftt_node, eq,
                                       ix)
         end
      end
      # computes and stores fttt, gttt; and puts them in respective place
      for i=1:nd # Loop over solution points
         uttt_node = get_node_vars(uttt, eq, i)
         multiply_add_to_node_vars!(U, # U += 1.0/24.0 * uttt
                                    1.0/24.0, uttt_node,
                                    eq, i)
         multiply_add_to_node_vars!(um,
                                    -1.0/6.0, uttt_node, # um -= 1.0/6.0*uttt
                                    eq, i)
         multiply_add_to_node_vars!(up,
                                     1.0/6.0, uttt_node, # up += 1.0/6.0*uttt
                                     eq, i)
         multiply_add_to_node_vars!(umm, # umm += 2.0*utt - 4.0/3.0*uttt
                                    -4.0/3.0, uttt_node,
                                    eq, i)
         multiply_add_to_node_vars!(upp, # upp += 2.0*utt + 4.0/3.0*uttt
                                     4.0/3.0, uttt_node,
                                     eq, i)
      end
      for i=1:nd # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         um_node = get_node_vars(um, eq, i)
         fm = flux(x_, um_node, eq)
         up_node = get_node_vars(up, eq, i)
         fp = flux(x_, up_node, eq)
         umm_node = get_node_vars(umm, eq, i)
         fmm = flux(x_, umm_node, eq)
         upp_node = get_node_vars(upp, eq, i)
         fpp = flux(x_, upp_node, eq)
         multiply_add_to_node_vars!(fttt,  0.5,  1.0, fpp, -2.0 , fp,
                                                 2.0, fm , -1.0 , fmm,
                                    eq, i)
         fttt_node = get_node_vars(fttt, eq, i)
         multiply_add_to_node_vars!(F, # F += 1.0/24.0 * fttt_node
                                    1.0/24.0, fttt_node,
                                    eq, i)
         for ix in Base.OneTo(nd) # uttt[n] = -lamx * Dm * ftt[n] for each n=1:nvar
            multiply_add_to_node_vars!(utttt, -lamx * Dm[ix,i], fttt_node, eq,
                                       ix)
         end
      end
      for i in Base.OneTo(nd) # Loop over solution points
         utttt_node = get_node_vars(utttt, eq, i)
         multiply_add_to_node_vars!(U, # U += 1.0/24.0 * uttt
                                    1.0/120.0, utttt_node,
                                    eq, i)
         multiply_add_to_node_vars!(um,
                                    1.0/24.0, utttt_node, # um += 1.0/24.0*utttt
                                    eq, i)
         multiply_add_to_node_vars!(up,
                                    1.0/24.0, utttt_node, # um += 1.0/24.0*utttt
                                    eq, i)
         multiply_add_to_node_vars!(umm,
                                    2.0/3.0, utttt_node, # um += 1.0/24.0*utttt
                                    eq, i)
         multiply_add_to_node_vars!(upp,
                                    2.0/3.0, utttt_node, # um += 1.0/24.0*utttt
                                    eq, i)
      end
      for i in Base.OneTo(nd) # Loop over solution points
         x_ = xc - 0.5 * dx + xg[i] * dx
         f_node = get_node_vars(f, eq, i)
         um_node = get_node_vars(um, eq, i)
         fm = flux(x_, um_node, eq)
         up_node = get_node_vars(up, eq, i)
         fp = flux(x_, up_node, eq)
         umm_node = get_node_vars(umm, eq, i)
         fmm = flux(x_, umm_node, eq)
         upp_node = get_node_vars(upp, eq, i)
         fpp = flux(x_, upp_node, eq)
         multiply_add_to_node_vars!(ftttt,  0.5, 1.0, fpp, -4.0 , fp,
                                                 6.0, f_node,
                                                -4.0, fm ,  1.0 , fmm,
                                    eq, i)
         ftttt_node = get_node_vars(ftttt, eq, i)
         multiply_add_to_node_vars!(F, # F += 1.0/24.0 * fttt_node
                                    1.0/120.0, ftttt_node,
                                    eq, i)
         F_node = get_node_vars(F, eq, i)
         for ix in 1:nd # utt[n] = -lamx * Dm * ft[n] for each n=1:nvar
            multiply_add_to_node_vars!(res, lamx * D1[ix,i], F_node, eq,
                                       ix, cell)
         end
      end
      # computes fttt, gttt and puts them in respective place; no need to store
      u = @view u1[:,:,cell]
      r = @view res[:,:,cell]
      blend.blend_cell_residual!(cell, eq, scheme, aux, lamx, dt, dx,
                                 grid.xf[cell], op, u1 , u, cache.ua, f, r)
      # Interpolate to faces
      for i in Base.OneTo(nd)
         U_node = get_node_vars(U, eq, i)
         multiply_add_to_node_vars!(Ub, Vl[i], U_node, eq, 1, cell)
         multiply_add_to_node_vars!(Ub, Vr[i], U_node, eq, 2, cell)
      end
      if bflux_ind == extrapolate
         for i in Base.OneTo(nd)
            Fl_node = get_node_vars(F, eq, i)
            Fr_node = get_node_vars(F, eq, i)
            multiply_add_to_node_vars!(Fb, Vl[i], Fl_node, eq, 1, cell)
            multiply_add_to_node_vars!(Fb, Vr[i], Fr_node, eq, 2, cell)
         end
      else
         ( ul, ur, upl, upr, uml, umr, uppl, uppr, umml, ummr, ftl,
           ftr, fttl, fttr, ftttl, ftttr , fttttl, fttttr) = eval_data[Threads.threadid()]
         refresh!.((ul,ur,upl,uml,umr,upr, uppl, uppr, umml, ummr, ftl, ftr,
                    fttl, fttr, ftttl, ftttr, fttttl, fttttr))
         xl, xr = grid.xf[cell], grid.xf[cell+1]
         for i in Base.OneTo(nd)
            u_node  = get_node_vars(u1, eq, i, cell)
            up_node = get_node_vars(up, eq, i)
            um_node = get_node_vars(um, eq, i)
            upp_node = get_node_vars(upp, eq, i)
            umm_node = get_node_vars(umm, eq, i)

            multiply_add_to_node_vars!(ul , Vl[i], u_node , eq, 1)
            multiply_add_to_node_vars!(ur , Vr[i], u_node , eq, 1)

            multiply_add_to_node_vars!(upl, Vl[i], up_node, eq, 1)
            multiply_add_to_node_vars!(upr, Vr[i], up_node, eq, 1)
            multiply_add_to_node_vars!(uml, Vl[i], um_node, eq, 1)
            multiply_add_to_node_vars!(umr, Vr[i], um_node, eq, 1)

            multiply_add_to_node_vars!(uppl, Vl[i], upp_node, eq, 1)
            multiply_add_to_node_vars!(uppr, Vr[i], upp_node, eq, 1)
            multiply_add_to_node_vars!(umml, Vl[i], umm_node, eq, 1)
            multiply_add_to_node_vars!(ummr, Vr[i], umm_node, eq, 1)
         end

         ul_node  = get_node_vars(ul , eq, 1)
         ur_node  = get_node_vars(ur , eq, 1)
         upl_node = get_node_vars(upl, eq, 1)
         upr_node = get_node_vars(upr, eq, 1)
         uml_node = get_node_vars(uml, eq, 1)
         umr_node = get_node_vars(umr, eq, 1)

         uppl_node = get_node_vars(uppl, eq, 1)
         uppr_node = get_node_vars(uppr, eq, 1)
         umml_node = get_node_vars(umml, eq, 1)
         ummr_node = get_node_vars(ummr, eq, 1)

         fl, fr = flux(xl, ul_node, eq), flux(xr, ur_node, eq)

         set_node_vars!(Fb, fl, eq, 1, cell)
         set_node_vars!(Fb, fr, eq, 2, cell)

         fml, fmr = flux(xl, uml_node, eq), flux(xr, umr_node, eq)
         fpl, fpr = flux(xl, upl_node, eq), flux(xr, upr_node, eq)

         fmml, fmmr = flux(xl, umml_node, eq), flux(xr, ummr_node, eq)
         fppl, fppr = flux(xl, uppl_node, eq), flux(xr, uppr_node, eq)

         # ftl = 1.0/12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
         multiply_add_to_node_vars!(ftl, 1.0/12.0, -1.0, fppl, 8.0, fpl ,
                                                   -8.0, fml , 1.0, fmml,
                                    eq, 1)
         # ftr = 1.0/12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)
         multiply_add_to_node_vars!(ftr, 1.0/12.0, -1.0, fppr, 8.0, fpr ,
                                                   -8.0, fmr , 1.0, fmmr,
                                    eq, 1)
         # fttl = 1.0/12.0 * (-fppl + 16.0 * fpl - 30.0 * fl + 16.0 * fml - fmml)
         multiply_add_to_node_vars!(fttl, 1.0/12.0, -1.0  , fppl, 16.0, fpl,
                                                    -30.0 , fl,
                                                     16.0 , fml, -1.0 , fmml,
                                    eq, 1)
         # fttr = 1.0/12.0 * (-fppr + 16.0 * fpr - 30.0 * fr + 16.0 * fmr - fmmr)
         multiply_add_to_node_vars!(fttr, 1.0/12.0, -1.0  , fppr, 16.0, fpr,
                                                    -30.0 , fr,
                                                     16.0 , fmr, -1.0 , fmmr,
                                    eq, 1)
         # ftttl = 0.5 * (fppl - 2.0 * fpl + 2.0 * fml - fmml)
         multiply_add_to_node_vars!(ftttl,  0.5, 1.0, fppl, -2.0 , fpl,
                                                 2.0, fml , -1.0 , fmml,
                                    eq, 1)
         # ftttr = 0.5 * (fppr - 2.0 * fpr + 2.0 * fmr - fmmr)
         multiply_add_to_node_vars!(ftttr,  0.5, 1.0, fppr, -2.0 , fpr,
                                                 2.0, fmr , -1.0 , fmmr,
                                    eq, 1)
         # fttttl = 0.5*(fppl - 4.0*fpl + 6.0*fl - 4.0*fml + fmml)
         multiply_add_to_node_vars!(fttttl,  0.5, 1.0, fppl, -4.0 , fpl,
                                                  6.0, fl,
                                                 -4.0, fml ,  1.0 , fmml,
                                    eq, 1)
         # fttttr = 0.5*(fppr - 4.0*fpr + 6.0*fr - 4.0*fmr + fmmr)
         multiply_add_to_node_vars!(fttttr,  0.5, 1.0, fppr, -4.0 , fpr,
                                                  6.0, fr,
                                                 -4.0, fmr ,  1.0 , fmmr,
                                    eq, 1)

         ftl_node  = get_node_vars(ftl,  eq, 1)
         ftr_node  = get_node_vars(ftr,  eq, 1)
         fttl_node  = get_node_vars(fttl,  eq, 1)
         fttr_node  = get_node_vars(fttr,  eq, 1)
         ftttl_node  = get_node_vars(ftttl,  eq, 1)
         ftttr_node  = get_node_vars(ftttr,  eq, 1)
         fttttl_node  = get_node_vars(fttttl,  eq, 1)
         fttttr_node  = get_node_vars(fttttr,  eq, 1)

         # F = f + 0.5 * ft + (1.0/6.0) * ftt + (1.0/24.0) * fttt
         multiply_add_to_node_vars!(Fb, # F += 1.0/24.0 * fttt_node
                                    0.5, ftl_node,
                                    1.0/6.0, fttl_node,
                                    1.0/24.0, ftttl_node,
                                    1.0/120.0, fttttl_node,
                                    eq, 1, cell)
         multiply_add_to_node_vars!(Fb, # F += 1.0/24.0 * fttt_node
                                    0.5, ftr_node,
                                    1.0/6.0, fttr_node,
                                    1.0/24.0, ftttr_node,
                                    1.0/120.0, fttttr_node,
                                    eq, 2, cell)
      end
   end
   return nothing
end


end # @muladd

end # module