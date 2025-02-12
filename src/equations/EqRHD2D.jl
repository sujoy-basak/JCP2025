module EqRHD2D

( # Methods to be extended in this module
  # Extended methods are also marked with SSFR. Example - SSFR.flux
import SSFR: flux, prim2con, prim2con!, con2prim, con2prim!,
             eigmatrix,
             apply_tvb_limiter!, apply_bound_limiter!, apply_bound_limiter_extreme!, initialize_plot,
             blending_flux_factors, zhang_shu_flux_fix,
             write_soln!, compute_time_step, post_process_soln,
             update_ghost_values_lwfr!
)

( # Methods explicitly imported for readability
using SSFR: get_filename, minmod, @threaded,
            periodic, dirichlet, neumann, reflect, dirichlet_neumann, hllc_bc,
            update_ghost_values_fn_blend!,
            get_node_vars,
            set_node_vars!,
            nvariables, eachvariable,
            add_to_node_vars!, subtract_from_node_vars!,
            multiply_add_to_node_vars!, multiply_add_set_node_vars!,
            comp_wise_mutiply_node_vars!, AbstractEquations
)

using SSFR.CartesianGrids: CartesianGrid2D, save_mesh_file


using Plots
using Polyester
using StaticArrays
using LoopVectorization
using TimerOutputs
using UnPack
using WriteVTK
using LinearAlgebra
using MuladdMacro
using Printf
using EllipsisNotation
using HDF5: h5open, attributes
using SSFR

using SSFR.FR2D: correct_variable!, correct_variable_extreme!
using SSFR.FR: symmetric_sum

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct RHD2D <: AbstractEquations{2,4}
   γ::Float64
   nvar::Int64
   name::String
   initial_values::Dict{String, Function}
   numfluxes::Dict{String, Function}
end

#------------------------------------------------------------------------------




# Extending the flux function
@inline @inbounds function SSFR.flux(x, y, U, eq::RHD2D, orientation::Integer)
    D, mx, my, E = U

    vx = get_velocity_x(eq,U)
    vy = get_velocity_y(eq,U)
    p  = get_pressure(eq,U)

   if orientation == 1 ##SEE
      F1 = D * vx
      F2 = mx * vx + p
      F3 = my * vx
      F4 = mx
      return SVector(F1,F2,F3,F4)
   else
      G1 = D * vy
      G2 = mx * vy
      G3 = my * vy + p
      G4 = my
      return SVector(G1,G2,G3,G4)
   end
end

# Extending the flux function
@inline @inbounds function SSFR.flux(x, y, U, eq::RHD2D)
    D, mx, my, E = U

    vx = get_velocity_x(eq,U)
    vy = get_velocity_y(eq,U)
    p  = get_pressure(eq,U)

    F1 = D * vx
    F2 = mx * vx + p
    F3 = my * vx
    F4 = mx
    F = SVector(F1,F2,F3,F4)

    G1 = D * vy
    G2 = mx * vy
    G3 = my * vy + p
    G4 = my
    G = SVector(G1,G2,G3,G4)

   return F, G
end

function get_con1(eq::RHD2D, prim)
    @unpack γ = eq
    v = sqrt(prim[2]^2 + prim[3]^2)
    U1 = prim[1]/sqrt(1-v^2)
    return U1
end

function get_con2(eq::RHD2D, prim)
    @unpack γ = eq
    v = sqrt(prim[2]^2 + prim[3]^2)
    hh = 1+(prim[4]*γ)/(prim[1]*(γ-1))
    U2 = (prim[1]/(1-v^2))*prim[2]*hh
    return U2
end

function get_con3(eq::RHD2D, prim)
   @unpack γ = eq
   v = sqrt(prim[2]^2 + prim[3]^2)
   hh = 1+(prim[4]*γ)/(prim[1]*(γ-1))
   U3 = (prim[1]/(1-v^2))*prim[3]*hh
   return U3
end

function get_con4(eq::RHD2D, prim)
    @unpack γ = eq
    v = sqrt(prim[2]^2 + prim[3]^2)
    hh = 1+(prim[4]*γ)/(prim[1]*(γ-1))
    U3 = (prim[1]/(1-v^2))*hh-prim[4]
    return U3
end

# function converting primitive variables to PDE variables
function SSFR.prim2con(eq::RHD2D, prim)
    U1 = get_con1(eq, prim)
    U2 = get_con2(eq, prim)
    U3 = get_con3(eq, prim)
    U4 = get_con4(eq, prim)
    U = SVector(U1, U2, U3, U4)
    return U
end

function SSFR.prim2con!(eq::RHD2D, ua)
    U1 = get_con1(eq, prim)
    U2 = get_con2(eq, prim)
    U3 = get_con3(eq, prim)
    U4 = get_con4(eq, prim)
    ua[1] = U1
    ua[2] = U2
    ua[3] = U3
    ua[4] = U4
   return nothing
end

function check_admissibility(u::AbstractArray, eq::RHD2D)
   D = u[1]
   q = u[4]-sqrt(u[1]^2+u[2]^2+u[3]^2)

   if D<0 || q<0
      @show D, q
   end
   return nothing
end
# function converting pde variables to primitive variables  #Source: https://link.springer.com/article/10.1007/s00033-020-1250-8

function newtonraphson(f, fd, u0, tol, ulb, uub)
   up=u0
   un=up+1
   i=0
   while (un-up)>tol || i<20
      un=up-f(up)/fd(up)
      up=un
      i = i+1
   end
   return un
end

 @inline function get_ab_velocity(eq::RHD2D, u::AbstractArray)
   @unpack γ = eq

   Md=sqrt(u[2]^2 + u[3]^2)
   if Md < 1e-15
      return 0
   else
      Dd=u[1]
      Ed = u[4]
      deno=((γ-1)^2)*(Md^2+Dd^2)

      if deno< 1e-15
         deno +=1e-20
      end

      a3=-(2*γ*(γ-1)*Md*Ed)/deno
      a2=(γ^2*Ed^2+2*(γ-1)*Md^2-(γ-1)^2*Dd^2)/deno
      a1=-(2*γ*Md*Ed)/deno
      a0=Md^2/deno

      # Using NR
      # f(x) = a0 + a1*x + a2*x^2 + a3*x^3 + x^4
      # fd(x) = a1 + 2*a2*x + 3*a3*x^2 + 4*x^3

      # sss=γ^2*u[4]^2-4*(γ-1)*Md^2
      # if sss<0
      #    #@show sss
      #    ulb = 0
      # else
      #    ulb= (1/(2*Md*(γ-1))) * (γ*u[4]-sqrt(abs(sss)))
      # end

      # uub= min(1,Md/(u[4]+1e-20))
      # if ulb > 1e-15
      #    z = (1/2) * (1-u[1]/(u[4]+1e-20))*(ulb-uub)
      # else
      #    z = 0
      # end
         
      # u_in = (1/2)*(ulb+uub) + z

      # if uub < 1.0e-15
      #    ab_v = 0.0
      # else
      #    ab_v = newtonraphson(f, fd, u_in, 1e-15, ulb, uub)
      # end
      # if ab_v <1.0e-15
      #    ab_v=0.0
      # #elseif ab_v > uub #|| ab_v<ulb
      #   # @show ulb, ab_v, uub
      # end

      #Using analytic solution
      b2=-a2
      b1=a1*a3-4*a0
      b0=-(a1^2+a0*a3^2-4*a0*a2)
      r=(b1*b2-3*b0)/6-(b2^3)/27
      q=b1/3-b2^2/9
      s2=cbrt(r-sqrt(abs(q^3+r^2)))
      s1=cbrt(r+sqrt(abs(q^3+r^2)))
      z1=(s1+s2)-b2/3
      Cd=z1/2-sqrt(abs((z1/2)^2-a0))
      Bd=a3/2+sqrt(abs((a3^2)/4+z1-a2))
      ab_v=(-Bd+sqrt(abs(Bd^2-4*Cd)))/2

       return ab_v
   end
end

 @inline function get_velocity_x(eq::RHD2D, u::AbstractArray)
    ab_v = get_ab_velocity(eq,u)
    Md=sqrt(u[2]^2 + u[3]^2)
    if Md <1e-15
      if ab_v>0.0001
         println("in get_velocity_x ab_v = $ab_v")
      end
       vx = 0.0
    else
       vx = (u[2]/Md)*ab_v
    end
    return vx
  end

  @inline function get_velocity_y(eq::RHD2D, u::AbstractArray)
    ab_v = get_ab_velocity(eq,u)
    Md=sqrt(u[2]^2 + u[3]^2)
    if Md <1e-15
      if ab_v>0.0001
         println("in get_velocity_y ab_v = $ab_v")
      end
       vy = 0.0
    else
       vy = (u[3]/Md)*ab_v
    end
    return vy
  end

  @inline function get_density(eq::RHD2D, u::AbstractArray)
    v=get_ab_velocity(eq,u)
    ρ = u[1]*sqrt(abs(1-v^2))
    return ρ
  end
 
 @inline function get_pressure(eq::RHD2D, u::AbstractArray)
    @unpack γ = eq
    vx=get_velocity_x(eq,u)
    vy=get_velocity_y(eq,u)
    ρ = get_density(eq, u)
    p = (γ-1.0)*(u[4]-(u[2]*vx+u[3]*vy)-ρ)
    return p
 end


function SSFR.con2prim(eq::RHD2D, U)
    ρd=get_density(eq,U)
    vx=get_velocity_x(eq,U)
    vy=get_velocity_y(eq,U)
    pd=get_pressure(eq,U)
    primitives = SVector(ρd, vx, vy, pd)
    return primitives
end

function SSFR.con2prim!(eq::RHD2D, ua, ua_)
    ρd=get_density(eq,ua)
    vx=get_velocity_x(eq,ua)
    vy=get_velocity_y(eq,ua)
    pd=get_pressure(eq,ua)
    ua_[1], ua_[2], ua_[3], ua_[4] = ( ρd, vx, vy, pd )
   return nothing
end

function SSFR.con2prim!(eq::RHD2D, ua)
    ρd=get_density(eq,ua)
    vx=get_velocity_x(eq,ua)
    vy=get_velocity_y(eq,ua)
    pd=get_pressure(eq,ua)
    ua[1]=ρd
    ua[2]=vx
    ua[3]=vy
    ua[4]=pd
   return nothing
end


@inline function get_rdensity(eq::RHD2D, u::AbstractArray)
   return u[1]
 end

@inline function get_q(eq::RHD2D, u::AbstractArray)
   q = u[4] - sqrt(u[1]^2 + (u[2]^2 + u[3]^2))
   return q
 end

function SSFR.is_admissible(eq::RHD2D, u::AbstractVector)
   md = sqrt(u[2]^2 + u[3]^2)
   q=u[4]-sqrt(abs(u[1]^2+md^2))
   if u[1] > 1e-15 && q > 1e-15 ##SEE
     return true
   else
     return false
   end
end

#-------------------------------------------------------------------------------
# Scheme information
#-------------------------------------------------------------------------------

function max_abs_eigenvalue_x(eq::RHD2D, ρ::Float64, vx::Float64, vy::Float64, p::Float64)
    @unpack γ = eq
    v=sqrt(vx^2 + vy^2)
    hd=1+(γ*p)/((γ-1)*ρ)
    if p<=0 || ρ<=0
      println("p = $p and rho = $ρ")
    end
    if hd<=0
      println("hd = $hd")
    end

    if abs(p)<1e-6
      p=abs(p)
    end

    c=sqrt(γ*p/(hd*ρ))
    Qx= 1-vx^2-c^2*vy^2

    eig1d=((1-c^2)*vx - c*sqrt(1-v^2)*sqrt(Qx))/(1-c^2 * v^2) ##SEE
    eig1=abs(eig1d)
    eig2=abs(vx)
    eig3=abs(vx)
    eig4d=((1-c^2)*vx + c*sqrt(1-v^2)*sqrt(Qx))/(1-c^2 * v^2) ##SEE
    eig4=abs(eig4d)
    eig=max(eig1,eig2,eig3,eig4)
    return eig  
 end

 function eigenvalue_x(eq::RHD2D, ρ::Float64, vx::Float64, vy::Float64, p::Float64)
   @unpack γ = eq
   v=sqrt(vx^2 + vy^2)
   hd=1+(γ*p)/((γ-1)*ρ)
   if p<=0 || ρ<=0
     println("p = $p and rho = $ρ")
   end
   if hd<=0
     println("hd = $hd")
   end

   if abs(p)<1e-6
     p=abs(p)
   end

   c=sqrt(γ*p/(hd*ρ))
   Qx= 1-vx^2-c^2*vy^2

   eig1d=((1-c^2)*vx - c*sqrt(1-v^2)*sqrt(Qx))/(1-c^2 * v^2)
   eig1=eig1d
   eig2=vx
   eig3=vx
   eig4d=((1-c^2)*vx + c*sqrt(1-v^2)*sqrt(Qx))/(1-c^2 * v^2)
   eig4=eig4d
   eig= SVector(eig1,eig2,eig3,eig4)
   return eig 
end

 function max_abs_eigenvalue_y(eq::RHD2D, ρ::Float64, vx::Float64, vy::Float64, p::Float64)
    @unpack γ = eq
    v=sqrt(vx^2 + vy^2)
    hd=1+(γ*p)/((γ-1)*ρ)
    if p<0 || ρ<0
      println("p = $p and rho = $ρ")
    end
    if hd<0
      println("hd = $hd")
    end

    if abs(p)<1e-6
      p=abs(p)
    end

    c=sqrt(γ*p/(hd*ρ))
    Qy= 1-vy^2-c^2*vx^2
    
    eig1d=((1-c^2)*vy - c*sqrt(1-v^2)*sqrt(Qy))/(1-c^2 * v^2)
    eig1=abs(eig1d)
    eig2=abs(vy)
    eig3=abs(vy)
    eig4d=((1-c^2)*vy + c*sqrt(1-v^2)*sqrt(Qy))/(1-c^2 * v^2)
    eig4=abs(eig4d)
    eig=max(eig1,eig2,eig3,eig4)
    return eig  
 end

 function eigenvalue_y(eq::RHD2D, ρ::Float64, vx::Float64, vy::Float64, p::Float64)
   @unpack γ = eq
   v=sqrt(vx^2 + vy^2)
   hd=1+(γ*p)/((γ-1)*ρ)
   if p<0 || ρ<0
     println("p = $p and rho = $ρ")
   end
   if hd<0
     println("hd = $hd")
   end

   if abs(p)<1e-6
     p=abs(p)
   end

   c=sqrt(γ*p/(hd*ρ))
   Qy= 1-vy^2-c^2*vx^2
   
   eig1d=((1-c^2)*vy - c*sqrt(1-v^2)*sqrt(Qy))/(1-c^2 * v^2)
   eig1=eig1d
   eig2=vy
   eig3=vy
   eig4d=((1-c^2)*vy + c*sqrt(1-v^2)*sqrt(Qy))/(1-c^2 * v^2)
   eig4=eig4d
   eig = SVector(eig1,eig2,eig3,eig4)
   return eig  
end


function SSFR.compute_time_step(eq::RHD2D, grid, aux, op, cfl, u1, ua)
   @timeit aux.timer "Time Step computation" begin
   @unpack dx, dy = grid
   nx, ny = grid.size
   @unpack γ = eq
   @unpack wg = op
   den    = 0.0
   corners = ( (0,0), (nx+1,0), (0,ny+1), (nx+1,ny+1) )
   for element in CartesianIndices((0:nx+1, 0:ny+1))
      el_x, el_y = element[1], element[2]
      if (el_x,el_y) ∈ corners # KLUDGE - Temporary hack
         continue
      end
      u_node = get_node_vars(ua, eq, el_x, el_y)
      rho, vx, vy, p = con2prim(eq, u_node)

      sx = max_abs_eigenvalue_x(eq,rho,vx,vy,p)
      sy = max_abs_eigenvalue_y(eq,rho,vx,vy,p)
      den = max(den, sx/dx[el_x] + sy/dy[el_y])

   end
   if den < 1e-15
      den += 1e-20
   end

   dt  = cfl / den
   return dt
   end # timer
end


function RHD_isentropic_vortex(x,y)
   γ = 5/3
   w = 0.5*sqrt(2)
   epsc = 5.0
   c1 = ((γ-1)*epsc^2)/(γ*8*pi^2)
   lf = 1/(sqrt(1-w^2))
   xs = x + (lf - 1)*(x+y)/2
   ys = y + (lf - 1)*(x+y)/2
   r = sqrt(xs^2+ys^2)
   c2 = (2*γ*c1*exp(1-r^2))/(2*γ-1-γ*c1*exp(1-r^2))
   f = sqrt(c2/(1+c2*r^2))

   rho = (1-c1*exp(1-r^2))^(1/(γ-1))
   vxs  = -ys*f
   vys  = xs*f
   p   = rho^γ
   vx = (1/(1-w*(vxs+vys)/sqrt(2)))*(vxs/lf - w/sqrt(2) + lf*w^2*(vxs+vys)/(2*(lf+1)))
   vy = (1/(1-w*(vxs+vys)/sqrt(2)))*(vys/lf - w/sqrt(2) + lf*w^2*(vxs+vys)/(2*(lf+1)))
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

#exact_RHD_isentropic_vortex(x,y,t) = RHD_isentropic_vortex(x-0.5*sqrt(2)*t,y-0.5*sqrt(2)*t)
exact_RHD_isentropic_vortex(x,y,t) = RHD_isentropic_vortex(x,y) #Assuming final time is 20

function RHD_riemann1(x,y)
   γ = 5/3
   if x < 0.5 && y < 0.5
      rho = 0.5
      vx  = 0.0
      vy  = 0.0
      p   = 1.0
   elseif x < 0.5 && y > 0.5
      rho = 0.1
      vx  = 0.99
      vy  = 0.0
      p   = 1.0
   elseif x > 0.5 && y < 0.5
      rho = 0.1
      vx  = 0.0
      vy  = 0.99
      p   = 1.0
   else
      rho = 0.1
      vx  = 0.0
      vy  = 0.0
      p   = 0.01
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_riemann1(x,y, t)= RHD_riemann1(x,y)

function RHD_riemann2(x,y)
   γ = 5/3
   if x >= 0.5 && y >= 0.5
      rho = 0.5
      vx  = 0.5
      vy  = -0.5
      p   = 5.0
   elseif x < 0.5 && y > 0.5
      rho = 1.0
      vx  = 0.5
      vy  = 0.5
      p   = 5.0
   elseif x < 0.5 && y < 0.5
      rho = 3.0
      vx  = -0.5
      vy  = 0.5
      p   = 5.0
   else
      rho = 1.5
      vx  = -0.5
      vy  = -0.5
      p   = 5.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_riemann2(x,y, t)= RHD_riemann2(x,y)

function RHD_riemann3(x,y)
   γ = 5/3
   if x >= 0.5 && y >= 0.5
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 1.0
   elseif x < 0.5 && y > 0.5
      rho = 0.5771
      vx  = -0.3529
      vy  = 0.0
      p   = 0.4
   elseif x < 0.5 && y < 0.5
      rho = 1.0
      vx  = -0.3529
      vy  = -0.3529
      p   = 1.0
   else #if x > 0.5 && y < 0.5
      rho = 0.5771
      vx  = 0.0
      vy  = -0.3529
      p   = 0.4
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_riemann3(x,y, t)= RHD_riemann3(x,y)

function RHD_riemann4(x,y)
   γ = 5/3
   if x >= 0.5 && y >= 0.5
      rho = 0.035145216124503
      vx  = 0.0
      vy  = 0.0
      p   = 0.162931056509027
   elseif x < 0.5 && y > 0.5
      rho = 0.1
      vx  = 0.7
      vy  = 0.0
      p   = 1.0
   elseif x < 0.5 && y < 0.5
      rho = 0.5
      vx  = 0.0
      vy  = 0.0
      p   = 1.0
   else #if x > 0.5 && y < 0.5
      rho = 0.1
      vx  = 0.0
      vy  = 0.7
      p   = 1.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_riemann4(x,y, t)= RHD_riemann4(x,y)

function RHD_riemann5(x,y)
   γ = 5/3
   if x < 0.5 && y < 0.5
      rho = 0.01
      vx  = 0.0
      vy  = 0.0
      p   = 0.05
   elseif x < 0.5 && y > 0.5
      rho = 0.00414329639576
      vx  = 0.9946418833556542
      vy  = 0.0
      p   = 0.05
   elseif x > 0.5 && y < 0.5
      rho = 0.00414329639576
      vx  = 0.0
      vy  = 0.9946418833556542
      p   = 0.05
   else #x >= 0.5 && y >= 0.5
      rho = 0.1
      vx  = 0.0
      vy  = 0.0
      p   = 20.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_riemann5(x,y, t)= RHD_riemann5(x,y)


function RHD_newriemann1(x,y)
   γ = 5/3
   if (x-215)^2 + y^2 < 25^2
      rho = 0.1358
      vx  = 0.0
      vy  = 0.0
      p   = 0.05
   elseif x < 265
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
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_newriemann1(x,y, t)= RHD_newriemann1(x,y)

function RHD_newriemann2(x,y)
   γ = 5/3
   if (x-215)^2 + y^2 < 25^2
      rho = 3.1538
      vx  = 0.0
      vy  = 0.0
      p   = 0.05
   elseif x < 265
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
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_newriemann2(x,y, t)= RHD_newriemann2(x,y)

function RHD_reljet1(x,y)
   γ = 5/3
   rho = 1.0
   vx  = 0.0
   vy  = 0.0
   p   = 2.353624072171881e-5
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_reljet1(x,y, t)= RHD_reljet1(x,y)

function reljet1_bv(x,y,t)
   γ = 5/3
   if abs(x) <= 0.5 && y == 0 
      rho = 0.1
      vx  = 0.0
      vy  = 0.99
      p   = 2.353624072171881e-5
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 2.353624072171881e-5
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
 end

function RHD_reljet2(x,y)
   γ = 5/3
   rho = 1.0
   vx  = 0.0
   vy  = 0.0
   p   = 2.3966375079777598e-5
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_reljet2(x,y, t)= RHD_reljet2(x,y)

function reljet2_bv(x,y,t)
   γ = 5/3
   if abs(x) <= 0.5 && y == 0
      rho = 0.1
      vx  = 0.0
      vy  = 0.999
      p   = 2.3966375079777598e-5
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 2.3966375079777598e-5
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
 end

function RHD_reljet3(x,y)
   γ = 5/3
   rho = 1.0
   vx  = 0.0
   vy  = 0.0
   p   = 2.399534418327213e-7
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD_reljet3(x,y, t)= RHD_reljet3(x,y)

function reljet3_bv(x,y,t)
   γ = 5/3
   if abs(x) <= 0.5+1e-15 && y == 0
      rho = 0.1
      vx  = 0.0
      vy  = 0.9999
      p   = 2.399534418327213e-7
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 2.399534418327213e-7
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
 end

 function RHD_kelvin_helmholtz_instability(x,y)
   γ = 4/3
   vs = 0.5
   a = 0.01
   η0 = 0.1
   σ = 0.1
   ρ0 = 0.505
   ρ1 = 0.495
   if y > 0
      vx = vs*tanh((y-0.5)/a)
      vy = η0*vs*sin(2*pi*x)*exp(-(y-0.5)^2/σ)
      rho = ρ0 + ρ1*tanh((y-0.5)/a)
   else
      vx = -vs*tanh((y+0.5)/a)
      vy = -η0*vs*sin(2*pi*x)*exp(-(y+0.5)^2/σ)
      rho = ρ0 - ρ1*tanh((y+0.5)/a)
   end
   p = 1.0
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
 end

 exact_RHD_kelvin_helmholtz_instability(x,y,t) = RHD_kelvin_helmholtz_instability(x,y)

function RHD_richtmyer_meshkov_instability(x,y)
   γ = 5/3
   rhol = 1.0
   rhom = 0.5
   rhor = 50
   vxl = 0.9
   vxm = 0.0
   vxr = 0.0
   pl = 0.01
   pm = 0.01
   pr = 0.01

   vy = 0.0

   x0 = 3
   a = 0.25
   λ = 2.5
   if x < 1
      rho = rhol
      vx = vxl
      p = pl
   elseif x < x0 + a*sin(pi/2 + 2*pi*y/λ)
      rho = rhom
      vx = vxm
      p = pm
   else
      rho = rhor
      vx = vxr
      p = pr
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
  end
  exact_RHD_richtmyer_meshkov_instability(x,y,t) = RHD_richtmyer_meshkov_instability(x,y)


######################################## 1D test cases  y dir ####################
function RHD1D_y_riemann1(x,y)
   γ = 5/3
   if y < 0.5
      rho = 1.0
      vy  = -0.6
      vx  = 0.0
      p   = 10.0
   else
      rho = 10.0
      vy  = 0.5
      vx  = 0.0
      p   = 20.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_y_riemann1(x,y, t)= RHD1D_y_riemann1(x,y)

function RHD1D_y_riemann2(x,y)
   γ = 5/3
   if y < 0.5
      rho = 1.0
      vy  = 0.0
      vx  = 0.0
      p   = (10.0)^3
   else
      rho = 1.0
      vy  = 0.0
      vx  = 0.0
      p   = (10.0)^(-2)
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_y_riemann2(x,y, t)= RHD1D_y_riemann2(x,y)


function RHD1D_y_riemann3(x,y)
   γ = 5/3
   if y < 0.5
      rho = 10.0
      vy  = 0.0
      vx  = 0.0
      p   = 40/3
   else
      rho = 1.0
      vy  = 0.0
      vx  = 0.0
      p   = (10.0)^(-6)
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_y_riemann3(x,y, t)= RHD1D_y_riemann3(x,y)

function RHD1D_y_density_pert(x,y)
   γ = 5/3
   if y < 0.5
      rho = 5.0
      vy  = 0.0
      vx  = 0.0
      p   = 50.0
   else
      rho = 2.0 + 0.3 * sin(50 * y)
      vy  = 0.0
      vx  = 0.0
      p   = 5.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_y_density_pert(x,y,t)= RHD1D_y_density_pert(x,y)

function RHD1D_y_blast(x,y)
   γ = 1.4
   if y < 0.1
      rho = 1.0
      vy  = 0.0
      vx  = 0.0
      p   = 1000.0
   elseif y >= 0.1 && y < 0.9
      rho = 1.0
      vy  = 0.0
      vx  = 0.0
      p   = 0.01
   else
      rho = 1.0
      vy  = 0.0
      vx  = 0.0
      p   = 100.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_y_blast(x,y,t) = RHD1D_y_blast(x,y)
##############################################################################################

######################################## 1D test cases  x dir ####################
function RHD1D_x_riemann1(x,y)
   γ = 5/3
   if x < 0.5
      rho = 1.0
      vx  = -0.6
      vy  = 0.0
      p   = 10.0
   else
      rho = 10.0
      vx  = 0.5
      vy  = 0.0
      p   = 20.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_x_riemann1(x,y, t)= RHD1D_x_riemann1(x,y)

function RHD1D_x_riemann2(x,y)
   γ = 5/3
   if x < 0.5
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = (10.0)^3
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = (10.0)^(-2)
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_x_riemann2(x,y, t)= RHD1D_x_riemann2(x,y)


function RHD1D_x_riemann3(x,y)
   γ = 5/3
   if x < 0.5
      rho = 10.0
      vx  = 0.0
      vy  = 0.0
      p   = 40/3
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = (10.0)^(-6)
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_x_riemann3(x,y, t)= RHD1D_x_riemann3(x,y)

function RHD1D_x_density_pert(x,y)
   γ = 5/3
   if x < 0.5
      rho = 5.0
      vx  = 0.0
      vy  = 0.0
      p   = 50.0
   else
      rho = 2.0 + 0.3 * sin(50 * x)
      vx  = 0.0
      vy  = 0.0
      p   = 5.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_x_density_pert(x,y,t)= RHD1D_x_density_pert(x,y)

function RHD1D_x_blast(x,y)
   γ = 1.4
   if x < 0.1
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 1000.0
   elseif x >= 0.1 && x < 0.9
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 0.01
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 100.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_x_blast(x,y,t) = RHD1D_x_blast(x,y)

########################################## 1D test cases xy dir ################################
#=
function RHD1D_xy_riemann1(x,y)
   γ = 5/3
   if x < 0.5 && y < 0.5 #THINK
      rho = 1.0
      vx  = -0.6
      vy  = -0.6
      p   = 10.0
   else
      rho = 10.0
      vx  = 0.5
      vy  = 0.5
      p   = 20.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_xy_riemann1(x,y, t)= RHD1D_xy_riemann1(x,y)


function RHD1D_xy_riemann2(x,y)
   γ = 5/3
   if x < 0.5 && y < 0.5 #THINK
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = (10.0)^3
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = (10.0)^(-2)
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_xy_riemann2(x,y, t)= RHD1D_xy_riemann2(x,y)

function RHD1D_xy_riemann3(x,y)
   γ = 5/3
   if x < 0.5 && y < 0.5 #THINK
      rho = 10.0
      vx  = 0.0
      vy  = 0.0
      p   = 40/3
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = (10.0)^(-6)
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_xy_riemann3(x,y, t)= RHD1D_xy_riemann3(x,y)


function RHD1D_xy_density_pert(x,y)
   γ = 5/3
   if x < 0.5 && y < 0.5
      rho = 5.0
      vy  = 0.0
      vx  = 0.0
      p   = 50.0
   else
      rho = 2.0 + 0.3 * sin(50 * sqrt(x^2+y^2))
      vy  = 0.0
      vx  = 0.0
      p   = 5.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_xy_density_pert(x,y, t)= RHD1D_xy_density_pert(x,y)


function RHD1D_xy_blast(x,y)
   γ = 1.4
   if x < 0.1 && y < 0.1
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 1000.0
   elseif x < 0.9 && y < 0.9
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 0.01
   else
      rho = 1.0
      vx  = 0.0
      vy  = 0.0
      p   = 100.0
   end
   prim = [rho, vx, vy, p]
   equation = get_equation(γ)
   U = prim2con(equation, prim)
   return U
end

exact_RHD1D_xy_blast(x,y, t)= RHD1D_xy_blast(x,y)
=#

##############################################################################################

 initial_values = Dict{String, Function}()

 initial_values["RHD_isentropic_vortex"] = RHD_isentropic_vortex
 initial_values["RHD_riemann1"] = RHD_riemann1
 initial_values["RHD_riemann2"] = RHD_riemann2
 initial_values["RHD_riemann3"] = RHD_riemann3
 initial_values["RHD_riemann4"] = RHD_riemann4
 initial_values["RHD_riemann5"] = RHD_riemann5

#-------------------------------------------------------------------------------
# Numerical Fluxes
#-------------------------------------------------------------------------------
function rusanov(x, ual, uar, Fl, Fr, Ul, Ur, eq::RHD2D, dir)
    rho_ll,vx_ll,vy_ll,p_ll=con2prim(eq,ual)
    rho_rr,vx_rr,vy_rr,p_rr=con2prim(eq,uar)
    if dir == 1
        λ = max(max_abs_eigenvalue_x(eq,rho_ll,vx_ll,vy_ll,p_ll),max_abs_eigenvalue_x(eq,rho_rr,vx_rr,vy_rr,p_rr))
    else
        λ = max(max_abs_eigenvalue_y(eq,rho_ll,vx_ll,vy_ll,p_ll),max_abs_eigenvalue_y(eq,rho_rr,vx_rr,vy_rr,p_rr))
    end
    F1  = 0.5*(Fl[1]+Fr[1]) - 0.5*λ*(Ur[1] - Ul[1])
    F2  = 0.5*(Fl[2]+Fr[2]) - 0.5*λ*(Ur[2] - Ul[2])
    F3  = 0.5*(Fl[3]+Fr[3]) - 0.5*λ*(Ur[3] - Ul[3])
    F4  = 0.5*(Fl[4]+Fr[4]) - 0.5*λ*(Ur[4] - Ul[4])
    return SVector(F1,F2,F3,F4)
end

function hll_speeds(ual, uar, dir, eq) #
   ρ_ll, v1_ll, v2_ll, p_ll = con2prim(eq, ual)
   ρ_rr, v1_rr, v2_rr, p_rr = con2prim(eq, uar)

   if dir == 1
      eig1_ll, eig2_ll, eig3_ll, eig4_ll = eigenvalue_x(eq, ρ_ll, v1_ll, v2_ll, p_ll)
      eig1_rr, eig2_rr, eig3_rr, eig4_rr = eigenvalue_x(eq, ρ_rr, v1_rr, v2_rr, p_rr)
   elseif dir == 2
      eig1_ll, eig2_ll, eig3_ll, eig4_ll = eigenvalue_y(eq, ρ_ll, v1_ll, v2_ll, p_ll)
      eig1_rr, eig2_rr, eig3_rr, eig4_rr = eigenvalue_y(eq, ρ_rr, v1_rr, v2_rr, p_rr)
   end

   sl = min(min(eig1_ll, eig2_ll, eig3_ll, eig4_ll), min(eig1_rr, eig2_rr, eig3_rr, eig4_rr))
   sr = max(max(eig1_ll, eig2_ll, eig3_ll, eig4_ll), max(eig1_rr, eig2_rr, eig3_rr, eig4_rr))
   return sl, sr
end

function hll(x, ual, uar, Fl, Fr, Ul, Ur, eq::RHD2D, dir::Integer)
   sl, sr = hll_speeds(ual, uar, dir, eq)

   if sl >= 0
      f1 = Fl[1]
      f2 = Fl[2]
      f3 = Fl[3]
      f4 = Fl[4]
   elseif sr <= 0
      f1 = Fr[1]
      f2 = Fr[2]
      f3 = Fr[3]
      f4 = Fr[4]
   else
      f1 = (sr*Fl[1] - sl*Fr[1] +sr*sl(Ur[1] - Ul[1]))/(sr-sl)
      f1 = (sr*Fl[2] - sl*Fr[2] +sr*sl(Ur[2] - Ul[2]))/(sr-sl)
      f1 = (sr*Fl[3] - sl*Fr[3] +sr*sl(Ur[3] - Ul[3]))/(sr-sl)
      f1 = (sr*Fl[4] - sl*Fr[4] +sr*sl(Ur[4] - Ul[4]))/(sr-sl)
   end
   return SVector(f1, f2, f3, f4)
end

function hllc(x, ual, uar, Fl, Fr, Ul, Ur, eq::RHD2D, dir::Integer)
  λ_l, λ_r = hll_speeds(ual, uar, dir, eq)

   if λ_l >= 0
      f1 = Fl[1]
      f2 = Fl[2]
      f3 = Fl[3]
      f4 = Fl[4]
   elseif λ_r <= 0
      f1 = Fr[1]
      f2 = Fr[2]
      f3 = Fr[3]
      f4 = Fr[4]
   else
      D_l, mx_l, my_l, E_l = Ul
      D_r, mx_r, my_r, E_r = Ur

      vx_l = get_velocity_x(eq, Ul)
      vy_l = get_velocity_y(eq, Ul)
      p_l = get_pressure(eq, Ul)
      vx_r = get_velocity_x(eq, Ur)
      vy_r = get_velocity_y(eq, Ur)
      p_r = get_pressure(eq, Ur)

      f1_hll = (λ_r*Fl[1] - λ_l*Fr[1] +λ_r*λ_l*(D_r - D_l))/(λ_r-λ_l)
      f2_hll = (λ_r*Fl[2] - λ_l*Fr[2] +λ_r*λ_l*(mx_r - mx_l))/(λ_r-λ_l)
      f3_hll = (λ_r*Fl[3] - λ_l*Fr[3] +λ_r*λ_l*(my_r - my_l))/(λ_r-λ_l)
      f4_hll = (λ_r*Fl[4] - λ_l*Fr[4] +λ_r*λ_l*(E_r - E_l))/(λ_r-λ_l)

      D_hll = (λ_r*D_r - λ_l*D_l + Fl[1] - Fr[1])/(λ_r - λ_l)
      mx_hll = (λ_r*mx_r - λ_l*mx_l + Fl[2] - Fr[2])/(λ_r - λ_l)
      my_hll = (λ_r*my_r - λ_l*my_l + Fl[3] - Fr[3])/(λ_r - λ_l)
      E_hll = (λ_r*E_r - λ_l*E_l + Fl[4] - Fr[4])/(λ_r - λ_l)

      if dir == 1
         λ_star = ((E_hll + f2_hll) - sqrt((E_hll + f2_hll)^2 - 4*f4_hll*mx_hll))/(2*f4_hll)

         if abs(λ_star) >= 1
            @assert false
            @show λ_star , "lambda_star is out of bound" 
         end

         if λ_star >= 0
            A = λ_l *E_l - mx_l
            B = mx_l*(λ_l - vx_l) - p_l
            p_star = (A * λ_star - B)/(1 + λ_l * λ_star)

            D_star = D_l * (λ_l - vx_l)/(λ_l - λ_star)
            mx_star = (mx_l * (λ_l - vx_l) + p_star - p_l)/(λ_l - λ_star)
            my_star = my_l * (λ_l - vx_l)/(λ_l - λ_star)
            E_star = (E_l * (λ_l - vx_l) + p_star * λ_star - p_l * vx_l)/(λ_l - λ_star)
         else
            A = λ_r * E_r - mx_r
            B = mx_r*(λ_r - vx_r) - p_r
            p_star = (A * λ_star - B)/(1 + λ_r * λ_star)

            D_star = D_r * (λ_r - vx_r)/(λ_r - λ_star)
            mx_star = (mx_r * (λ_r - vx_r) + p_star - p_r)/(λ_r - λ_star)
            my_star = my_r * (λ_r - vx_r)/(λ_r - λ_star)
            E_star = (E_r * (λ_r - vx_r) + p_star * λ_star - p_r * vx_r)/(λ_r - λ_star)
         end
         vx_star = λ_star
         f1 = D_star * vx_star
         f2 = mx_star * vx_star + p_star
         f3 = my_star * vx_star
         f4 = mx_star
      elseif dir == 2
         λ_star = ((E_hll + f3_hll) - sqrt((E_hll + f3_hll)^2 - 4*f4_hll*my_hll))/(2*f4_hll)

         if abs(λ_star) >= 1
            @assert false
            @show λ_star , "lambda_star is out of bound"
         end

         if λ_star >= 0
            A = λ_l * E_l - my_l
            B = my_l*(λ_l - vy_l) - p_l
            p_star = (A * λ_star - B)/(1 + λ * λ_star)

            D_star = D_l * (λ_l - vy_l)/(λ_l - λ_star)
            mx_star = (mx_l * (λ_l - vy_l))/(λ_l - λ_star)
            my_star = (my_l * (λ_l - vy_l) + p_star - p_l)/(λ_l - λ_star)
            E_star = (E_l * (λ_l - vy_l) + p_star * λ_star - p_l * vy_l)/(λ_l - λ_star)
         else
            A = λ_r * E_r - my_r
            B = my_r*(λ_r - vy_r) - p_r
            p_star = (A * λ_star - B)/(1 + λ * λ_star)

            D_star = D_r * (λ_r - vy_r)/(λ_r - λ_star)
            mx_star = (mx_r * (λ_r - vy_r))/(λ_r - λ_star)
            my_star = (my_r * (λ_r - vy_r) + p_star - p_r)/(λ_r - λ_star)
            E_star = (E_r * (λ_r - vy_r) + p_star * λ_star - p_r * vy_r)/(λ_r - λ_star)
         end
         vy_star = λ_star
         f1 = D_star * vy_star
         f2 = mx_star * vy_star
         f3 = my_star * vy_star + p_star
         f4 = my_star
      end
   end
   return SVector(f1, f2, f3, f4)
end
#------------------------------------------------------------------------------
# Limiters
#------------------------------------------------------------------------------
# Zhang-Shu limiting procedure for one variable
function SSFR.apply_bound_limiter!(eq::RHD2D, grid, scheme, param, op, ua,
                                   u1, aux)
   if scheme.bound_limit == "no"
      return nothing
   end
   @unpack eps = param
   @timeit aux.timer "Bound limiter" begin
   correct_variable!(eq, get_rdensity, op, aux, grid, u1, ua, eps)
   correct_variable!(eq, get_q, op, aux, grid, u1, ua, eps)
   return nothing
   end # timer
end

function SSFR.apply_bound_limiter_extreme!(eq::RHD2D, grid, scheme, param, op, ua,
   u1, aux)
if scheme.bound_limit == "no"
return nothing
end
@unpack eps = param
@timeit aux.timer "Bound limiter" begin
correct_variable_extreme!(eq, get_rdensity, op, aux, grid, u1, ua, eps)
correct_variable_extreme!(eq, get_q, op, aux, grid, u1, ua, eps)
return nothing
end # timer
end

#------------------------------------------------------------------------------
# Blending limiter
#------------------------------------------------------------------------------
@inbounds @inline function rho_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
      ρ = get_density(eq, un[:,ix,iy])
      un[1,ix,iy] = ρ
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function pressure_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
      p = get_pressure(eq, un[:,ix,iy])
      un[1,ix,iy] = p
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function vx_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
    vx = get_velocity_x(eq,un[:,ix,iy])
    vy = get_velocity_y(eq,un[:,ix,iy])
    un[1,ix,iy]=vx
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function vy_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
    vy = get_velocity_y(eq,un[:,ix,iy])
    un[1,ix,iy]=vy
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function vx_vy_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
    vx = get_velocity_x(eq,un[:,ix,iy])
    vy = get_velocity_y(eq,un[:,ix,iy])
    un[1,ix,iy]=vx*vy
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function rho_p_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
      ρ = get_density(eq, un[:,ix,iy])
      p = get_pressure(eq, un[:,ix,iy])
      un[1,ix,iy] = ρ*p
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function rho_lorentz_p_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=1:nd_p2, ix=1:nd_p2 # loop over dofs and faces
    ρ  = get_density(eq,un[:,ix,iy])

    q=get_q(eq, un[:,ix,iy])
   if un[1,ix,iy]<0 || q<0
      @show un[1,ix,iy], q, ix, iy "indicator var"
   end

    vx = get_velocity_x(eq,un[:,ix,iy])
    vy = get_velocity_y(eq,un[:,ix,iy])
    lf = 1/sqrt(1-(vx^2+vy^2))
    p  = get_pressure(eq,un[:,ix,iy])
    un[1,ix,iy]=ρ*lf*p
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function rho_lorentz_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=2:nd_p2, ix=2:nd_p2 # loop over dofs and faces
    ρ  = get_density(eq,un[:,ix,iy])
    vx = get_velocity_x(eq,un[:,ix,iy])
    vy = get_velocity_y(eq,un[:,ix,iy])
    lf = 1/sqrt(1-(vx^2+vy^2))
    un[1,ix,iy]=ρ*lf
   end
   n_ind_var = 1
   return n_ind_var
end

@inbounds @inline function lorentz_indicator!(un, eq::RHD2D)
   nd_p2 = size(un, 2)
   for iy=2:nd_p2, ix=2:nd_p2 # loop over dofs and faces
    vx = get_velocity_x(eq,un[:,ix,iy])
    vy = get_velocity_y(eq,un[:,ix,iy])
    lf = 1/sqrt(1-(vx^2+vy^2))
    un[1,ix,iy]=lf
   end
   n_ind_var = 1
   return n_ind_var
end

function SSFR.blending_flux_factors(eq::RHD2D, ua, dx, dy)
   # This method is done differently for different equations

   # TODO - temporary hack. FIX!
   return 0.5, 0.5
end

function limit_variable_slope(eq, variable, slope, u_star_ll, u_star_rr, ue, xl, xr)
   # By Jensen's inequality, we can find theta's directly for the primitives
   var_star_ll, var_star_rr = variable(eq, u_star_ll), variable(eq, u_star_rr)
   var_low = variable(eq, ue)
   threshold = 0.1*var_low
   eps = 1e-15
   if var_star_ll < eps || var_star_rr < eps
      ratio_ll = abs(threshold - var_low) / (abs(var_star_ll - var_low) + 1e-20)
      ratio_rr = abs(threshold - var_low) / (abs(var_star_rr - var_low) + 1e-20)
      theta = min(ratio_ll, ratio_rr, 1.0)
      slope *= theta
      u_star_ll = ue + 2.0*xl*slope
      u_star_rr = ue + 2.0*xr*slope
   end
   return slope, u_star_ll, u_star_rr
end

function SSFR.limit_slope(eq::RHD2D, slope, ufl, u_star_ll, ufr, u_star_rr,
                           ue, xl, xr, el_x = nothing, el_y = nothing)

   # The MUSCL-Hancock scheme is guaranteed to be admissibility preserving if
   # slope is chosen so that
   # u_star_l = ue + 2.0*slope*xl, u_star_r = ue+2.0*slope*xr are admissible
   # ue is already admissible and we know we can find sequences of thetas
   # to make theta*u_star_l+(1-theta)*ue is admissible.
   # This is equivalent to replacing u_star_l by
   # u_star_l = ue + 2.0*theta*s*xl.
   # Thus, we simply have to update the slope by multiplying by theta.


   slope, u_star_ll, u_star_rr = limit_variable_slope(
      eq, get_rdensity, slope, u_star_ll, u_star_rr, ue, xl, xr)

   slope, u_star_ll, u_star_rr = limit_variable_slope(
      eq, get_q, slope, u_star_ll, u_star_rr, ue, xl, xr)


   ufl = ue + slope*xl
   ufr = ue + slope*xr
   

   return ufl, ufr, slope

end

function SSFR.zhang_shu_flux_fix(eq::RHD2D,
                                 uprev,    # Solution at previous time level
                                 ulow,     # low order update
                                 Fn,       # Blended flux candidate
                                 fn_inner, # Inner part of flux
                                 fn,       # low order flux
                                 c         # c is such that unew = u - c(fr-fl)
                                 )

   uhigh = uprev - c * (Fn-fn_inner) # First candidate for high order update
   D_low, D_high = get_rdensity(eq, ulow), get_rdensity(eq, uhigh)
   eps = 0.1*D_low
   ratio = abs(eps-D_low)/(abs(D_high-D_low)+1e-20)
   theta = min(ratio, 1.0)
   if theta < 1.0
      Fn = theta*Fn + (1.0-theta)*fn # Second candidate for flux
   end
   uhigh = uprev - c * (Fn - fn_inner) # Second candidate for uhigh
   q_low, q_high = get_q(eq, ulow), get_q(eq, uhigh)
   eps   = 0.1*q_low
   ratio = abs(eps-q_low)/(abs(q_high-q_low) + 1e-20)
   theta = min(ratio, 1.0)
   if theta < 1.0
      Fn = theta*Fn + (1.0-theta)*fn # Final flux
   end
   return Fn
end



#------------------------------------------------------------------------------
# Ghost values functions
#------------------------------------------------------------------------------

function SSFR.update_ghost_values_lwfr!(problem, scheme, eq::RHD2D,
                                        grid, aux, op, cache, t, dt) ##SEE
   @timeit aux.timer "Update ghost values" begin
   @unpack Fb, Ub, ua = cache
   update_ghost_values_periodic!(eq, problem, Fb, Ub)

   @unpack periodic_x, periodic_y = problem
   if periodic_x && periodic_y
      return nothing
   end

  nx, ny = grid.size
  nvar = nvariables(eq)
  @unpack degree, xg, wg = op
  nd = degree + 1
  @unpack dx, dy, xf, yf = grid
  @unpack boundary_condition, boundary_value = problem
  left, right, bottom, top = boundary_condition

  refresh!(u) = fill!(u, zero(eltype(u)))

  pre_allocated = cache.ghost_cache

   # Julia bug occuring here. Below, we have unnecessarily named
   # x1,y1, x2, y2,.... We should have been able to just call them x,y
   # Otherwise we were getting a type instability and variables were
   # called Core.Box. This issue is probably related to
   # https://discourse.julialang.org/t/type-instability-of-nested-function/57007
   # https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
   # https://github.com/JuliaLang/julia/issues/15276
   # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured-1

   # For Dirichlet bc, use upwind flux at faces by assigning both physical
   # and ghost cells through the bc.
   if left == dirichlet
      x1 = xf[1]
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            y1     = yf[j] + xg[k] * dy[j]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ub_value = problem.boundary_value(x1, y1, tq)
               fb_value = flux(x1, y1, ub_value, eq, 1)
               multiply_add_to_node_vars!(ub, wg[l], ub_value, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fb_value, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 2, 0, j)
            set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)

            # Purely upwind at boundary
            # if abs(y1) < 0.055
               set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
               set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
            # end
         end
      end
   elseif left == hllc_bc
      x1 = xf[1]
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            y1     = yf[j] + xg[k] * dy[j]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ub_value = problem.boundary_value(x1, y1, tq)
               fb_value = flux(x1, y1, ub_value, eq, 1)
               multiply_add_to_node_vars!(ub, wg[l], ub_value, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fb_value, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 2, 0, j)
            set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)

            ##############
            Ur_node = get_node_vars(Ub, eq, k, 1, 1, j)
            Fr_node = get_node_vars(Fb, eq, k, 1, 1, j)
            Ul_node = get_node_vars(Ub, eq, k, 2, 0, j)
            Fl_node = get_node_vars(Fb, eq, k, 2, 0, j)
            ual, uar = get_node_vars(ua, eq, 0, j), get_node_vars(ua, eq, 1, j)

            X = SVector{2}(x1, y1)
            Fn = hllc(X, ual, uar, Fl_node, Fr_node, Ul_node, Ur_node, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
            set_node_vars!(Fb, Fn     , eq, k, 1, 1, j)
            set_node_vars!(Fb, Fn     , eq, k, 2, 0, j)
            ################

            # # Purely upwind at boundary
            # set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
            # set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
         end
      end
   elseif left in (neumann, reflect)
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            Ub_node = get_node_vars(Ub, eq, k, 1, 1, j)
            Fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
            set_node_vars!(Ub, Ub_node, eq, k, 2, 0, j)
            set_node_vars!(Fb, Fb_node, eq, k, 2, 0, j)
            if left == reflect
               Ub[2, k, 2, 0, j] *= -1.0
               Fb[1, k, 2, 0, j] *= -1.0
               Fb[3, k, 2, 0, j] *= -1.0
               Fb[4, k, 2, 0, j] *= -1.0
            end
         end
      end
   elseif left == dirichlet_neumann
      x1 = xf[1]
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            y1     = yf[j] + xg[k] * dy[j]
            if abs(y1)<= 0.5+1e-15

               ub, fb = pre_allocated[Threads.threadid()]
               refresh!.(( ub, fb ))
               for l in Base.OneTo(nd)
                  tq = t + xg[l]*dt
                  ub_value = problem.boundary_value(x1, y1, tq)
                  fb_value = flux(x1, y1, ub_value, eq, 1)
                  multiply_add_to_node_vars!(ub, wg[l], ub_value, eq, 1)
                  multiply_add_to_node_vars!(fb, wg[l], fb_value, eq, 1)
               end
               ub_node = get_node_vars(ub, eq, 1)
               fb_node = get_node_vars(fb, eq, 1)
               set_node_vars!(Ub, ub_node, eq, k, 2, 0, j)
               set_node_vars!(Fb, fb_node, eq, k, 2, 0, j)

            # Purely upwind at boundary
            # if abs(y1) < 0.055
               set_node_vars!(Ub, ub_node, eq, k, 1, 1, j)
               set_node_vars!(Fb, fb_node, eq, k, 1, 1, j)
            # end
            else
               Ub_node = get_node_vars(Ub, eq, k, 1, 1, j)
               Fb_node = get_node_vars(Fb, eq, k, 1, 1, j)
               set_node_vars!(Ub, Ub_node, eq, k, 2, 0, j)
               set_node_vars!(Fb, Fb_node, eq, k, 2, 0, j)
            end
         end
      end

   
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   if right == dirichlet
      x2  = xf[nx+1]
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            y2  = yf[j] + xg[k] * dy[j]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ubvalue = boundary_value(x2, y2, tq)
               fbvalue = flux(x2, y2, ubvalue, eq, 1)
               multiply_add_to_node_vars!(ub, wg[l], ubvalue, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fbvalue, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 1, nx+1, j)
            set_node_vars!(Fb, fb_node, eq, k, 1, nx+1, j)

            # Purely upwind
            # set_node_vars!(Ub, ub_node, eq, k, 2, nx, j)
            # set_node_vars!(Fb, fb_node, eq, k, 2, nx, j)
         end
      end
   elseif right == hllc_bc
      x2  = xf[nx+1]
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            y2  = yf[j] + xg[k] * dy[j]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ubvalue = boundary_value(x2, y2, tq)
               fbvalue = flux(x2, y2, ubvalue, eq, 1)
               multiply_add_to_node_vars!(ub, wg[l], ubvalue, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fbvalue, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 1, nx+1, j)
            set_node_vars!(Fb, fb_node, eq, k, 1, nx+1, j)

            Ur_node = get_node_vars(Ub, eq, k, 1, nx+1, j)
            Fr_node = get_node_vars(Fb, eq, k, 1, nx+1, j)
            Ul_node = get_node_vars(Ub, eq, k, 2, nx, j)
            Fl_node = get_node_vars(Fb, eq, k, 2, nx, j)
            ual, uar = get_node_vars(ua, eq, nx, j), get_node_vars(ua, eq, nx+1, j)

            X = SVector{2}(x2, y2)
            Fn = hllc(X, ual, uar, Fl_node, Fr_node, Ul_node, Ur_node, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 2, nx, j)
            set_node_vars!(Fb, Fn     , eq, k, 2, nx, j)
            set_node_vars!(Fb, Fn     , eq, k, 1, nx+1, j)

            # Purely upwind
            # set_node_vars!(Ub, ub_node, eq, k, 2, nx, j)
            # set_node_vars!(Fb, fb_node, eq, k, 2, nx, j)
         end
      end
   elseif right in (reflect, neumann)
      @threaded for j=1:ny
         for k in Base.OneTo(nd)
            Ub_node = get_node_vars(Ub, eq, k, 2, nx, j)
            Fb_node = get_node_vars(Fb, eq, k, 2, nx, j)
            set_node_vars!(Ub, Ub_node, eq, k, 1, nx+1, j)
            set_node_vars!(Fb, Fb_node, eq, k, 1, nx+1, j)

            if right == reflect
               Ub[2, k, 1, nx+1, j] *= -1.0
               Fb[1, k, 1, nx+1, j] *= -1.0
               Fb[3, k, 1, nx+1, j] *= -1.0
               Fb[4, k, 1, nx+1, j] *= -1.0
            end
         end
      end
   else
      println("Incorrect bc specified at right.")
      @assert false
   end

   if bottom == dirichlet
      y3 = yf[1]
      @threaded for i=1:nx
         for k in Base.OneTo(nd)
            x3  = xf[i] + xg[k] * dx[i]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ubvalue = boundary_value(x3, y3, tq)
               fbvalue = flux(x3, y3, ubvalue, eq, 2)
               multiply_add_to_node_vars!(ub, wg[l], ubvalue, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fbvalue, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 4, i, 0)
            set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)

            # Purely upwind

            # set_node_vars!(Ub, ub, eq, k, 3, i, 1)
            # set_node_vars!(Fb, fb, eq, k, 3, i, 1)
         end
      end
   elseif bottom == hllc_bc
      y3 = yf[1]
        @threaded for i in 1:nx
            for k in Base.OneTo(nd)
                x3 = xf[i] + xg[k] * dx[i]
                ub, fb = pre_allocated[Threads.threadid()]
                refresh!.((ub, fb))
                for l in Base.OneTo(nd)
                    tq = t + xg[l] * dt_scaled
                    ubvalue = boundary_value(x3, y3, tq)
                    fbvalue = flux(x3, y3, ubvalue, eq, 2)
                    multiply_add_to_node_vars!(ub, wg_scaled[l], ubvalue, eq, 1)
                    multiply_add_to_node_vars!(fb, wg_scaled[l], fbvalue, eq, 1)
                end

                ub_node = get_node_vars(ub, eq, 1)
                fb_node = get_node_vars(fb, eq, 1)
                set_node_vars!(Ub, ub_node, eq, k, 4, i, 0)
                set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)

                Uu_node = get_node_vars(Ub, eq, k, 3, i, 1)
                Fu_node = get_node_vars(Fb, eq, k, 3, i, 1)
                Ud_node = get_node_vars(Ub, eq, k, 4, i, 0)
                Fd_node = get_node_vars(Fb, eq, k, 4, i, 0)
                uad, uau = get_node_vars(ua, eq, i, 0), get_node_vars(ua, eq, i, 1)

                X = SVector{2}(x3, y3)
                Fn = hllc(X, uad, uau, Fd_node, Fu_node, Ud_node, Uu_node, eq, 2)
                set_node_vars!(Ub, ub_node, eq, k, 3, i, 1)
                set_node_vars!(Fb, Fn     , eq, k, 3, i, 1)
                set_node_vars!(Fb, Fn     , eq, k, 4, i, 0)

                # Purely upwind

                # set_node_vars!(Ub, ub, eq, k, 3, i, 1)
                # set_node_vars!(Fb, fb, eq, k, 3, i, 1)
            end
        end
   elseif bottom in (reflect, neumann)
      @threaded for i=1:nx
         for k in Base.OneTo(nd)
            Ub_node = get_node_vars(Ub, eq, k, 3, i, 1)
            Fb_node = get_node_vars(Fb, eq, k, 3, i, 1) ##WHAT IS 3
            set_node_vars!(Ub, Ub_node, eq, k, 4, i, 0) ##WHAT IS 4
            set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)
            if bottom == reflect
               Ub[3, k, 4, i, 0] *= -1.0
               Fb[1, k, 4, i, 0] *= -1.0
               Fb[2, k, 4, i, 0] *= -1.0
               Fb[4, k, 4, i, 0] *= -1.0
            end
         end
      end
   ###############################
   elseif bottom == dirichlet_neumann
      y3 = yf[1]
      @threaded for i=1:nx
         for k in Base.OneTo(nd)
            x3 = xf[i] + xg[k] * dx[i]
            if abs(x3)<=0.5+1e-15 ##
               ub, fb = pre_allocated[Threads.threadid()]
               refresh!.(( ub, fb ))
               for l in Base.OneTo(nd)
                  tq = t + xg[l]*dt
                  ubvalue = boundary_value(x3, y3, tq)
                  fbvalue = flux(x3, y3, ubvalue, eq, 2)
                  multiply_add_to_node_vars!(ub, wg[l], ubvalue, eq, 1)
                  multiply_add_to_node_vars!(fb, wg[l], fbvalue, eq, 1)
               end
               ub_node = get_node_vars(ub, eq, 1)
               fb_node = get_node_vars(fb, eq, 1)
               set_node_vars!(Ub, ub_node, eq, k, 4, i, 0)
               set_node_vars!(Fb, fb_node, eq, k, 4, i, 0)
            else
               Ub_node = get_node_vars(Ub, eq, k, 3, i, 1)
               Fb_node = get_node_vars(Fb, eq, k, 3, i, 1) ##WHAT IS 3
               set_node_vars!(Ub, Ub_node, eq, k, 4, i, 0) ##WHAT IS 4
               set_node_vars!(Fb, Fb_node, eq, k, 4, i, 0)
            end
         end
      end
   ##################

   elseif periodic_y
      nothing
   else
      @assert typeof(bottom) <: Tuple{Any, Any, Any}
      bc! = bottom[1]
      bc!(grid, eq, op, Fb, Ub, aux)
   end

   if top == dirichlet
      y4 = yf[ny+1]
      @threaded for i=1:nx
         for k in Base.OneTo(nd)
            x4  = xf[i] + xg[k] * dx[i]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ubvalue = boundary_value(x4, y4, tq)
               fbvalue = flux(x4, y4, ubvalue, eq, 2)
               multiply_add_to_node_vars!(ub, wg[l], ubvalue, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fbvalue, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 3, i, ny+1)
            set_node_vars!(Fb, fb_node, eq, k, 3, i, ny+1)

            # Purely upwind
            # set_node_vars!(Ub, ub_node, eq, k, 4, i, ny)
            # set_node_vars!(Fb, fb_node, eq, k, 4, i, ny)
         end
      end
   elseif top == hllc_bc
      y4 = yf[ny+1]
      @threaded for i=1:nx
         for k in Base.OneTo(nd)
            x4  = xf[i] + xg[k] * dx[i]
            ub, fb = pre_allocated[Threads.threadid()]
            refresh!.(( ub, fb ))
            for l in Base.OneTo(nd)
               tq = t + xg[l]*dt
               ubvalue = boundary_value(x4, y4, tq)
               fbvalue = flux(x4, y4, ubvalue, eq, 2)
               multiply_add_to_node_vars!(ub, wg[l], ubvalue, eq, 1)
               multiply_add_to_node_vars!(fb, wg[l], fbvalue, eq, 1)
            end
            ub_node = get_node_vars(ub, eq, 1)
            fb_node = get_node_vars(fb, eq, 1)
            set_node_vars!(Ub, ub_node, eq, k, 3, i, ny+1)
            set_node_vars!(Fb, fb_node, eq, k, 3, i, ny+1)

            Uu_node = get_node_vars(Ub, eq, k, 3, i, ny+1)
            Fu_node = get_node_vars(Fb, eq, k, 3, i, ny+1)
            Ud_node = get_node_vars(Ub, eq, k, 4, i, ny)
            Fd_node = get_node_vars(Fb, eq, k, 4, i, ny)
            uad, uau = get_node_vars(ua, eq, i, ny), get_node_vars(ua, eq, i, ny+1)

            X = SVector{2}(x4, y4)
            Fn = hllc(X, uad, uau, Fd_node, Fu_node, Ud_node, Uu_node, eq, 2)
            set_node_vars!(Ub, ub_node, eq, k, 4, i, ny)
            set_node_vars!(Fb, Fn     , eq, k, 4, i, ny)
            set_node_vars!(Fb, Fn     , eq, k, 3, i, ny+1)
         end
      end
   elseif top in (reflect, neumann)
      @threaded for i=1:nx
         for k in Base.OneTo(nd)
            Ub_node = get_node_vars(Ub, eq, k, 4, i, ny)
            Fb_node = get_node_vars(Fb, eq, k, 4, i, ny)
            set_node_vars!(Ub, Ub_node, eq, k, 3, i, ny+1)
            set_node_vars!(Fb, Fb_node, eq, k, 3, i, ny+1)
            if top == reflect
               Ub[3, k, 3, i, ny+1] *= -1.0
               Fb[1, k, 3, i, ny+1] *= -1.0
               Fb[2, k, 3, i, ny+1] *= -1.0
               Fb[4, k, 3, i, ny+1] *= -1.0
            end
         end
      end
   elseif periodic_y
      nothing
   else
      @assert false "Incorrect bc specific at top"
   end
   if scheme.limiter.name == "blend"
      update_ghost_values_fn_blend!(eq, problem, grid, aux)
   end

   if scheme.limiter.name == "blend"
      update_ghost_values_fn_blend!(eq, problem, grid, aux)
   end

   return nothing
   end # timer
end

#-------------------------------------------------------------------------------
# Write solution to a vtk file
#-------------------------------------------------------------------------------
function SSFR.initialize_plot(eq::RHD2D, op, grid, problem, scheme, timer, u1, ua)
   return nothing
end

function write_poly(eq::RHD2D, grid, op, u1, fcount)
   filename = get_filename("output/sol", 3, fcount)
   @show filename
   @unpack xf, yf, dx, dy = grid
   nx, ny = grid.size
   @unpack degree, xg = op
   nd = degree + 1
   nvar = eq.nvar
   # Clear and re-create output directory

   u1_prim = zeros(size(u1))
   for j=1:ny,jj=1:nd,i=1:nx,ii=1:nd
      @views con2prim!(eq, u1[:,ii,jj,i,j], u1_prim[:,ii,jj,i,j])
   end

   nu = max(nd, 2)
   xu = LinRange(0.0, 1.0, nu)
   Vu = Vandermonde_lag(xg, xu)

   Mx, My = nx*nu, ny*nu
   grid_x = zeros(Mx)
   grid_y = zeros(My)
   for i=1:nx
      i_min = (i-1)*nu + 1
      i_max = i_min + nu-1
      # grid_x[i_min:i_max] .= LinRange(xf[i], xf[i+1], nu)
      grid_x[i_min:i_max] .= xf[i] .+ dx[i]*xg
   end

   for j=1:ny
      j_min = (j-1)*nu + 1
      j_max = j_min + nu-1
      # grid_y[j_min:j_max] .= LinRange(yf[j], yf[j+1], nu)
      grid_y[j_min:j_max] .= yf[j] .+ dy[j]*xg
   end

   # if nx == 111 ##Change
   #    diag_xx = LinRange(0,1,nx*nd)
   #    diag_density_arr = zeros(nvar,Mx)

   #    for s=1:size(u1_prim)[4]-2
   #       for ss=1:size(u1_prim)[2]
   #          diag_density_arr[:,nd*(s-1)+ss] = u1_prim[:,ss,ss,s,s]
   #       end
   #    end

   #    plotlyjs()
   #    plot_den = plot(diag_xx,diag_density_arr[1,:], color = :red, label = "den_xy")
   #    savefig(plot_den, "output/den_sol_xy.html")
   # else

      vtk_sol = vtk_grid(filename, grid_x, grid_y)

      for s = 1:nvar
         u_equi = zeros(Mx, My)
         u = zeros(nu)
         for j=1:ny
            for i=1:nx
               # to get values in the equispaced thing
               for jy=1:nd
                  i_min = (i-1)*nu + 1
                  i_max = i_min + nu-1
                  u_ = @view u1_prim[s,:,jy,i,j]
                  mul!(u, Vu, u_)
                  j_index = (j-1)*nu + jy
                  u_equi[i_min:i_max,j_index] .= @view u1_prim[s,:,jy,i,j]
               end
            end
         end
      
         if s == 1
            vtk_sol["Density"] = u_equi
            ln_u_equi = log.(u_equi)
            vtk_sol["log Density"] = ln_u_equi
         elseif s == 2
            vtk_sol["Velocity_x"] = u_equi
         elseif s == 3
            vtk_sol["Velocity_y"] = u_equi
         else
            if sum(u_equi.<0)>0 #SEE
               @show u_equi
            end
            ln_u_equi = log.(abs.(u_equi)) #TEMPORARY HACK
            vtk_sol["log Pressure"] = ln_u_equi
            vtk_sol["Pressure"] = u_equi
         end      
      end
      println("Wrote pointwise solution to $filename")

      out = vtk_save(vtk_sol)
   # end
end

function SSFR.write_soln!(base_name, fcount, iter, time, dt, eq::RHD2D,
                          grid, problem, param, op,
                          z, u1, aux, ndigits=3) ## z is ua
   @timeit aux.timer "Write solution" begin
   @unpack final_time = problem
   # Clear and re-create output directory
   if fcount == 0
      run(`rm -rf output`)
      run(`mkdir output`)
      save_mesh_file(grid, "output")
   end

   nx, ny = grid.size
   @unpack exact_solution = problem
   exact(x) = exact_solution(x[1],x[2],time)
   @unpack xc, yc = grid
   filename = get_filename("output/avg", ndigits, fcount)
   # filename = string("output/", filename)
   vtk = vtk_grid(filename, xc, yc)
   xy = [ [xc[i], yc[j]] for i=1:nx,j=1:ny ]
   # KLUDGE - Do it efficiently
   prim = @views copy(z[:,1:nx,1:ny])
   exact_data = exact.(xy)
   for j=1:ny,i=1:nx
      @views con2prim!(eq, z[:,i,j], prim[:,i,j])
   end
   density_arr = prim[1,1:nx,1:ny]
   ln_density_arr = log.(density_arr)
   velx_arr = prim[2,1:nx,1:ny]
   vely_arr = prim[3,1:nx,1:ny]
   pres_arr = prim[4,1:nx,1:ny]
   ln_pres_arr = log.(pres_arr)

   # if nx==111 #DIAG
   #    plotlyjs()
   #    diag_density_arr=[density_arr[s,s] for s in 1:size(density_arr)[1]]
   #    x_diag = sqrt.(xc.^2+yc.^2)
   #    plot_den = plot(x_diag,diag_density_arr, color = :red, label = "den_xy")
   #    savefig(plot_den, "output/den_xy.html")
   #    write_poly(eq, grid, op, u1, fcount)

   # elseif ny == 111 || ny == 111  || ny == 111
   #    #=
      
   #    #p_ua = plot() #[plot() for _=1:2];
   #    #labels = ["log Density", "log Pressure"]
   #    #for n=1:2
   #    #   if n==1
   #          uad = ln_density_arr[:,1]
   #       elseif n==2
   #          uad = ln_press_arr[:,1]
   #       end
   #    end
   #    =#
   #    plotlyjs()
   #    plot_den = plot(xc,density_arr[:,3], color = :red, label = "den_x")
   #    savefig(plot_den, "output/den_x.html")
   #    plot_pres = plot(xc,pres_arr[:,3], color = :blue, label = "pres_x")
   #    savefig(plot_pres, "output/pres_x.html")
   if nx == 1
      #plotlyjs()
      #=
      p_ua = [plot() for _=1:2];
      labels = ["log Density", "log Pressure"]
      for n=1:2
         if n==1
            uad = ln_density_arr[1,:]
         elseif n==2
            uad = ln_press_arr[1,:]
         end
         plot!(p_ua[n],yc,uad, color = :red, legend=false)
         savefig(p_ua, "output/avg_y.html")
      end
      =#
      plotlyjs()
      plot_den = plot(yc,density_arr[1,:], color = :red, label = "den_y")
      savefig(plot_den, "output/den_y.html")
      plot_pres = plot(yc,pres_arr[1,:], color = :blue, label = "pres_y")
      savefig(plot_pres, "output/pres_y.html")
   else
      # plotlyjs()
      # plot_den = plot(xc,density_arr[:,5], color = :red, label = "den_x")
      # savefig(plot_den, "output/den_x.html")

   #vtk["sol"] = density_arr
   vtk["Density"] = density_arr
   vtk["log density"] = ln_density_arr
   vtk["Velocity_x"] = velx_arr
   vtk["Velocity_y"] = vely_arr
   vtk["Pressure"] = pres_arr
   vtk["log pressure"] = ln_pres_arr

   vtk["CYCLE"] = iter
   vtk["TIME"] = time
   out = vtk_save(vtk)
   println("Wrote file ", out[1])
   write_poly(eq, grid, op, u1, fcount)
   
   if final_time - time < 1e-10
      cp("$filename.vtr","./output/avg.vtr")
      println("Wrote final average solution to avg.vtr.")
   end
   
   end

   fcount += 1

   # HDF5 file
   element_variables = Dict()
   element_variables[:density] = vec(density_arr)
   element_variables[:velocity_x] = vec(velx_arr)
   element_variables[:velocity_y] = vec(vely_arr)
   element_variables[:pressure] = vec(pres_arr)
   # element_variables[:indicator_shock_capturing] = vec(aux.blend.cache.alpha[1:nx,1:ny])
   filename = save_solution_file(u1, time, dt, iter, grid, eq, op, element_variables) # Save h5 file
   println("Wrote ", filename)
   return fcount
   end # timer
end

function save_solution_file(u_, time, dt, iter,
                            mesh,
                            equations, op,
                            element_variables=Dict{Symbol,Any}();
                            system="")
   # Filename without extension based on current time step
   output_directory = "output"
   if isempty(system)
      filename = joinpath(output_directory, @sprintf("solution_%06d.h5", iter))
   else
      filename = joinpath(output_directory, @sprintf("solution_%s_%06d.h5", system, iter))
   end

   solution_variables(u) = con2prim(equations, u) # For broadcasting

   nx, ny = mesh.size
   u = @view u_[:,:,:,1:nx, 1:ny] # Don't plot ghost cells

   # Convert to different set of variables if requested
   # Reinterpret the solution array as an array of conservative variables,
   # compute the solution variables via broadcasting, and reinterpret the
   # result as a plain array of floating point numbers
   # OffsetArray(reinterpret(eltype(ua), con2prim_.(reinterpret(SVector{nvariables(equation), eltype(ua)}, ua))))
   u_static_reinter = reinterpret(SVector{nvariables(equations),eltype(u)}, u)
   data = Array(reinterpret(eltype(u), solution_variables.(u_static_reinter)))

   # Find out variable count by looking at output from `solution_variables` function
   n_vars = size(data, 1)

   # Open file (clobber existing content)
   h5open(filename, "w") do file
      # Add context information as attributes
      attributes(file)["ndims"] = 2
      attributes(file)["equations"] =  "2D RHD Equations" #"2D Euler Equations"
      attributes(file)["polydeg"] = op.degree
      attributes(file)["n_vars"] = n_vars
      attributes(file)["n_elements"] = nx * ny
      attributes(file)["mesh_type"] = "StructuredMesh" # For Trixi2Vtk
      attributes(file)["mesh_file"] = "mesh.h5"
      attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
      attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
      attributes(file)["timestep"] = iter

      # Store each variable of the solution data
      var_names = ("Density", "Velocity x", "Velocity y", "Pressure")
      for v in 1:n_vars
         # Convert to 1D array
         file["variables_$v"] = vec(data[v, .., :])

         # Add variable name as attribute
         var = file["variables_$v"]
         attributes(var)["name"] = var_names[v]
      end

      # Store element variables
      for (v, (key, element_variable)) in enumerate(element_variables)
         # Add to file
         file["element_variables_$v"] = element_variable

         # Add variable name as attribute
         var = file["element_variables_$v"]
         attributes(var)["name"] = string(key)
      end
   end

   return filename
end

function get_equation(γ)
   name = "2d RHD Equations"
   numfluxes = Dict{String, Function}()
   nvar = 4
   return RHD2D(γ, nvar, name, initial_values, numfluxes)
end

(
export flux, update_ghost_values_lwfr!
)
end # @muladd

end
