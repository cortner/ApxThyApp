### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 3f1cfd12-7b86-11eb-1371-c5795b87ef5b
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
		  PlutoUI, BenchmarkTools, ForwardDiff, Printf, FFTW, Colors, 
		  Random, LinearMaps, Arpack
	include("tools.jl")
end;

# ╔═╡ 76e9a7f6-86a6-11eb-2741-6b8759be971b
md"""
## §5 Approximation in Moderate and High Dimension

After our interlude on polynomial approximation, rational approximation, etc, we now return to trigonometric approximation by in dimension ``d > 1``. That is, we will consider the approximation of functions 
```math
	f \in C_{\rm per}(\mathbb{R}^d)
	:= \big\{ g \in C(\mathbb{R}^d) \,|\, 
			  g(x + 2\pi \zeta) = g(x) \text{ for all } \zeta \in \mathbb{Z}^d \big\}.
```


Our three goals for this lecture are

* 5.1 Approximation in moderate dimension, ``d = 2, 3``
* 5.2 Spectral methods for solving PDEs in two and three dimensions
* 5.3 Approximation in "high dimension"
"""


# ╔═╡ 551a3a58-9b4a-11eb-2596-6dcc2edd334b
md"""
**TODO:** Review of Fourier series, trigonometric interpolation.
"""

# ╔═╡ a7e7ece4-9b48-11eb-26a8-213556563a81
md"""
## § 5.1 Approximation in Moderate Dimension

Our first task is to figure out how to construct approximations from trigonometric polynomials which are inherently one-dimensional objects.

### 2D Case

We will first develop all ideas in 2D, and then quickly generalize them to ``d`` dimensions.

Let ``f \in C_{\rm per}(\mathbb{R}^2)``, i.e., ``f(x_1, x_2)`` is continuous and ``2\pi`` periodic in each of the two coordinate directions. In particular, if we "freeze" the coordinate ``x_1 = \hat{x}_1`` then we obtain a one-dimensional function 
```math
	x_2 \mapsto f(\hat{x}_1, x_2) \in C_{\rm per}(\mathbb{R}).
```
which we can approximate by a trigonometric interpolant, 
```math
	f(\hat{x}_1, x_2) \approx I_N^{(2)} f(\hat{x}_1, x_2) 
	= \sum_{k_2 = -N}^N c_k(\hat{x}_1) e^{i k_2 x_2},
```
where the superscript ``(2)`` in ``I_N^{(2)}`` indicates that the interpolation is performed with respect to the ``x_2`` coordinate. 
"""

# ╔═╡ 102f6eca-9b4a-11eb-23b8-210bd9100faa
md"""
Since trigonometric interpolation is a continuous operation it follows that ``c_k(x_1)`` is again a continuous and ``2\pi``-periodic function of ``x_1``. This takes a bit of work to prove, and we won't really need it later so let's not worry too much about it. But if we accept this as fact, then we can now approximate each ``c_k(x_1)`` again by its trigonometric interpolant, 
```math
	c_{k_2}(x_2)  \approx I_N^{(1)} c_{k_2}(x_1) 
			= \sum_{k_1 = -N}^N c_{k_1 k_2} e^{i k_1 x_1}.
```
Inserting this identity above we deduce 
```math
	f(x_1, x_2) \approx I_N^{(1)} I_N^{(2)} f(x_1, x_2) = 
			\sum_{k_1, k_2 = -N}^N c_{k_1 k_2} e^{i (k_1 x_1 + k_2 x_2)}.
```
Indeed we will momentarily see that this in fact has very similar (excellent) approximation properties as we have seen in the univariate setting. 

We have constructed an approximation to ``f(x_1, x_2)`` in terms of a multi-variate basis 
```math
	e^{i (k_1 x_1 + k_2 x_2)} = e^{i k_1 x_1} e^{i k_2 x_2}.
```
Next, let us write down the interpolation conditions for ``I^{(1)} I^{(2)} f``: 
"""

# ╔═╡ 974d10ae-9b9c-11eb-324f-21c03e345ca4
md"""
* First, ``I^{(2)}_N f`` imposes the interpolation condition 
```math
	I^{(2)}_N f(x_1, \xi_j) = f(x_1, \xi_j), \qquad j = 0, \dots, 2N-1, \quad \forall x_1 \in [0, 2\pi).
```
where ``\xi_j = \pi/N`` are the university equispaced interpolation nodes, 
* Next, we have applied ``I^{(1)}_N`` giving ``I^{(1)}_N I^{(2)}_N f`` which restricts this identity only to the interpolation nodes, i.e., 
```math
	I^{(1)}_N I^{(2)}_Nf(\xi_{j_1}, \xi_{j_2}) = f(\xi_{j_1}, \xi_{j_2}) = f(\xi_i, \xi_j), 
	\qquad j_1, j_2 = 0, \dots, 2N-1.
```
That is, we are imposing identity between target ``f`` and interpolant ``I^{(1)}_N I^{(2)}_Nf`` on the tensor product grid 
```math
	\{ \boldsymbol{\xi}_{j_1 j_2}  = (\xi_{j_1}, \xi_{j_2}) \,|\, 
	   j_1, j_2 = 0, \dots, 2N-1 \}.
```
"""

# ╔═╡ 6e96c814-9b9a-11eb-1200-bf99536f9369
begin
	xgrid(N) = range(0, 2π - π/N, length=2N)
	xygrid(Nx, Ny=Nx) = (xgrid(Nx) * ones(2Ny)'), (ones(2Nx) * xgrid(Ny)')
end;

# ╔═╡ 4741a420-9b9a-11eb-380e-07fc50b805c9
let N = 8
	X = xgrid(8)
	P1 = plot(X, 0*X .+ π, lw=0, ms=3, m=:o, label = "x grid",
			  title = "Univariate grids", xlims = [-0.1, 2π], ylims = [-0.1, 2π])
	plot!(P1, 0*X .+ π, X, lw=0, ms=3, m=:o, label = "y grid")
	X, Y = xygrid(N)
	P2 = plot(X[:], Y[:], lw=0, m=:o, ms=3, label = "", size = (300,300), 
			  title = "Tensor grid (x,y)", xlims = [-0.1, 2π], ylims = [-0.1, 2π])
	plot(P1, P2, size = (400, 200))
end 

# ╔═╡ 95edae16-9b9d-11eb-2698-8d52b0f18a57
md"""
In particular we have seen that the order of applying the two interpolation operators does not matter, and we can now simply write ``I_N f = I_N^{(1)} I_N^{(2)} f``. 

Our next question is how to determine the coefficients of the 2D trigonometric interpolant: 
```math
	I_N f(x_1, x_2) = \sum_{k_1, k_2 = -N}^N \hat{F}_{k_1 k_2} e^{i (k_1 x_1 + k_2 x_2)}.
```
We can of course write down the interpolation conditions again and solve the linear system
```math
	A \hat{F} = F, \qquad \text{where} \qquad 
	A_{j_1 j_2, k_1 k_2} = \exp\big(i (k_1 \xi_{j_1} + k_2 \xi_{j_2})\big)
```
In 1D we used the fact that the corresponding operator could be inverted using the FFT. This still remains true:
"""

# ╔═╡ 2ff639c8-9b9f-11eb-000e-37bbbea50dc5
begin
	"univariate k-grid"
	kgrid(N) = [ 0:N; -N+1:-1 ]
		
	"""
	Evaluation of a two-dimensional trigonometric polynomial
	Note that we only need the univariate k-grid since we just evaluate 
	it in a double-loop!
	"""
	function evaltrig(x, y, F̂::Matrix) 
		Nx, Ny = size(F̂)
		Nx = Nx ÷ 2; Ny = Ny ÷ 2
		return sum( real(exp(im * (x * kx + y * ky)) * F̂[jx, jy])
			        for (jx, kx) in enumerate(kgrid(Nx)), 
		                (jy, ky) in enumerate(kgrid(Ny)) )
	end
	
	"""
	2D trigonometric interpolant via FFT 
	"""
	triginterp2d(f::Function, Nx, Ny=Nx) = 
			fft( f.(xygrid(Nx, Ny)...) ) / (4 * Nx * Ny)
end;

# ╔═╡ bedaf224-9b9e-11eb-0a7d-ad170bfd73a7
md"""
Maybe this is a good moment to check - numerically for now - whether the excellent approximation properties that we enjoyed in 1D are retained! We start with a 2D version of the periodic witch of Agnesi.
"""

# ╔═╡ f7454b4a-9bc2-11eb-2ab9-dfc016fd57de
md"""
We will later prove a theoretical result that correctly predicts this rate.

Our first reaction is that this is wonderful, we obtain the same convergence rate as in 1D. But we have to be careful! While the rate is the same in terms of the degree ``N`` the **cost** associated with evaluating ``f`` now scales like ``N^2``. So in terms of the **cost**, the convergence rate we observe here is 
```math
	\epsilon = {\rm error} \sim \exp\Big( - \alpha \sqrt{{\rm cost}} \Big)
```
or equivalently, 
```math
	{\rm cost}  \sim \alpha^{-1} |\log \epsilon|^2.
```
We call this *polylogarithmic cost*. This can become prohibitive in high dimension; we will return to this in the third part of the lecture.
"""

# ╔═╡ 1506eaf4-9b9a-11eb-170d-95834314fb84
md"""
### General Case

Let's re-examine the approximation we constructed, 
```math
	f(x_1, x_2) \approx \sum_{k_1, k_2} c_{k_1 k_2} e^{i k_1 x_1} e^{i k_2 x_2}.
```
These 2-variate basis functions
```math
e^{i k_1 x_1} e^{i k_2 x_2}
```
are *tensor products* of the univariate basis ``e^{i k x}``. One normally writes
```math
	(a \otimes b)(x_1, x_2) = a(x_1) b(x_2).
```
If we define ``\varphi_k(x) := e^{i k x}`` then 
```math
	(\varphi_{k_1} \otimes \varphi_{k_2})(x_1, x_2) = \varphi_{k_1}(x_1) \varphi_{k_2}(x_2)
	= e^{i k_1 x_1} e^{i k_2 x_2}
```
"""

# ╔═╡ 1eb00bb6-9bc8-11eb-3af9-c33ef5b6ab15
md"""
Now suppose that we are in ``d`` dimensions, i.e., ``f \in C_{\rm per}(\mathbb{R}^d)``, then we can approxiate it by the ``d``-dimensional tensor products: let 
```math
\begin{aligned}
	\varphi_{\bf k}({\bf x}) &:= \Big(\otimes_{t = 1}^d \varphi_{k_t} \Big)({\bf x}), \qquad \text{or, equivalently,} \\ 
	\varphi_{k_1 \cdots k_d}(x_1, \dots, x_d)
	&= 
	\prod_{t = 1}^d \varphi_{k_t}(x_t)
	= \prod_{t = 1}^d e^{i k_t x_t}
	= \exp\big(i {\bf k} \cdot {\bf x}).
\end{aligned}
```
"""

# ╔═╡ 28aa73fe-9bc8-11eb-3266-f725f73a3159
md"""
The interpolation condition generalises similarly: 
```math
   \sum_{{\bf k} \in \{-N,\dots,N\}^d} \hat{F}_{\bf k} e^{i {\bf k} \cdot \boldsymbol{\xi} } = f(\boldsymbol{\xi}) 
	\qquad \forall \boldsymbol{\xi} = (\xi_{j_1}, \dots, \xi_{j_d}), \quad 
	j_t = 0, \dots, 2N-1.
```
And the nodal interpolant can be evaluated using the multi-dimensional Fast Fourier transform. Here we implement it just for 3D since we won't need it beyond three dimensions for now.
"""

# ╔═╡ 9b31019a-9bc8-11eb-0372-43ea0c0d8fc3
begin
	
	function xyzgrid(Nx, Ny=Nx, Nz=Nx) 
		X = [ x for x in xgrid(Nx), y = 1:2Ny, z = 1:2Nz ]
		Y = [ y for x in 1:2Nx, y in xgrid(Ny), z = 1:2Nz ]
		Z = [ z for x in 1:2Nx, y = 1:2Nz, z in xgrid(Nz) ]
		return X, Y, Z
	end

	"""
	Evaluation of a three-dimensional trigonometric polynomial
	Note that we only need the univariate k-grid since we just evaluate 
	it in a double-loop!
	"""
	function evaltrig(x, y, z, F̂::Array{T,3}) where {T} 
		Nx, Ny, Nz = size(F̂)
		Nx = Nx ÷ 2; Ny = Ny ÷ 2; Nz = Nz ÷ 2
		return sum( real(exp(im * (x * kx + y * ky + z * kz)) * F̂[jx, jy, jz])
			        for (jx, kx) in enumerate(kgrid(Nx)), 
		                (jy, ky) in enumerate(kgrid(Ny)),
						(jz, kz) in enumerate(kgrid(Nz)) )
	end
	
	"""
	2D trigonometric interpolant via FFT 
	"""
	triginterp3d(f::Function, Nx, Ny=Nx, Nz=Nx) = 
			fft( f.(xyzgrid(Nx, Ny, Nz)...) ) / (8 * Nx * Ny * Nz)

	;
end

# ╔═╡ bd5807fc-9b9e-11eb-1434-87af91d2d296
let N = 8, f = (x, y) -> exp(-cos(x)^2-cos(y)^2-cos(x)*cos(y))
	# evaluate the function at the interpolation points 
	X, Y = xygrid(N)
	F = f.(X, Y)
	# transform to trigonometric polynomial coefficients
	F̂ = fft(F) / (2N)^2
	# evaluate the trigonometric polynomial at the interpolation nodes
	Feval = evaltrig.(X, Y, Ref(F̂))
	# check that it satisfies the interpolation condition
	F ≈ Feval
	# and while we are at it, we can also check that this is the same 
	# as inverting the FFT
	Finv = real.(ifft(F̂)) * (2N)^2
	F ≈ Feval ≈ Finv
end

# ╔═╡ bdcd0340-9b9e-11eb-0ee6-e7e0993e8fe8
let f = (x, y) -> 1 / (1 + 10 * cos(x)^2 + 10 * cos(y)^2), NN = 4:2:20
	Xerr, Yerr = xygrid(205)
	err(N) = norm( f.(Xerr, Yerr) - evaltrig.(Xerr, Yerr, Ref(triginterp2d(f, N))), Inf )
	plot(NN, err.(NN), lw=3, ms=4, m=:o, label = "",
		 xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10, size = (400, 250) )		
	plot!(NN[5:end], 2* exp.( - NN[5:end] / sqrt(10) ), lw = 2, ls=:dash, 
		  label = L"\sim \exp( - N / \sqrt{10} )")
end

# ╔═╡ c9b0923c-9bce-11eb-3861-4d201b1765a8
md"""
Let us test this implementation on a three-dimensional periodic witch of agnesi: 
```math
	f(x, y, z) = \frac{1}{1 + c (\cos^2 x + \cos^2 y + \cos^2 z)}
```
"""

# ╔═╡ abc29124-9bc8-11eb-268d-c5115c341b58
let f = (x,y,z) -> 1 / (1 + 10 * (cos(x)^2+cos(y)^2+cos(z)^2)), N = 4
	# evaluate the function at the interpolation points 
	X, Y, Z = xyzgrid(N)
	F = f.(X, Y, Z)
	# transform to trigonometric polynomial coefficients
	F̂ = fft(F) / (2N)^3
	# evaluate the trigonometric polynomial at the interpolation nodes
	Feval = evaltrig.(X, Y, Z, Ref(F̂))
	# check that it satisfies the interpolation condition; also check that 
	# this is the same as inverting the FFT
	Finv = real.(ifft(F̂)) * (2N)^3
	F ≈ Feval ≈ Finv		
end

# ╔═╡ 8de3bc38-9b50-11eb-2ed2-436e4a3da804
md"""
Of course what we are doing here is **complete madness** - the cost of each call 
```julia
evaltrig(x, y, z, Ref(F̂))
```
is ``O(N^3)``! We will therefore try to avoid evaluating trigonometric polynomials on general sets, but only on grids. And when we do that, then we can just use the `ifft`!! Then we can evaluate the trigonometric polynomial at ``O(M^3)`` gridpoints for ``O( (M \log M)^3 )`` operations instead of ``O( (M N)^3 )``.
"""

# ╔═╡ a0c7fb86-9b50-11eb-02c6-37fc9fc78d57
begin 
	"""
	Note this function can evaluate the trigonometric interpolant on a 
	grid that is much finer than the one we used to construct it!
	The implementation is messy, but the idea is simple.
	"""
	function evaltrig_grid(F̂::Matrix, Mx, My=Mx)
		Nx, Ny = size(F̂); Nx ÷= 2; Ny ÷= 2
		Ix = 1:Nx+1; Jx = Nx+2:2Nx; Kx = (2Mx-Nx+2):2Mx 
		Iy = 1:Ny+1; Jy = Ny+2:2Ny; Ky = (2My-Ny+2):2My		
		Ĝ = zeros(ComplexF64, 2Mx, 2My)
		Ĝ[Ix, Iy] = F̂[Ix, Iy] 
		Ĝ[Ix, Ky] = F̂[Ix, Jy] 
		Ĝ[Kx, Iy] = F̂[Jx, Iy]
		Ĝ[Kx, Ky] = F̂[Jx, Jy]
		return real.(ifft(Ĝ) * (4 * Mx * My))
	end 
	
	function trigerr(f::Function, F̂::Matrix, Mx, My=Mx)
		G = evaltrig_grid(F̂, Mx, My)
		X, Y = xygrid(Mx, My) 
		return norm(f.(X, Y) - G, Inf)		
	end
end

# ╔═╡ 047a3e4a-9c83-11eb-142c-73e070fd4731
let f = (x, y) -> sin(x) * sin(y), N = 2, M = 30
	F̂ = triginterp2d(f, N)
	# coarse grid function 
	xc = xgrid(N)
	Fc = real.(ifft(F̂) * (2N)^2)
	# fine grid function 
	xf = xgrid(M)
	Ff = evaltrig_grid(F̂, M)
	plot( surface(xc, xc, Fc, colorbar=false), 
		  surface(xf, xf, Ff, colorbar=false), size = (500, 200) )
end

# ╔═╡ aed51e0c-9b50-11eb-2b2b-0553df4ec03b
md"""
### Approximation results

Approximation in dimension ``d > 1`` is much more subtle than in one dimension. Here, we can only focus on some basic results, but we will return to explore some of that complexity in the third part of this lecture. 

In this first result we are making a very simple statement: if ``f`` has a uniform one-dimensional regularity along all possible one-dimensional coordinate direction slices of ``d``-dimensional space, then the one-dimensional convergence results are recovered up to possibly worse constants (which we sweep under the carpet) and ``d``-dependent logarithmic factors.

**Theorem:** Suppose that ``f \in C_{\rm per}(\mathbb{R}^d)`` that
```math
    x_t \mapsto f({\bf x}) 
	\qquad 
	\begin{cases}
		\in C^p, \\ 
		\text{is analytic and bdd in } \Omega_\alpha,
	\end{cases}
	\qquad \forall {\bf x} \in [0, 2\pi)^d
```
where ``x_t \mapsto f({\bf x})`` means that we keep all other coordinates fixed! Then, 
```math
	\| f - I_N f \|_{L^\infty(\mathbb{R}^d)} 
	\lesssim 
	(\log N)^d \cdot 
	\begin{cases}
		  N^{-p}, \\ 
		 e^{-\alpha N}
	\end{cases}
```

**Proof:** Whiteboard or [LN, Sec. 8.1, 8.2].

Note: for the sake of simplicity we won't go into the subtleties of additional factors due to the modulus of continuity, but this can be easily incorporated.
"""

# ╔═╡ b80ba91e-9b50-11eb-2e67-bfd1a71f069e
md"""
## §5.2 Spectral Methods in 2D and 3D

We now return to the solution of (partial) differential equations using trigonometric polynomials, i.e. spectral methods. The ideas carry over from the one-dimensional setting without essential changes. 

### §5.2.0 Review Differentiation Operators

Recall from §2 that the fundamental property of Fourier spectral methods that we employed is that, if 
```math
	u_N(x) = \sum_{k = -N}^N \hat{U}_k e^{i k x}
```
is a trigonometric polynomial, then 
```math 
	u_N'(x) = \sum_{k = -N}^N \big[ i k \hat{U}_k \big]  e^{i k x}, 
```
that is, all homogeneous differentiation operators are *diagonal* in Fourier space. This translates of course to ``d`` dimensions: if 
```math 
	u_N({\bf x}) = \sum_{{\bf k}} \hat{U}_{\bf k} e^{ i {\bf k} \cdot {\bf x}}, 
```
then 
```math 
	\frac{\partial u_N}{\partial x_t}({\bf x}) = 
		\sum_{{\bf k}} \big[ i k_t \hat{U}_{\bf k} \big] e^{ i {\bf k} \cdot {\bf x}}, 
```
More generally, if ``L`` is a homogeneous differential operator, 
```math
	Lu = \sum_{\bf a} c_{\bf a} \prod_{t=1}^d \partial_{x_t}^{a_t} u, 
```
then this becomes 
```math 
	\widehat{Lu}(k) = \hat{L}(k) \hat{u}(k).
```
where 
```math
	\hat{L}(k) = \sum_{\bf a} c_{\bf a} \prod_{t=1}^d (i k_{t})^{a_t} 
```
We will now heavily use this property to efficiently evaluate differential operators.
"""

# ╔═╡ 677ed7ee-9c15-11eb-17c1-bf6e27df8dba
md"""
### §5.2.1 Homogeneous elliptic boundary value problem

We begin with a simple boundary value problem the *biharmonic equation*, a 4th order PDE modelling thin structures that react elastically to external forces.
```math
	\Delta^2 u = f, \qquad \text{ with PBC.}	
```

[TODO whiteboard: derive the multiplier] 

If ``f_N, u_N`` are trigonometric polynomials and ``f_N = \Delta^2 u_N`` then we can write 
```math
	\hat{F}_{\bf k} = |{\bf k}|^4 \hat{U}_{\bf k}
```
This determines ``\hat{U}_k`` except when ``k = 0``. Since the PDE determines the solution only up to a constant we can either prescribe that constant or pick any constant that we like. It  is common to require that ``\int u_N = 0``, which amounts to ``\hat{U}_0 = 0``. 

"""

# ╔═╡ 1abf39ea-9c19-11eb-1bb0-178d4f3e7de4
begin
	# kgrid(N) = [0:N; -N+1:-1]
	kgrid2d(Nx, Ny=Nx) = (
			[ kx for kx in kgrid(Nx), ky in 1:2Ny ], 
			[ ky for kx in 1:2Nx, ky in kgrid(Ny) ] )
end

# ╔═╡ ba97522c-9c16-11eb-3734-27688484ef6f
let N = 64, f = (x, y) -> exp(-3(cos(x)sin(y))) - exp(-3(sin(x)cos(y)))
	F̂ = triginterp2d(f, N)
	Kx, Ky = kgrid2d(N)
	L̂ = (Kx.^2 + Ky.^2).^2
	L̂[1] = 1
	Û = F̂ ./ L̂
	Û[1] = 0
	U = real.(ifft(Û) * (2N)^2)
	x = xgrid(N)
	contourf(x, x, U, size = (300,300), colorbar=false)
end

# ╔═╡ 9c7a4006-9c19-11eb-15ac-890dce21f2ec
md"""

### Error Analysis

The error analysis proceeds essentially as in the one-dimensional case. We won't give too many details here, but only confirm that neither the results nor the techniques change fundamentally.

To proceed we need just one more ingredient: the multi-dimensional Fourier series. We just state the results without proof: Let ``f \in C_{\rm per}(\mathbb{R}^d)`` then 
```math
	f({\bf x}) = \sum_{{\bf k} \in \mathbb{Z}^d} \hat{f}_{\bf k} e^{i {\bf k} \cdot {\bf x}},
```
where the convergence is in the least square sense (``L^2``), and uniform if ``f`` is e.g. Hölder continuous. 

Thus, for sufficiently smooth ``f`` we can write the solution of the biharmonic equation ``u`` also as a Fourier series with coefficients 
```math
	\hat{u}_{\bf k} = 
	\begin{cases}
		\hat{f}_{\bf k} / |{\bf k}|^{-4}, & {\bf k} \neq 0, \\ 
 	    0, & \text{otherwise.}
	\end{cases}
```
Note that this requires ``\hat{f}_{\bf 0} = 0``. 

Since ``|{\bf k}|^{-4}`` is summable it follows readily that the equation is max-norm, stable, i.e., 
```math
	\|u\|_\infty \leq C \|f\|_\infty,
```
and we cannow argue as in the 1D case that 
```math
	\Delta^2 (u - u_N) = f - f_N \qquad \Rightarrow \qquad 
	\|u - u_N \|_\infty \leq C \| f - f_N \|_\infty.
```
Thus, the approximation error for ``f - f_N`` translates into an approximation error for the solution. 

"""

# ╔═╡ e6b1374a-9c7f-11eb-1d90-1d6d40914d90
md"""
We can test the result for a right-hand side where we have a clearly defined rate, e.g., 
```math
	f(x, y) = \frac{1}{1 + 10 (\cos^2 x + \cos^2 y)}.
```
According to our results above we expect the rate 
```math 
	e^{-\alpha N}, \qquad \text{where} \qquad 
	\alpha = \sinh^{-1}(1 / \sqrt{10})
```
"""

# ╔═╡ 56eeada8-9c80-11eb-1422-79f53b49b47e
let f = (x,y) -> 1 / (1 + 10 * (cos(x)^2 + cos(y)^2)), NN = 4:4:40, M = 300
	err(N) = trigerr(f, triginterp2d(f, N), M)
	plot(NN, err.(NN), lw=2, ms=4, m=:o, label = "error", 
		 yscale = :log10, size = (300, 250), 
	     xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty")
	α = asinh(1 / sqrt(10))
	plot!(NN[5:end], 2 * exp.( - α * NN[5:end]), c=:black, lw=2, ls=:dash, label = L"\exp(-\alpha N)")
end

# ╔═╡ 64a5c40a-9c84-11eb-22fa-ff8524822e62
md"""
We will leave the domain of rigorous error analysis now and focus more on issues of implementation. Although everything we do here can be easily extended to 3D, we will focus purely on 2D since the runtimes are more manageable and the visualization much easier.
"""

# ╔═╡ 6c189d08-9c15-11eb-1d47-a902e8997cc3
md"""
### §5.2.2 A 2D transport equation

```math
	u_t + {\bf v} \cdot \nabla u = 0
```
We discretize this as 
```math
	\frac{d\hat{U}_{\bf k}}{dt}
	+ i ({\bf v} \cdot {\bf k}) \hat{U}_{\bf k} = 0
```
And for the time-discretisation we use the leapfrog scheme resulting in 
```math
	\frac{\hat{U}_{\bf k}^{n+1} - \hat{U}_{\bf k}^{n-1}}{2 \Delta t}
	= - i ({\bf v} \cdot {\bf k}) \hat{U}_{\bf k}^n.
```
"""

# ╔═╡ 993e30d8-9c86-11eb-130a-41a1bb825e7c
let u0 = (x, y) -> exp(- 3 * cos(x) - cos(y)), N = 20, Δt = 3e-3, Tf = 2π
		v = [1, 1]
	
	Kx, Ky = kgrid2d(N)
	dt_im_vdotk = 2 * Δt * im * (v[1] * Kx + v[2] * Ky)
	Û1 = triginterp2d(u0, N)
	Û0 = Û1 + 0.5 * dt_im_vdotk .* Û1
	
	xx = xgrid(N)
	
	t = 0.0
	@gif for _ = 1:ceil(Int, Tf/Δt)
		Û0, Û1 = Û1, Û0 - dt_im_vdotk .* Û0
		contourf(xx, xx, real.(ifft(Û1) * (2N)^2), 
			     colorbar=false, size = (300, 300), 
				 color=:viridis)
	end every 15
end

# ╔═╡ 704b6176-9c15-11eb-11ba-ffb4491a6003
md"""

### §5.2.3 The Cahn--Hilliard Equation

Next, we solve a nonlinear evolution equation: the Cahn--Hilliard equation, which models phase separation of two intermixed liquids,
```math
	(-\Delta)^{-1} u_t = \epsilon \Delta u - \frac{1}{\epsilon} (u^3 - u)
```
The difficulty here is that a trivial semi-implicit time discretisation
```math
	u^{(n+1)} + \epsilon \tau \Delta^2 u^{(n+1)} = 
	u^{(n)} + \frac{\tau}{\epsilon} \Delta (u^3 - u)
```
has time-step restriction ``O( \epsilon N^{-2} )``. We can stabilise with a (local) convex-concave splitting such as
```math
	(1 + \epsilon \tau \Delta^2 - C \tau \Delta) u^{(n+1)}
	= (1-C \tau \Delta) u^{(n)} + \frac{\tau}{\epsilon} \Delta (u^3 - u)^{(n)}
```
Since ``(u^3-u)' = 3 u^2 - 1 \in [-1, 2]`` we need ``C \geq 2/\epsilon`` to get ``\tau``-independent stability. We then choose the time-step ``\tau = h \epsilon`` to make up for the loss of accuracy.

In reciprocal space, the time step equation becomes
```math
	(1+\epsilon \tau |k|^4 + C \tau |k|^2) \hat{u}^{(n+1)} 
	= 
	\big(1+C\tau |k|^2 + \frac{\tau}{\epsilon} |k|^2\big) \hat{u}^{(n)} 
	- \frac{\tau}{\epsilon} |k|^2 (\widehat{u^3})^{(n)}
```
(For something more serious we should probably implement a decent adaptive time-stepping strategy.)	
"""


# ╔═╡ fd07f1d2-9c89-11eb-023f-e96ccdd43a7e
let N = 64, ϵ = 0.1,  Tfinal = 8.0
	h = π/N     # mesh size 
	C = 2/ϵ     # stabilisation parameter
	τ = ϵ * h   # time-step 

	# real-space and reciprocal-space grids, multipliers
	xx = xgrid(N)
	Kx, Ky = kgrid2d(N)
	Δ = - Kx.^2 - Ky.^2

	# initial condition
	U = rand(2N, 2N) .- 0.5

	# time-stepping loop
	@gif for n = 1:ceil(Tfinal / τ)
		F̂ =  (1 .- C*τ*Δ) .* fft(U) + τ/ϵ * (Δ .* fft(U.^3 - U))
		U = real.(ifft( F̂ ./ (1 .+ ϵ*τ*Δ.^2 - C*τ*Δ) ))
		contourf(xx, xx, U, color=:viridis, 
			     colorbar=false, size = (400,400))
	end every 3
end

# ╔═╡ 7489004a-9c15-11eb-201d-91f34cb40c6f
# An equivalent formulation is 
# ```math
# 	u = \arg\min_{\|u\|_{L^2} = 1} 
# 		\int \frac12 \Big(|\nabla u|^2 + V(x) u^2\Big) + \frac14 u^4 \, dx.
# ```

md"""

### §5.2.4 An eigenvalue problem 

We consider a Schrödinger-type eigenvalue problem 

```math
	- \Delta u + V(x) u = \lambda u, \qquad \|u\|_{L^2} = 1.
```

* Approximate ``u \approx u_N``, ``V \approx V_N := I_N V``
* Discretise ``V u \approx V_N u_N \approx I_N(V_N u_N)`` and ``u^3 \approx u_N^3 \approx I_N u_N^3``
```math
	-\Delta u_N + I_N \big[ V_N u_N \big] = \lambda_N u_N
```
Then evaluate the term ``\Delta u_N`` in reciprocal space.

To solve this nonlinear system via `eig` we would need to assemble the matrix representing the linear operator 
```math
	L u_N = -\Delta u_N + I_N \big[ V_N u_N \big].
```
An alternative (sometimes better) approach is to supply the action of the linear operator on vectors (functions). This can then be used in iterative solvers. The functionality for this is implemented in [`LinearMaps.jl`](https://jutho.github.io/LinearMaps.jl/stable/).
"""

# ╔═╡ 2617153a-9ed3-11eb-35d2-fd79230055f4
let N = 64
	Random.seed!(10)
	Rs = [ 2*π*rand(2) for n = 1:10 ]
	Vfun = (x1,x2) -> sum( exp( -30*(-1+cos(x1-x[1]))^2-30*(-1+cos(x2-x[2]))^2 ) for x in Rs )
	xp = range(0, 2π, length=100)
	P1 = contourf(xp, xp, Vfun, colorbar=false, title = L"V")
	
	# discretised potential 
	X, Y = xygrid(N)
	V = Vfun.(X, Y)
	v = V[:]
	
	# multipliers 
	Kx, Ky = kgrid2d(N)
	Δ̂ = (Kx.^2 + Ky.^2)[:]

	# construct the linear PDE operator; we need to convert between 
	# vectors and arrays. The map is self-adjoint!
	L = u -> real.(ifft( (Kx.^2+Ky.^2) .* fft(reshape(u, 2N, 2N)) )[:]) .+ v .* u

	A = LinearMap(L, L, (2N)^2, ismutating=false)
	λ, u = Arpack.eigs(A; nev = 1, which=:SR, tol = 1e-6, maxiter=1_000)
	λ = real(λ[1]); u = real.(u) * (2N)^2
	
	# @show norm(L(u) - λ * u), norm(u) / (2N)^2
	
	U = reshape(u, 2N, 2N)
	xp = xgrid(N)
	P2 = contourf(xp, xp, real.(U), colorbar=false, title = L"u  \qquad (\lambda = %$(round(λ, digits=3)))")
	plot(P1, P2, size = (600, 300))
end

# ╔═╡ 5bebcb4c-9ee2-11eb-28b3-370f4d719493
md"""
The next step might be to solve a nonlinear eigenvalue problem: e.g. the non-Linear Gross-Pitaevskii type eigenvalue problem [[wiki]](https://en.wikipedia.org/wiki/Gross–Pitaevskii_equation)
```math
	- \Delta u + V(x) u + u^3 = \lambda u, \qquad \|u\|_{L^2} = 1.
```
describes the ground state of a quantum system of identical bosons. In numerical analysis it is also commonly used as a toy model for the much more complex quantum mechanical models for ferminons. Some problems are similar, especially their structure as nonlinear eigenvalue problems. (please talk to Professor Chen about this!)

Here, one would now need to add an outer iteration to solve the nonlinearity. This is much more challenging to do in a robust way; see e.g. [[Dusson & Maday]](https://hal.sorbonne-universite.fr/hal-00903715/document) and the following hidden code which does not work!!
"""



# ╔═╡ 658bd0bc-9ee6-11eb-052e-c7f8f9536633
# To solve this nonlinear system we use a sequence of eigenvalue problems. Suppose that at step ``t`` of this sequence ``u_N^{t}`` is an approximation to the solution. Then we improve the solution by solving *linear eigenvalue problem*, 
# ```math
# 	- \Delta u_N^{t+1} + I_N\big[  \big(V_N + (u_N^t)^2\big) u_N^{t+1} \big]
# 		= \lambda_N^{t+1} u_N^{t+1}.
# ```
# **IF** the sequence converges, ``u_N^t \to u_N``, then the limit ``u_N`` must solve the discretised nonlinear eigenvalue problem.


# let N = 32
# 	Random.seed!(10)
# 	Rs = [ 2*π*rand(2) for n = 1:10 ]
# 	Vfun = (x1,x2) -> sum( exp( -30*(-1+cos(x1-x[1]))^2-30*(-1+cos(x2-x[2]))^2 ) for x in Rs )
# 	xp = range(0, 2π, length=100)
	
# 	# discretised potential 
# 	X, Y = xygrid(N)
# 	V = Vfun.(X, Y)
# 	v = V[:]
	 
# 	# multipliers 
# 	Kx, Ky = kgrid2d(N)
# 	Δ̂ = (Kx.^2 + Ky.^2)

# 	function grosspitmap(Veff) 
# 		L = u -> real.(ifft( Δ̂ .* fft(reshape(u, 2N, 2N)) )[:]) .+ Veff[:] .* u
# 		return LinearMap(L, L, (2N)^2, ismutating=false)
# 	end
	
	
# 	τ = 0.03
# 	λ = 0.5
# 	u = zeros((2N)^2)
# 	for t = 1:100
# 		A = grosspitmap(v + u.^2)
# 		@show norm(A * u - λ * u, Inf)
# 		λ, ũ = Arpack.eigs(A; nev = 1, which=:SR, tol = 1e-10, maxiter=1_000)
# 		λ = real(λ[1])
# 		u = (1-τ) * u + τ * ( real.(ũ)/ (norm(real.(ũ))/(2N)) )[:,1]
# 		u = u / (norm(u)/(2N))
# 	end 
	
# 	U = reshape(u, 2N, 2N)
# 	xp = xgrid(N)
# 	contourf(xp, xp, real.(U), 
# 			colorbar=false, title = L"u  \qquad (\lambda = %$(round(λ, digits=3)))", 
# 			size = (300,300))
# end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Arpack = "~0.5.3"
BenchmarkTools = "~1.3.1"
Colors = "~0.12.8"
DataFrames = "~1.3.2"
FFTW = "~1.4.6"
ForwardDiff = "~0.10.25"
LaTeXStrings = "~1.3.0"
LinearMaps = "~3.6.0"
Plots = "~1.27.2"
PlutoUI = "~0.7.37"
PrettyTables = "~1.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "e99b76cded02965cba0ed9103cc249efa158a0f2"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.6.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "90021b03a38f1ae9dbd7bf4dc5e3dcb7676d302c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "28ef6c7ce353f0b35d0df0d5930e0d072c1f5b9b"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6976fab022fea2ffea3d945159317556e5dad87c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─3f1cfd12-7b86-11eb-1371-c5795b87ef5b
# ╟─76e9a7f6-86a6-11eb-2741-6b8759be971b
# ╟─551a3a58-9b4a-11eb-2596-6dcc2edd334b
# ╟─a7e7ece4-9b48-11eb-26a8-213556563a81
# ╟─102f6eca-9b4a-11eb-23b8-210bd9100faa
# ╟─974d10ae-9b9c-11eb-324f-21c03e345ca4
# ╠═6e96c814-9b9a-11eb-1200-bf99536f9369
# ╟─4741a420-9b9a-11eb-380e-07fc50b805c9
# ╟─95edae16-9b9d-11eb-2698-8d52b0f18a57
# ╠═2ff639c8-9b9f-11eb-000e-37bbbea50dc5
# ╠═bd5807fc-9b9e-11eb-1434-87af91d2d296
# ╟─bedaf224-9b9e-11eb-0a7d-ad170bfd73a7
# ╟─bdcd0340-9b9e-11eb-0ee6-e7e0993e8fe8
# ╟─f7454b4a-9bc2-11eb-2ab9-dfc016fd57de
# ╟─1506eaf4-9b9a-11eb-170d-95834314fb84
# ╟─1eb00bb6-9bc8-11eb-3af9-c33ef5b6ab15
# ╟─28aa73fe-9bc8-11eb-3266-f725f73a3159
# ╠═9b31019a-9bc8-11eb-0372-43ea0c0d8fc3
# ╟─c9b0923c-9bce-11eb-3861-4d201b1765a8
# ╠═abc29124-9bc8-11eb-268d-c5115c341b58
# ╟─8de3bc38-9b50-11eb-2ed2-436e4a3da804
# ╠═a0c7fb86-9b50-11eb-02c6-37fc9fc78d57
# ╠═047a3e4a-9c83-11eb-142c-73e070fd4731
# ╟─aed51e0c-9b50-11eb-2b2b-0553df4ec03b
# ╟─b80ba91e-9b50-11eb-2e67-bfd1a71f069e
# ╟─677ed7ee-9c15-11eb-17c1-bf6e27df8dba
# ╠═1abf39ea-9c19-11eb-1bb0-178d4f3e7de4
# ╠═ba97522c-9c16-11eb-3734-27688484ef6f
# ╟─9c7a4006-9c19-11eb-15ac-890dce21f2ec
# ╟─e6b1374a-9c7f-11eb-1d90-1d6d40914d90
# ╟─56eeada8-9c80-11eb-1422-79f53b49b47e
# ╟─64a5c40a-9c84-11eb-22fa-ff8524822e62
# ╟─6c189d08-9c15-11eb-1d47-a902e8997cc3
# ╠═993e30d8-9c86-11eb-130a-41a1bb825e7c
# ╟─704b6176-9c15-11eb-11ba-ffb4491a6003
# ╠═fd07f1d2-9c89-11eb-023f-e96ccdd43a7e
# ╟─7489004a-9c15-11eb-201d-91f34cb40c6f
# ╠═2617153a-9ed3-11eb-35d2-fd79230055f4
# ╟─5bebcb4c-9ee2-11eb-28b3-370f4d719493
# ╟─658bd0bc-9ee6-11eb-052e-c7f8f9536633
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
