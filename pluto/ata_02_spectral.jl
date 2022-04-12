### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 3f1cfd12-7b86-11eb-1371-c5795b87ef5b
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra, 
		  PlutoUI, BenchmarkTools, ForwardDiff, Printf, SIAMFANLEquations
	include("tools.jl")
end;

# ╔═╡ 10164e58-86ab-11eb-24c7-630872bf513d
# to use the fft algorithms, import them as follows:
using FFTW

# ╔═╡ 76e9a7f6-86a6-11eb-2741-6b8759be971b
md"""
## §2 Spectral Methods 

The purpose of the second group of lectures is to study a class of numerical methods for solving differential equations, called *Fourier Spectral Methods*. For example we will learn how to solve periodic boundary value problems such as 
```math
\begin{aligned}
  - u'' + u &= f, \\ 
	u(-\pi) &= u(\pi), \\ 
    u'(-\pi) &= u'(\pi)
\end{aligned}
```
to very high accuracy. 

Before we begin we will develop a fast algorithm to evaluate the trigonometric interpolant, and more generally to convert between nodal values and fourier coefficients.
"""

# ╔═╡ 04b54946-86a7-11eb-1de0-2f61e7b8f790
md"""
## §2.1 The Fast Fourier Transform 

### The Discrete Fourier Transform

Recall from §1 that the trigonometric interpolant ``I_N f`` of a function ``f`` is given by
```math
	I_N f(x) = \sum_{k = -N+1}^{N-1} \hat{F}_k e^{i k x} + \hat{F}_N \cos(N x)
```
and the coefficients are determined by the linear system 
```math
	\sum_{k = -N+1}^N \hat{F}_k e^{i k x_j} = F_j, \qquad j = 0, \dots, 2N-1.
```
where ``F_j = f(x_j)`` and ``x_j = j \pi / N``. We have moreover shown numerically and proved this in A1 that the system matrix is orthogonal (up to rescaling), i.e., if 
```math
	A = \big( e^{i k x_j} \big)_{k,j}
```
then 
```math
	A A^H = 2N I
```
In particular ``A`` is invertible, i.e., the mapping ``F \mapsto \hat{F}, \mathbb{C}^{2N} \to \mathbb{C}^{2N}`` is invertible. 
This mapping is called the discrete fourier transform (DFT) and its inverse is called the inverse discrete fourier transform (IDFT, ``\hat{F} \mapsto F``). Both use a different scaling than we use here; specifically, the most commen definition is 
```math
\begin{aligned}
	{\rm DFT}[G]_k &= \sum_{j = 0}^{2N-1} e^{- i k j \pi / N} G_j, \\ 
	{\rm IDFT}[\hat{G}]_j &= \frac{1}{2N} \sum_{k = -N+1}^N e^{i k j \pi / N} \hat{G}_k.
\end{aligned}
```
This means the the mappings ``F \mapsto \hat{F}, \hat{F} \mapsto F`` can be written as 
```math 
	\hat{F} = (2N)^{-1} \cdot {\rm DFT}[F], \qquad F = 2N \cdot {\rm IDFT}[\hat{F}]
```
"""

# ╔═╡ d57bead8-86b3-11eb-3095-1fee9108c6b1
md"""
The cost of evaluating the DFT and IDFT naively is ``O(N^2)`` (matrix-vector multiplication) but the special structures in the DFT make it possible to evaluate them in ``O(N \log (N))`` operations. This was first observed by Gauss (1876), and much later rediscovered and popularized by [Cooley & Tukey (1965)](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm). It is generally considered one of the [most important algorithms of the 20th century](https://www.computer.org/csdl/magazine/cs/2000/01/c1022/13rRUxBJhBm). 

In Julia, the FFT is implemented in the [FFTW package](https://github.com/JuliaMath/FFTW.jl) (the Fastest Fourier Transform in the West). Before we study it, we can try it out:
"""

# ╔═╡ 7b0578ee-8744-11eb-0bf1-555f11fbb0fd
begin
	# let's also define some general utility functions
	
	# the strange (??) ordering of the k-grid is determined by 
	# the convention used for the FFT algorithms
	xgrid(N) = [ j * π / N  for j = 0:2N-1 ]
	kgrid(N) = [ 0:N; -N+1:-1 ]
	
	function dft(F)
		N = length(F) ÷ 2
		A = [ exp(im * k * x) for k in kgrid(N), x in xgrid(N) ]
		return (A' * F) / (2*N)
	end
end

# ╔═╡ 13e8fc40-86ab-11eb-1f63-9d2ed7538e7e
let N = 100
	# run a random tests to confirm FFT = DFT
	F = rand(ComplexF64, N)
	norm( dft(F) - fft(F) / N )
end

# ╔═╡ c3e57120-86ab-11eb-2268-4f7338540556
let N = 100
	# run a random test to see how fft, ifft work
	F = rand(ComplexF64, N)
	norm(F - ifft(fft(F)))
end

# ╔═╡ fc6671fa-8748-11eb-3d6b-e50f405b446f
md"Finally, let's compare the Timing of DFT vs FFT (times in seconds):"

# ╔═╡ 96d114ee-8748-11eb-05f8-a72869439a84
let NN = [5, 10, 20, 40, 80, 160]
	FF = [ rand(ComplexF64, 2*N) for N in NN ]   # random trial vectors 
	times_dft = [ @belapsed dft($F) for F in FF ]
	times_fft = [ @belapsed fft($F) for F in FF ]
	ata_table( (NN, "``N``", "%d"), 
		       (times_dft, "DFT", "%1.2e"), 
		       (times_fft, "FFT", "%1.2e"), 
	           (times_fft./times_dft, "FFT/DFT", "%1.1e"), 
			   )
end

# ╔═╡ 6da69574-86b4-11eb-3300-9b1d62ede475
md"""
What is the idea behind the FFT that gives it such a great performance? Note that the ``O(N \log(N))`` scaling is very close to the theoretically optimal complexity. There are many good references to study the FFT, and there is little point in reproducing this here. But we can at least discuss the main idea of the radix-2 FFT; see whiteboard lecture, and [LN, Sec. 3.6]. We will prove the following result: 

**Theorem:** If ``N = 2^n`` then the DFT (and the IDFT) can be evaluated with ``O(N \log(N))`` operations and ``O(N)`` storage.

With this in hand, we can now rewrite our trigonometric interpolation routines as follows. (Though sometimes we will simply use `fft` and `ifft` directly.)
"""


# ╔═╡ 3c81eca4-86b5-11eb-0e54-d53593b063bc
begin
	"""
	construct the coefficients of the trigonometric interpolant
	"""
	triginterp(f, N) = fft(f.(xgrid(N))) / (2*N)
	
	
	"""
	to evaluate a trigonometric polynomial just sum coefficients * basis
	we the take the real part because we assume the function we are 
	approximating is real.
	"""
	evaltrig(x, F̂) = sum( real(F̂ₖ * exp(im * x * k))
						  for (F̂ₖ, k) in zip(F̂, kgrid(length(F̂) ÷ 2)) )
end 

# ╔═╡ c93dbae2-86b5-11eb-1468-bd709534e1af
md"""
Approximating ``f(x) = \sin(2x) / (0.1 + \cos^2(x))``  

Choose a polynomial degree:  $(@bind _N1 Slider(5:20))
"""

# ╔═╡ a18b061c-86b5-11eb-3c44-0bc846854b1b
let f = x -> sin(2*x) / (0.1 + cos(x)^2)
	xp = range(0, 2*π, length=500)
	X = xgrid(_N1)
	F̂ = triginterp(f, _N1)
	plot(xp, f.(xp), lw=4, label = L"f", size = (500, 300))
	plot!(xp, evaltrig.(xp, Ref(F̂)), lw=2, label = L"I_N f")
	plot!(X, f.(X), lw=0, c=2, m=:o, ms=3, label = "", 
		  title = latexstring("N = $(_N1)"))
end

# ╔═╡ bc30cf3c-86b6-11eb-1f21-ff29b647a839
md"""
Approximating ``f(x) = e^{- |\sin(x)|}``

Choose a polynomial degree:  $(@bind _p2 Slider(2:10))
"""

# ╔═╡ e02f56bc-86b6-11eb-3a66-0d0b94677262
let f = x -> exp( - abs(sin(x)) )
	N2 = 2^_p2
	xp = range(0, 2*π, length=1000)
	X = xgrid(N2)
	F̂ = triginterp(f, N2)
	plot(xp, f.(xp), lw=4, label = L"f", size = (500, 300))
	plot!(xp, evaltrig.(xp, Ref(F̂)), lw=2, label = L"I_N f")
	plot!(X, f.(X), lw=0, c=2, m=:o, ms=3, label = "", 
		  title = latexstring("N = $(N2)"))
end

# ╔═╡ 240250ae-86b7-11eb-1046-7f29472897fd
md"""

## §2.2 Fourier transform of linear homogeneous differential operators

Let 
```math
	t_N(x) = \sum_k \hat{F}_k e^{i k x}
``` 
be a trigonometric polynomial, then 
```math
	t_N'(x) = \frac{d t_N(x)}{dx} = \sum_k \hat{F}_k (i k) e^{i k x}
```
We have two nice properties: 
* If ``t_N \in \mathcal{T}_N`` then ``t_N' = dt_N/dx \in \mathcal{T}_N`` as well.
* If ``t_N \in \mathcal{T}_N'`` then ``t_N'' \in \mathcal{T}_N'`` as well.
* the differentiation of ``t_N`` corresponds to multiplying the Fourier coefficients ``\hat{F}_k`` by ``i k``. 

In other words if we represent a function by its fourier coefficients then we can *exactly* represent differentiation operator ``d/dx`` by a diagonal matrix, 
```math
	\hat{F} \mapsto \hat{d} {\,.\!\!*\,} \hat{F} = \big( i k \hat{F}_k )_{k = -N+1}^N.
```
where ``{\,.\!\!*\,}`` denotes element-wise multiplication. This is an extremely convenient property when we try to discretise a differential equation and extends to general linear homogeneous differential operators: 
```math
	L := \sum_{p = 0}^P a_p \frac{d^p}{dx^p} \qquad \text{becomes} \qquad 
	\hat{L}(k) = \sum_{p = 0}^P a_p (i k)^p.
```
By which we mean that 
```math
	s_N = L f_N, \qquad \Rightarrow \qquad 
	\hat{S}_k =  \hat{L}_k \hat{F}_k.
```


There are other important operators that also become diagonal under the Fourier transform, the most prominent being the convolution operator.

"""

# ╔═╡ 5ebefefe-86b7-11eb-227f-3d5e02a142fd
md"""

## §2.3 Spectral methods for linear homogeneous problems

Let us return to the introductory example, 
```math
	- u'' + u = f, 
```
and imposing periodic boundary conditions. We now perform the following steps: 

* Approximate ``u`` by a trigonometric polynomial ``u_N \in \mathcal{T}_N'``. 
* Approximate ``f`` by a trigonometric polynomial ``f_N \in \mathcal{T}_N'``. 
In real space the equation becomes ``- u_N'' + u_N = f_N``, and expanded 
```math
	\sum_{k = -N+1}^N \hat{U}_k \big[ - (i k)^2 + 1 \big] e^{i kx}
	= \sum_{k = -N+1}^N \hat{F}_k e^{i kx}
```
* Equating coefficients and noting that ``-(ik)^2 = k^2`` we obtain 
```math
	(1 + k^2) \hat{U}_k = \hat{F}_k
```
or, equivalently, 
```math
	\hat{U}_k = (1+k^2)^{-1} \hat{F}_k.
```
This is readily implemented in a short script.
"""

# ╔═╡ 452c65b2-8806-11eb-2d7a-3f4312071cd1
md"""
Polynomial degree:  $(@bind _N2 Slider(5:20, show_value=true))

Right-hand side ``f(x) = ``: $(@bind _fstr2 TextField())
"""

# ╔═╡ 503b45d0-8e65-11eb-0e77-15314d82de1a
_ffun2 = ( _fstr2 == "" ? x -> abs(exp(sin(x) + 0.5 * sin(x)^2)) 
				        : Meta.eval(Meta.parse("x -> " * _fstr2)) );

# ╔═╡ f3c1ba14-8e64-11eb-33ea-4341480e50b3
_Ûex2 = let N = 100, f = _ffun2
		F̂ = triginterp(f, N)
		K = kgrid(N) 
		F̂ ./ (1 .+ K.^2)
	end ;

# ╔═╡ b5359ee2-86de-11eb-1446-b10b9815f448
let N = _N2, f = _ffun2
	F̂ = triginterp(f, N)
	K = kgrid(N) 
	Û = F̂ ./ (1 .+ K.^2)
	xp = range(0, 2π, length=200)
	plot(xp, evaltrig.(xp, Ref(_Ûex2)), lw=4, label = L"u", size = (400, 300), 
		title = L"N = %$N", xlabel = L"x")
	plot!(xp, evaltrig.(xp, Ref(Û)), lw=3, label = L"u_N", size = (400, 300))				
	plot!(xgrid(N), evaltrig.(xgrid(N), Ref(Û)), lw=0, ms=3, m=:o, c=2, label = "")
end 

# ╔═╡ 0c84dcde-86e0-11eb-1877-932742501593
md"""
### Convergence of spectral methods 

What can we say about the convergence of the method? Let us start with a very simple argument, which we will then generalise. The key observation is that our approximate solution ``u_N`` satisfies the full DE but with a perturbed right-hand side ``f \approx f_N``. 
```math
\begin{aligned}
	- u'' + u &= f, \\ 
   - u_N'' + u_N &= f_N.
\end{aligned}
```
Because the differential operator is linear, we can subtract the two lines and obtain  the *error equation*
```math 
   -e_N'' + e_N = f - f_N, 
```
where ``e_N = u - u_N`` is the error. At this point, we have several options how to proceed, but since so far we have studied approximation in the max-norm we can stick with that. We have the following result: 

**Lemma 2.3.1:** If ``- u'' + u = f`` with ``f \in C_{\rm per}`` then 
```math
	\|u \|_\infty \leq C \|f \|_\infty,
```
where ``C`` is independent of ``f, u``.

**Proof:** via maximum principle or Fourier analysis. Note the result is far from sharp, but it is enough for our purposes. Via Fourier analysis you would in fact easily get a much stronger result such as ``\|u\|_\infty \leq C \|f\|_{H^{-1}}`` and even that can still be improved.

Applying Lemma 2.3.1 to the error equation we obtain 
```math
	\| e_N \|_\infty \leq C \|f - f_N \|_\infty.
```
For example, if ``f`` is analytic, then we know that 
```math 
	\|f - f_N \|_\infty \leq M_f e^{-\alpha N}
```
for some ``\alpha > 0``, and hence we will also obtain 
```math 
	\| u - u_N \|_\infty \leq C M_f e^{-\alpha N}. 
```
That is, we have proven that our spectral method converges exponentially fast:

**Theorem 2.3.2:** If ``f`` is analytic then there exist ``C, \alpha > 0`` such that 
```math
	\|u - u_N\|_\infty \leq C e^{-\alpha N}.
```
"""

# ╔═╡ bb33932c-8769-11eb-0fb7-a39703fa96cc
md"""
We can test this numerically using the *method of manufactured solutions*. We start from a solution ``u(x)`` and compute the corresponding right-hand side ``f(x) = - u''(x) + u(x)``. Then we solve the BVP for that right-hand side for increasing degree ``N`` and observe the convergence rate.

Here we choose 
```math
	u(x) = \cos\big( (0.2 + \sin^2 x)^{-1} \big)
```
"""

# ╔═╡ a0e33748-8769-11eb-26b4-416a32282bc2
let NN = 10:10:120, u = x -> cos( 1 / (0.2 + sin(x)^2) )
	du = x -> ForwardDiff.derivative(u, x)
	d2u = x -> ForwardDiff.derivative(du, x)
	f = x -> u(x) - d2u(x)
	xerr = range(0, 2π, length = 1_000)
	solve(N) = triginterp(f, N) ./ (1 .+ kgrid(N).^2)
	error(N) = norm(u.(xerr) - evaltrig.(xerr, Ref(solve(N))), Inf)
	errs = error.(NN) 
	plot(NN, errs, yscale = :log10, lw = 3, m=:o, ms=4, label = "error", 
				  size = (400, 300), xlabel = L"N", ylabel = L"\Vert u - u_N \Vert_\infty")
	plot!(NN[5:9], 1_000*exp.( - 0.33 * NN[5:9]), lw=2, ls=:dash, c=:black, label = "rate")
	hline!([1e-15], lw=2, c=:red, label = L"\epsilon")
end

# ╔═╡ 9a6facbe-876b-11eb-060a-7b7e717237be
md"""
Try reaching machine precision with a finite difference or finite element method!
"""

# ╔═╡ f63dcd36-8ac8-11eb-3831-57f0a5088c98
md"""
#### An a priori approach ...

How can we determine a good discretisation parameter *a priori*? Suppose, e.g., that we wish to achieve 10 digits of accuracy for our solution. We know from the arguments above that ``\|u - u_N\|_\infty \leq C \|f - f_N \|_\infty``. We don't know the constant, but for sufficiently simple problems we can legitimately hope that it is ``O(1)``. Hence, we could simply check the convergence of the trigonometric interpolant:
"""

# ╔═╡ 33f2ab24-8ac9-11eb-220e-e71d3ebb93fa
let NN = 40:20:140, u = x -> cos( 1 / (0.2 + sin(x)^2) )
	du = x -> ForwardDiff.derivative(u, x)
	d2u = x -> ForwardDiff.derivative(du, x)
	f = x -> u(x) - d2u(x)
	err = Float64[] 
	xerr = range(0, 2π, length = 1_234)
	for N in NN 
		F̂ = triginterp(f, N) 
		push!(err, norm(f.(xerr) - evaltrig.(xerr, Ref(F̂)), Inf))
	end
	ata_table( (NN, L"N"), (err, L"\|f - f_N\|") )
end

# ╔═╡ a0797ece-8ac9-11eb-2922-e3f0295d4787
md"""
The error stagnates, suggesting that we have reached machine precision at around ``N = 120``. And we have likely achieved an accuracy of ``10^{-10}`` for around ``N = 90``. The error plot shows that in fact we have reached ``10^{-10}`` accuracy already for ``N = 70``, but this is ok. We are only after rough guidance here.

An alternative approach is to look at the decay of the Fourier coefficients. By plotting their magnitude we can get a sense at what degree to truncate. Of course we cannot compute the exact Fourier series coefficients, but the coefficients of the trigonometric interpolant closely approximate them (cf Assignment 1).
"""

# ╔═╡ 057c7508-8aca-11eb-0718-21826e314bc7
let N = 150, u = x -> cos( 1 / (0.2 + sin(x)^2) )
	du = x -> ForwardDiff.derivative(u, x)
	d2u = x -> ForwardDiff.derivative(du, x)
	f = x -> u(x) - d2u(x)
	F̂ = triginterp(f, N) 
	K = kgrid(N)
	plot(abs.(K), abs.(F̂), lw=0, ms=2, m=:o, label ="", 
		 xlabel = L"|k|", ylabel = L"|\hat{F}_k|", 
		 yscale = :log10, size = (350, 200))
	hline!([1e-10], lw=2, c=:red, label = "")
end 


# ╔═╡ 59485024-8aca-11eb-2e65-3962c096e9df
md"""
Entirely consistent with our previous results we observe that 
the Fourier coefficients drop below a value of ``10^{-10}`` 
just below ``|k| = 100``. This gives us a second strong indicator 
that for ``N \geq 100`` we will obtain the desired accuracy of 
``10^{-10}``.
"""

# ╔═╡ c9a6c2ce-876b-11eb-182c-d90997ea2cab
md"""
### General Case 
More generally consider a linear operator equation (e.g. differential or integral equation)
```math
	L u = f 
```
which we discretise as 
```math
	L u_N = f_N
```
and where we assume that it transforms under the DFT as
```math
	\hat{L}_k \hat{U}_k = \hat{F}_k
```
where ``u_N = {\rm Re} \sum_k \hat{U}_k e^{i k x}`` and ``f_N = {\rm Re} \sum_k \hat{F}_k e^{i  x}``.

Now we make two closely related assumptions (2. implies 1.): 
1. ``\hat{L}_k \neq 0`` for all ``k``
2. ``L`` is max-norm stable : ``\| u \|_\infty \leq C \| f \|_\infty``

From 1. we obtain that ``\hat{U}`` and hence ``u_N`` are well-defined. From 2. we obtain 
```math
	\| u - u_N \|_\infty \leq C \| f - f_N \|_\infty
```
and the rate of approximation of the right-hand side will determine the rate of approximation of the solution. We can explore more cases and examples in the assignment.
"""


# ╔═╡ 5b3e4e06-8e6e-11eb-0f31-5546b0ae450a
md"""

### Summary Spectral Methods / Perspective

Numerical Analysis and Scientific Computing for (P)DEs : 
* regularity theory: how smooth are the solutions of the DE?
* approximation theory: how well can we approximate the solution in principle?
* discretisation, error analysis: how do we discretize the DE to guarantee convergence of the discretized solution? Optimal rates as predicted by approximation theory?
* Fast algorithms: scaling of computational cost matters! e.g. FFT provides close to optimal computational complexity, in general this is difficult to achieve. 
"""

# ╔═╡ 6deea4c4-86b7-11eb-0fe7-5d6d0f3007ef
md"""

## §2.4 Spectral methods for time-dependent, inhomogeneous and nonlinear problems

In the following we will implement a few examples that go beyond the basic theory above and showcase a few more directions in which one could explore spectral methods. We will see a few techniques to treat cases for which spectral methods are more difficult to use, namely for differential operators with inhomogeneous coefficients and for nonlinear problems.
"""

# ╔═╡ 7620d684-88fe-11eb-212d-efcbc1803608
md"""
### Wave equation 
```math
	u_{tt} = u_{xx}
```
We first discretise in space, 
```math 
	u_{tt} = u_{N, xx},
```
then transform to Fourier coefficients, 
```math
	\frac{d^2\hat{U}_k}{d t^2}  = - k^2 \hat{U}_k,
```
and finally discretize in time
```math 
	\frac{\hat{U}_k^{n+1} - 2 \hat{U}_k^n + \hat{U}_k^{n-1}}{\Delta t^2}
	= - k^2 \hat{U}_k^n
```
"""

# ╔═╡ 42513412-88fb-11eb-2591-c90c16e91d6e
let N = 20, dt = 0.2 / N, Tfinal = 30.0, u0 = x -> exp(-10*(1 + cos(x)))
	xp = xgrid(200)
	k = kgrid(N)
	Û0 = triginterp(u0, N)
	Û1 = Û0  # zero initial velocity 
	@gif for n = 1:ceil(Int, Tfinal / dt)
    	Û0, Û1 = Û1, 2 * Û1 - Û0 - dt^2 * k.^2 .* Û1 
	    plot(xp, evaltrig.(xp, Ref(Û1)), lw = 3, label = "", size = (400, 300), 
			 xlims = [0, 2*π], ylims = [-0.1, 1.1] )
	end every 5			
end

# ╔═╡ ae55b0e4-88ff-11eb-36f0-152089e43c93
md"""

### Inhomogeneous Transport Equation

```math
	u_t + c(x) u_x = 0
```
First discretise in time using the Leapfrog scheme 
```math
	\frac{u^{n+1} - u^{n-1}}{2 \Delta t} + c (u^n)_x = 0.
```
Now we discretise both ``c`` and ``u^n`` using a trigonometric polynomials, ``c \approx c_N`` and ``u^n \approx u^n_N \in \mathcal{T}_N'``. We can easily apply ``d/dx`` in the Fourier domain, ``\hat{U}_k^n \to (i k) \hat{U}_k^n``, but what can we do with the product ``c_N (u^n_N)_x``? The trick is to differentiate in the Fourier domain, but apply the product in real space, i.e., 
* Apply ``d/dx`` in Fourier space
* Convert back to real space
* apply pointwise multiplication at interpolation nodes
"""

# ╔═╡ 00f2a760-8907-11eb-3ed1-376e9bf97fa8
let N = 256,  dt = π/(4N), tmax = 16.0, 
				cfun = x -> 0.2 + sin(x - 1)^2, 
				  u0 = x ->  exp(-100*(x-1)^2)
	X = xgrid(N)
	K = kgrid(N)
	t = 0.0
	# differentiation operator in Fourier space 
	D̂ = im*K
	
	# transport coefficient in real space 
	C = cfun.(X)
	# initial condition, we also need one additional v in the past
	# (this takes one step of the PDE backward in time)
	V = u0.(X)
	Vold = V + dt * C .* real.( ifft( D̂ .* fft(V) ) ) 
	
	function plot_soln(t, X, v, c)
		P = plot( xaxis = ([0, 2*π], ), yaxis = ([0.0, 1.5],) )
		plot!(X, 0.5*c, lw=1, c=:black, label = L"c/2")
		plot!(X, v, lw=3, label = L"v", size = (500, 300))
		return P
	end
	
	# time-stepping loop
	@gif for t = 0:dt:tmax
		# differentiation in reciprocal space
		W = real.( ifft( D̂ .* fft(V) ) )   
		# multiplication and update in real space
		V, Vold = Vold - 2 * dt * C .* W, V
		plot_soln(t, X, V, C)
	end every 20
end

# ╔═╡ babc0fae-88ff-11eb-0516-6b7841fc0a6a
md"""

### Nonlinear BVP

Steady state viscous Burgers equation
```math
		u u_x = \epsilon u_{xx} - 0.1 \sin(x)
```
We write a nonlinear system 
```math
	F_j := u_N(x_j) u_{N,x}(x_j) - \epsilon u_{N,xx}(x_j) + 0.1 sin(x)
```
and use a generic nolinear solver to solve
```math
	F_j = 0, \qquad j = 0, \dots, 2N-1.
```
This is not a magic bullet, often one needs specialized tools to solve these resulting nonlinear systems.
"""


# ╔═╡ 7d7d43e2-8a71-11eb-2031-f76c30b64f5e
# using SIAMFANLEquations  # we use this package to solve nonlinear systems

# ╔═╡ f2dab6a0-890a-11eb-1e48-a747d18f6c93
let N = 30, ϵ = 0.1
	function burger(U)
		N = length(U) ÷ 2 
		k = kgrid(N)
		Û = fft(U) 
		F = sin.(xgrid(N))
		return real.(U .* ifft( im * k .* Û ) + ϵ * ifft( k.^2 .* Û ) .+ 0.1*F)
	end
	U0 = sin.(xgrid(N))
	U = nsoli(burger, U0, maxit = 1_000)
	Û = fft(U) / (2N)
	plot(x -> evaltrig(x, Û), -π, π, lw=3, size = (400, 150), 
	     label = "Residual = " * @sprintf("%.2e\n", norm(burger(U), Inf)), 
		 xlabel = L"x", ylabel = L"u(x)")
end 

# ╔═╡ 7b5b1e6c-86b7-11eb-2b32-610393a24da4
md"""

## §2.5 Outlook: PDEs in higher dimension

Just one example; more in §4 of this course.

### 2D Wave equation 

```math
	u_{tt} = u_{xx} + u_{yy}
```
Discrete fourier transform for ``u(x, y)`` becomes ``\hat{U}_{k_1 k_2}``. 
After discretising in time and space, and transforming to the Fourier domain, 
```math
	\frac{\hat{U}_{k_1k_2}^{n+1} - 2 \hat{U}_{k_1k_2}^n + \hat{U}^{n-1}_{k_1 k_2}}{\Delta t^2} 
	=  -(k_1^2 + k_2^2) \hat{U}_{k_1 k_2}.
```
"""

# ╔═╡ b90093f6-8906-11eb-2e69-6d4b807866a4
let N = 64, u0 = (x, y) -> exp(-10*(1 + cos(x))) * exp.(-10*(1 + cos(y)))
	x = xgrid(N); Xx = kron(x, ones(2*N)'); Xy = Xx'
	k = kgrid(N); Kx = kron(k, ones(2*N)'); Ky = Kx'
	U0 = u0.(Xx, Xy)
	Û0 = fft(U0)
	Û1 = Û0  # zero initial velocity 
	dt = 0.2 / N 
	@gif for n = 1:4_000
		Û0, Û1 = Û1, 2 * Û1 - Û0 - dt^2 * (Kx.^2 + Ky.^2) .* Û1 
		Plots.surface(x, x, real.(ifft(Û1)), zlims = [-0.3, 0.3], color=:viridis, 
					  size = (400, 300))
	end every 5
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"

[compat]
BenchmarkTools = "~1.3.0"
DataFrames = "~1.3.2"
FFTW = "~1.4.5"
ForwardDiff = "~0.10.25"
LaTeXStrings = "~1.3.0"
Plots = "~1.25.8"
PlutoUI = "~0.7.34"
PrettyTables = "~1.3.1"
SIAMFANLEquations = "~0.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
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

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e1ba79094cae97b688fb42d31cbbfd63a69706e4"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.7.8"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "ce68f8c2162062733f9b4c9e3700d5efc4a8ec47"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "0.16.11"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "be0cff14ad0059c1da5a017d66f763e6a637de6a"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.0"

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
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

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
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

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
git-tree-sha1 = "84083a5136b6abf426174a58325ffd159dd6d94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.1"

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
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

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
git-tree-sha1 = "4a740db447aae0fbeb3ee730de1afbb14ac798a1"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.63.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

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
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

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
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

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
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

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
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

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
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

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
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "eb1432ec2b781f70ce2126c277d120554605669a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.8"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8979e9802b4ac3d58c503a20f2824ad67f9074dd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.34"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

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
git-tree-sha1 = "37c1631cb3cc36a535105e6d5557864c82cd8c2b"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.0"

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

[[deps.SIAMFANLEquations]]
deps = ["AbstractFFTs", "BandedMatrices", "FFTW", "LaTeXStrings", "LinearAlgebra", "Printf", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "f2a95ba91c2b4fb67858762cb9db976f372c7eb8"
uuid = "084e46ad-d928-497d-ad5e-07fa361a48c4"
version = "0.4.3"

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
git-tree-sha1 = "8d0c8e3d0ff211d9ff4a0c2307d876c99d10bdf1"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

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
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

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
# ╟─04b54946-86a7-11eb-1de0-2f61e7b8f790
# ╟─d57bead8-86b3-11eb-3095-1fee9108c6b1
# ╠═10164e58-86ab-11eb-24c7-630872bf513d
# ╠═7b0578ee-8744-11eb-0bf1-555f11fbb0fd
# ╠═13e8fc40-86ab-11eb-1f63-9d2ed7538e7e
# ╠═c3e57120-86ab-11eb-2268-4f7338540556
# ╟─fc6671fa-8748-11eb-3d6b-e50f405b446f
# ╟─96d114ee-8748-11eb-05f8-a72869439a84
# ╟─6da69574-86b4-11eb-3300-9b1d62ede475
# ╠═3c81eca4-86b5-11eb-0e54-d53593b063bc
# ╟─c93dbae2-86b5-11eb-1468-bd709534e1af
# ╟─a18b061c-86b5-11eb-3c44-0bc846854b1b
# ╟─bc30cf3c-86b6-11eb-1f21-ff29b647a839
# ╟─e02f56bc-86b6-11eb-3a66-0d0b94677262
# ╟─240250ae-86b7-11eb-1046-7f29472897fd
# ╟─5ebefefe-86b7-11eb-227f-3d5e02a142fd
# ╟─452c65b2-8806-11eb-2d7a-3f4312071cd1
# ╟─503b45d0-8e65-11eb-0e77-15314d82de1a
# ╟─f3c1ba14-8e64-11eb-33ea-4341480e50b3
# ╠═b5359ee2-86de-11eb-1446-b10b9815f448
# ╟─0c84dcde-86e0-11eb-1877-932742501593
# ╟─bb33932c-8769-11eb-0fb7-a39703fa96cc
# ╠═a0e33748-8769-11eb-26b4-416a32282bc2
# ╟─9a6facbe-876b-11eb-060a-7b7e717237be
# ╟─f63dcd36-8ac8-11eb-3831-57f0a5088c98
# ╟─33f2ab24-8ac9-11eb-220e-e71d3ebb93fa
# ╟─a0797ece-8ac9-11eb-2922-e3f0295d4787
# ╟─057c7508-8aca-11eb-0718-21826e314bc7
# ╟─59485024-8aca-11eb-2e65-3962c096e9df
# ╟─c9a6c2ce-876b-11eb-182c-d90997ea2cab
# ╟─5b3e4e06-8e6e-11eb-0f31-5546b0ae450a
# ╟─6deea4c4-86b7-11eb-0fe7-5d6d0f3007ef
# ╟─7620d684-88fe-11eb-212d-efcbc1803608
# ╠═42513412-88fb-11eb-2591-c90c16e91d6e
# ╟─ae55b0e4-88ff-11eb-36f0-152089e43c93
# ╠═00f2a760-8907-11eb-3ed1-376e9bf97fa8
# ╟─babc0fae-88ff-11eb-0516-6b7841fc0a6a
# ╠═7d7d43e2-8a71-11eb-2031-f76c30b64f5e
# ╠═f2dab6a0-890a-11eb-1e48-a747d18f6c93
# ╟─7b5b1e6c-86b7-11eb-2b32-610393a24da4
# ╠═b90093f6-8906-11eb-2e69-6d4b807866a4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
