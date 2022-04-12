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

# ╔═╡ 66772e00-9981-11eb-1941-d9132b99c780
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
          PlutoUI, BenchmarkTools, ForwardDiff, Printf, Random, FFTW

# ╔═╡ 465e7582-95c7-11eb-0c7b-ed7dddc24b4f
begin
	function ingredients(path::String)
		# this is from the Julia source code (evalfile in base/loading.jl)
		# but with the modification that it returns the module instead of the last object
		name = Symbol(basename(path))
		m = Module(name)
		Core.eval(m,
			Expr(:toplevel,
				 :(eval(x) = $(Expr(:core, :eval))($name, x)),
				 :(include(x) = $(Expr(:top, :include))($name, x)),
				 :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
				 :(include($path))))
		m
	end;
	
	M = ingredients("tools.jl")
	IRLSQ = M.IRLSQ
	
	xgrid(N) = [ j * π / N  for j = 0:2N-1 ]
	kgrid(N) = [ 0:N; -N+1:-1 ]
	triginterp(f, N) = fft(f.(xgrid(N))) / (2*N)
	evaltrig(x, F̂) = sum( real(F̂ₖ * exp(im * x * k))
						  for (F̂ₖ, k) in zip(F̂, kgrid(length(F̂) ÷ 2)) )
	function trigerr(f, N; xerr = range(0, 2π, length=13*N)) 
		F̂ = triginterp(f, N) 
		return norm( evaltrig.(xerr, Ref(F̂)) - f.(xerr), Inf )
	end
end;

# ╔═╡ 58d79c34-95c7-11eb-0679-eb9741761c10
md"""
## §4 Miscellaneous 

In this section of the mini-course we will cover a small number of topics, without going too deeply into any of them, simply to get some exposure to some important and fun ideas:
* Chebyshev polynomials and Chebyshev points 
* Rational Approximation 
* Max-norm approximation with iteratively reweighted least squares
"""

# ╔═╡ 9c169770-95c7-11eb-125a-4f174da56d36
md"""
## §4.1 Approximation with Algebraic Polynomials

The witch of Agnesi: 
```math
	f(x) = \frac{1}{1+ 25 x^2}, \qquad x \in [-1, 1]. 
```
"""

# ╔═╡ 05ec7c1e-95c8-11eb-176a-3372a765d4d7
begin
	agnesi(x) = 1 / (1 + 25 * x^2)
	plot(agnesi, -1, 1, lw=3, label = "", 
		 size = (350, 200), title = "Witch of Agnesi", 
		 xlabel = L"x", ylabel = L"f(x)")
end

# ╔═╡ 311dd26e-95ca-11eb-2ff5-03a53a662a62
md"""
We can rescale ``f`` and repeat it periodically and then use trigonometric polynomials to approximate it. Because the periodic extension is only ``C^{0,1}``, i.e., Lipschitz, but no more we only get a rate of ``N^{-1}``. 
"""

# ╔═╡ 4428c460-95c8-11eb-16fc-6327acc80bb6
let f = x -> agnesi((x - π)/π), NN = (2).^(2:10)
	plot(NN, trigerr.(Ref(f), NN), lw=2, m=:o, ms=4,
		 size = (350, 250), xscale = :log10, yscale = :log10, 
		 label = "error", title = "Approximation of Agnesi", 
		 xlabel = L"N", ylabel = L"\Vert f- I_N f \Vert_\infty")
	plot!(NN[5:end], 3e-2*NN[5:end].^(-1), lw=2, ls=:dash, c=:black, 
		  label = L"\sim N^{-1}" )
end

# ╔═╡ 3533717c-9630-11eb-3fc6-3f4b29eaa63a
md"""
And no surprise - the periodic repetition of ``f`` is only Lipschitz:
"""

# ╔═╡ 467c7f76-9630-11eb-1cdb-6dedae3c459f
let f = x -> agnesi((mod(x+1, 2)-1)/2)
	plot(f, -1, 4, lw=2, label = "", 
		 size = (400, 200), xlabel = L"x", ylabel = L"f(x)")
end

# ╔═╡ 6cbcac96-95ca-11eb-05f9-c7a781b38061
md"""
But it is a little ridiculous that we get such a poor rate: After all ``f`` is an analytic function on it domain of definition ``[-1, 1]``. Can we replicate the excellent approximation properties that trigonometric polynomials have for periodic analytic functions?

Our first idea is to use algebraic polynomials 
```math
p(x) = \sum_{n = 0}^N c_n x^n
```
such that ``p \approx f`` in ``[-1,1]``. Analogously as for trigonometric polynomials, we could try to determine the coefficients via interpolation, 
```math
	p(x_n) = f(x_n),  \qquad x_n = -1 + 2n/N, \qquad n = 0, \dots, N. 
```
here with equispaced nodes. 

This is a bad idea: 
"""

# ╔═╡ 77dcea8e-95ca-11eb-2cf1-ad342f0f6d7d
# Implementation of Runge example 
let f = x -> 1/(1+25*x^2), NN1 = [5, 8, 10], NN2 =  5:5:30
	function poly_fit(N)
		# this is numerically unstable - do not do this!!! 
		# we will learn in the next lecture how to do stable numerical interpolation
		A = [   (-1 + 2*m/N)^n  for m = 0:N, n = 0:N ]
		F = [ f((-1 + 2*m/N)) for m = 0:N ]
		return A \ F 
	end
	# do not do this either, it is neither efficient nor stable!
	poly_eval(x, c) = sum( c[n] * x^(n-1) for n = 1:length(c) )
	
	# first plot 
	xp = range(-1, 1, length=300)
	P1 = plot(xp, f.(xp); lw=4, label = "exact",
			  size = (400, 400), xlabel = L"x")
	for (iN, N) in enumerate(NN1)
		xi = [(-1 + 2*m/N) for m = 0:N]
		c = poly_fit(N)
		plot!(P1, xp, poly_eval.(xp, Ref(c)), c = iN+1, lw=2,label = L"p_{%$(N)}")
		plot!(P1, xi, f.(xi), lw=0, c = iN+1, m = :o, ms=3, label = "")
	end 
	
	# second plot 
	xerr = range(-1, 1, length=3_000)
	err = [ norm( f.(xerr) - poly_eval.(xerr, Ref(poly_fit(N))), Inf )
			for N in NN2 ]
	P2 = plot(NN2, err, lw = 3, label = L"\Vert f - I_N f \Vert", 
			  yscale = :log10, xlabel = L"N", legend = :topleft)
	plot(P1, P2, size = (600, 300), title = "Witch of Agnesi")
end

# ╔═╡ 27f7e2c8-95cb-11eb-28ca-dfe90be89670
md"""
## The Chebyshev Idea

The idea is to lift ``f`` to the complex unit circle: 

```math 
	g(\theta) := g(\cos\theta)
```
"""

# ╔═╡ cc8fe0f4-95cd-11eb-04ec-d3fc8cced91b
let
	tt = range(0, 2π, length=300)
	P1 = plot(cos.(tt), sin.(tt), lw=3, label = "",  
		 size = (250, 250), xlabel = L"x = {\rm Re} z", ylabel = L"{\rm Im} z")
	plot!([-1,1], [0,0], lw=3, label = "", legend = :bottomright)
	for x in -1:0.2:1
		plot!([x,x], [sqrt(1-x^2), -sqrt(1-x^2)], c=2, lw=1, label = "")

	end
	P1
end

# ╔═╡ dfc0866c-95cb-11eb-1869-7bd67468c233
let
	tt = range(0, 2π, length=300)
	xx = range(-1, 1, length=200)
	P2 = plot(cos.(tt), sin.(tt), 0*tt, c = 1, lw = 3, label = "")
	# for x in -1:0.1:1
	# 	f = agnesi(x)
	# 	plot!([x,x], [sqrt(1-x^2), -sqrt(1-x^2)], [f, f], 
	# 		  c=:grey, lw=1, label = "")
	# end 
	plot!([-1, 1], [0, 0], [0, 0], c = 2, lw = 3, label = "")
	plot!(xx, 0*xx, agnesi.(xx), c = 2, lw = 2, label = "Agnesi")
	plot!(cos.(tt), sin.(tt), agnesi.(cos.(tt)), c=3, lw=3, label= "Lifted", 
		  size = (400, 300))
end 

# ╔═╡ f3cd1326-95cd-11eb-192a-efc3371d17b6
md"""
For our Agnesi example we obtain 
```math
g(\theta) = \frac{1}{1+ 25 \cos^2(\theta)}
```
which we already know to be analytic. This is a general principle. The regularity of ``f`` transforms into the same regularity for ``g`` but we gain periodicity. We will state some general results below. For now, let us continue to investigate:

Since ``g`` is periodic analytic we can apply a trigonometric approximation: 
```math
	g(\theta) \approx \sum_{k = -N}^N \hat{g}_k e^{i k \theta} 
```
and we know that we obtain an exponential rate of convergence. 
"""

# ╔═╡ 5181b8f4-95cf-11eb-30fe-cd93f4ed0012
md"""
### Chebyshev Polynomials

We could simply use this to construct approximations for ``f(x) = g(\cos^{-1}(x))``, in fact this is one of the most efficient and numericall stable ways to construct chebyshev polynomials, but it is very instructive to transform this approximation back to ``x`` coordinates. To that end we first note that 
```math
	g(-\theta) = f(\cos(-\theta)) = f(\cos\theta) = g(\theta)
```
and moreover, ``g(\theta) \in \mathbb{R}``. These two facts allow us to show that ``\hat{g}_{-k} = \hat{g}_k`` and moreover that $\hat{g}_k \in \mathbb{R}$. Thus, we obtain ,
```math
	f(x) = g(\theta) = \hat{g}_0 + \sum_{k = 1}^N \hat{g}_k (e^{i k \theta} + e^{-i k \theta})
	= \hat{g}_0 + \sum_{k = 1}^N \hat{g}_k \cos(k \theta).
```
Now comes maybe the most striking observation: if we define basis functions ``T_k(x)`` via the identify 
```math
	T_k( \cos \theta ) = \cos( k \theta )
```
so that we can write 
```math
	f(x) = \sum_{k = 0}^N \tilde{f}_k T_k(x)
```
where $\tilde{f}_0 = \hat{g}_0$ and $\tilde{f}_k = 2 \hat{g}_k$ of $k \geq 1$, then we have the following result: 

**Lemma:** The function ``T_k(x)`` is a polynomial of degree ``k``. In particular, they form a basis of the space of polynomials, called the *Chebyshev Basis*. An equivalent definition is the 3-term recursion 
```math
\begin{aligned}
	T_0(x) &= 1,   \\
	T_1(x) &= x,   \\ 
	T_k(x) &= 2 x T_{k-1}(x) + T_{k-2}(x) \quad \text{for } k = 2, 3, \dots.
\end{aligned}
```
**Proof:** see [LN, Lemma 4.2]
"""


# ╔═╡ 7661921a-962a-11eb-04c0-5b7e486aea05
md"""
### Chebyshev Interpolant 

Next, we transform the trigonometric interpolant ``I_N g(\theta)`` to ``x`` coordinates. We already know now that it will be a real algebraic polynomial and we can derive a more derict way to define it: namely, the interpolation nodes are given by 
```math
	x_j = \cos( \theta_j ) = \cos( \pi j / N ), \qquad j = 0, \dots, 2N-1
```
but because of the reflection symmetry of ``\cos`` they are repeated, i.e., ``x_j = x_{-j}`` and in fact we only need to keep 
```math 
	x_j = \cos( \pi j / N ), \qquad j = 0, \dots, N.
```
which are called the *Chebyshev nodes* or *Chebyshev points*. Thus, trigonometric interpolation of ``g(\theta)`` corresponds to polynomial interpolation of ``f(x)`` at those points: 
```math
	I_N f(x_j) = f(x_j), \qquad j = 0, \dots, N.
```
"""

# ╔═╡ 4ca65bec-962b-11eb-3aa2-d5500fd24677
md"""
If we write ``I_N f(x) = p_N(x) = \sum_{n = 0}^N c_n T_n(x)`` then we obtain an ``(N+1) \times (N+1)`` linear system for the coefficients ``\boldsymbol{c}=  (c_n)_{n = 0}^N``, 
```math
	\sum_{n = 0}^N c_n T_n(x_j) = f(x_j), \qquad j = 0, \dots, N
```
The resulting polynomial ``I_N f`` is called the *Chebyshev interpolant*. A naive implementation can be performed as follows:
"""

# ╔═╡ 69dfa184-962b-11eb-0ed2-c17aa6c15a45
begin 
	"""
	reverse the nodes so they go from -1 to 1, i.e. left to right
	"""
	chebnodes(N) = [ cos( π * n / N ) for n = N:-1:0 ]

	function chebbasis(x::T, N) where {T}
		B = zeros(T, N+1)
		B[1] = one(T)
		B[2] = x
		for k = 2:N
			B[k+1] = 2 * x * B[k] - B[k-1]
		end
		return B
	end
	
	
	"""
	Naive implementation of chebyshev interpolation. This works fine for 
	basic experiments, but to scale to larger problems, use the FFT!!!
	See `chebinterp` below!
	"""
	function chebinterp_naive(f, N)	
		X = chebnodes(N) 
		F = f.(X) 
		A = zeros(N+1, N+1)
		for (ix, x) in enumerate(X)
			A[ix, :] = chebbasis(x, N)
		end
		return A \ F
	end 
	
	
	"""
	Fast and stable implementation based on the FFT. This uses 
	the connection between Chebyshev and trigonometric interpolation.
	But this transform needs the reverse chebyshev nodes.
	"""
	chebinterp(f, N) = fct(f.(reverse(chebnodes(N))))
	
	function fct(A::AbstractVector)
		N = length(A)
		F = ifft([A[1:N]; A[N-1:-1:2]])
	   return [[F[1]]; 2*F[2:(N-1)]; [F[N]]]
	end
	
	
	"""
	Evaluate a polynomial with coefficients F̃ in the Chebyshev basis
	"""
	function chebeval(x, F̃) 
		T0 = one(x); T1 = x 
		p = F̃[1] * T0 + F̃[2] * T1 
		for n = 3:length(F̃)
			T0, T1 = T1, 2*x*T1 - T0 
			p += F̃[n] * T1 
		end 
		return p 
	end 
end

# ╔═╡ 3ef9e58c-962c-11eb-0172-a700d2f7e72c
let f = agnesi, NN = 6:6:60
	xerr = range(-1, 1, length=2013)
	err(N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp(f, N))), Inf)
	plot(NN, err.(NN), lw=2, m=:o, ms=4, label = "error", 
		 size = (300, 250), xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10,
		 title = "Chebyshev Interpolant")
	plot!(NN[6:end], 2*exp.( - 0.2 * NN[6:end]), lw=2, ls=:dash, c=:black, 
		  label = L"\exp( - N/5 )")
end

# ╔═╡ 19509e38-962d-11eb-1f5c-811188a57261
md"""
See [LN, Sec. 4.1, 4.2] and whiteboard summary to understand this rate fully. Concepts to understand include 
* Chebyshev series: essentially the Fourier series for ``g(\theta)``
* Chebyshev coefficients: essentially the Fourier coefficients of ``g(\theta)``
* Jukousky map: the transformation between ``[-1,1]`` and the unit circle exteneded into the complex plane
* Bernstein ellipse: the tranformation of the strip ``\Omega_\alpha`` for ``g``

For the sake of completeness, we briefly state the two main approximation theorems which can be obtained fairly easily (though it would take some time) from the ideas we developed for far:

**Theorem [Jackson for Algebraic Polynomials]** Let ``f \in C^r([-1,1])`` and suppose that ``f^{(r)}`` has modulus of continuity ``\omega``, then 
```math
	\inf_{p_N \in \mathcal{P}_N} \| f - p_N \|_\infty \lesssim N^{-r} \omega(N^{-1}).
```

**Theorem:** Let ``f \in C([-1,1])`` also be analytic and bounded in the interior of the Bernstein ellipse 
```math
	E_\rho := \bigg\{ z = x + i y \in \mathbb{C} : 
  				\bigg(\frac{x}{\frac12 (\rho+\rho^{-1})}\bigg)^2
				+ \bigg(\frac{y}{\frac12 (\rho-\rho^{-1})}\bigg)^2 \leq 1 \bigg\}
```
where ``\rho > 1``, then 
```math
	\inf_{p_N \in \mathcal{P}_N} \| f- p_N \|_\infty \lesssim \rho^{-N}.
```

"""

# ╔═╡ 16591e8e-9631-11eb-1157-992452ed6514
md"""
With these results in hand we can perform a few more tests. We have already shown exponential convergence for an analytic target function; now we experiment with $$C^{j,\alpha}$$ smoothness, 
```math
\begin{aligned}
	f_1(x) &= |\cos(\pi x)|^{t} 
\end{aligned}
```
where ``t`` is given by:  $(@bind _t1 Slider(0.5:0.01:4.0, show_value=true))
"""

# ╔═╡ 7b835804-9694-11eb-0692-8dab0a07203e
let t = _t1, f = x -> abs(cos(π * x))^t, NN = (2).^(2:10)
	xerr = range(-1, 1, length=2013)
	err(N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp(f, N))), Inf)
	plot(NN, err.(NN), lw=2, m=:o, ms=4, label = "error", 
		 size = (300, 250), xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10, xscale = :log10, 
		 title = L"f(x) = |\cos(\pi x)|^{%$t}")
	plot!(NN[4:end], Float64.(NN[4:end]).^(-t), lw=2, ls=:dash, c=:black, 
		  label = L"N^{-%$t}")	
end

# ╔═╡ c1e69df8-9694-11eb-2dde-673888b5f7e2
md"""
## §4.2 An Example from Computational Chemistry

According to the Fermi-Dirac model, the distribution of particles over energy states in systems consisting of many identical particles that obey the Pauli exclusion principle is given by 
```math
	f_\beta(E) = \frac{1}{1 + \exp(\beta (E-\mu))},
```
where ``\mu`` is a chemical potential and ``\beta = \frac{1}{k_B T}`` the inverse temperature. 

Choose a ``\beta``: $(@bind _beta0 Slider(5:100, show_value=true))
"""

# ╔═╡ 5522bc18-9699-11eb-2d67-759be6fc8d62
plot(x -> 1 / (1 + exp(_beta0 * x)), -1, 1, lw=2, label = "", 
	title = L"\beta = %$_beta0", size = (300, 200), 
	xlabel = L"E", ylabel = L"f_\beta(E)")

# ╔═╡ 4fe01af2-9699-11eb-1fb7-9b43a947b51a
md"""
We will return to the chemistry context momentarily but first simply explore the approximation of the Fermi-Dirac function by polynomials. To this end we set ``\mu = 0`` and assume ``E \in [-1, 1]``.

To determine the rate of convergence, we have to find the largest possible Bernstein ellipse. The singularlities of ``f`` are given by 
```math
	i \pi / \beta (1 + 2n), \qquad n \in \mathbb{Z}.
```
This touches the semi-minor axis provided that 
```math
	\frac12 (\rho - 1/\rho) = \pi / \beta
	\qquad \Leftrightarrow \qquad 
	\rho = \sqrt{(\pi/\beta)^2 + 1} + \pi/\beta.
```

Choose a ``\beta``: $(@bind _beta1 Slider(5:100, show_value=true))
"""

# ╔═╡ b3bde6c8-9697-11eb-3ca2-1bacec3f2ad1
let β = _beta1, f = x -> 1 / (1 + exp(β * x)), NN = 6:6:110
	ρ = sqrt( (π/β)^2 + 1 ) + π/β
	xerr = range(-1, 1, length=2013)
	err(N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp(f, N))), Inf)
	plot(NN, err.(NN), lw=2, m=:o, ms=4, label = "error", 
		 size = (300, 250), xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10, ylims = [1e-16, 1.0], legend = :bottomleft, 
		 title = L"\beta = %$(β)")
	plot!(NN[4:end], ρ.^(-NN[4:end]), lw=2, ls=:dash, c=:black, 
		  label = L"\rho^{-N}")	
end 

# ╔═╡ 0c74de06-9699-11eb-1234-eb64590644d7
md"""
Although the rate is always exponential it deteriorates severely as ``\beta \to \infty``. Unfortunately these large values of ``\beta`` are imporant and realistic for applications. We will return to this momentarily.
"""

# ╔═╡ 330aec36-9699-11eb-24de-794562156ef4
md"""
### Approximation of a Matrix Function

Approximating ``f_\beta`` by a polynomial is in itself not useful. Since we have very accurate and performant means to evaluate ``e^x`` this is simply not necessary. But there is a different reason why a polynomial approximation is useful. 

Let ``H \in \mathbb{R}^{P \times P}`` be a hamiltonian describing the interaction of ``P`` fermions (e.g. electrons). Many properties of this quantum mechanical system can be extracted from the *density matrix*, 
```math
	\Gamma := f_\beta(H).
```
A simple way to define this matrix function is to diagonalize ``H = Q \Lambda Q^*`` and then evaluate 
```math
	\Gamma = Q f_\beta(\Lambda) Q^* 
		= Q {\rm diag} \big( f_\beta(\lambda_1), \dots, f_\beta(\lambda_P) \big) Q^*
```
This requires diagonalisation of ``H`` an operation that costs ``O(P^3)`` operations. 

Our idea now is to substitute ``f_\beta`` for a polynomial approximation and analyse the effect this has on the matrix function.

**Proposition:** If ``\|f_\beta - p_N\|_\infty \leq \epsilon`` on ``\sigma(H)`` then 
```math
	\| f_\beta(H) - p_N(H) \|_{\rm op} \leq \sup_{E \in \sigma(H)} \big| f_\beta(E) - p_N(E) \big|
```
"""

# ╔═╡ 53439934-975f-11eb-04f9-43dc19404d6b
md"""
We can evaluate ``p_N(H)`` with ``N`` matrix multiplications. If ``H`` is dense then we don't gain anything, but if ``H`` is sparse e.g. that it has only ``O(P)`` entries, then one can show that 
```math
	{\rm cost} \approx N^2 P \approx \beta^2 \log^2(\epsilon) P.
```
where ``\epsilon`` is the desired accuracy. For small or moderate ``\beta`` this is an excellent result allowing us to trade accuracy against computational cost, when ``P`` is very large.

However when ``\beta`` is large, then the poor convergence rate makes this a less attractive algorithm. E.g., if ``\sigma(H) \subset [-1,1]`` and we require an accuracy of ``\epsilon = 10^{-6}`` (this is a typical target) then we require that 
```math
 	\beta \leq \frac{P}{13}
```
for the polynomial algorithm to be more efficient that the diagonalisation algorithm, which is quite restrictive. Note that ``\beta \in [30, 300]`` is a physically realistic range: 
	$(@bind _betap Slider(30:300, show_value=true))
"""

# ╔═╡ e23de7e6-98c5-11eb-1d25-df4887dab5f6
# Example Code on Evaluating a Matrix Polynomial
# ------------------------------------------------
let β = _betap, P = 10, N = 50, f = x -> 1/(1+exp(β*x))
	
	# a random "hamiltonian" and the exact Γ via diagonalisation
	H = 0.6 * (rand(P,P) .- 0.5); H = 0.5 * (H + H')
	F = eigen(H)
	Γ = F.vectors * Diagonal(f.(F.values)) * F.vectors'
	
	# The polynomial approximant: 
	F̃ = chebinterp_naive(f, N)
	ΓN = chebeval(H, F̃)    # <----- can use the generic code!!!
	norm(Γ - ΓN)
end

# ╔═╡ e3466610-9760-11eb-1562-61a6a1bf76bf
let β = _betap, P = 10, NN = 4:10:100, f = x -> 1 / (1 + exp(β * x))
	# a random "hamiltonian" and the exact Γ
	Random.seed!(3234)
	H = 0.6 * (rand(P,P) .- 0.5); H = 0.5 * (H + H')
	F = eigen(H)
	Γ = F.vectors * Diagonal(f.(F.values)) * F.vectors'

	errs = [] 
	errsinf = []
	xerr = range(-1, 1, length=2013)
	for N in NN 
		F̃ = chebinterp_naive(f, N)
		ΓN = chebeval(H, F̃)
		push!(errs, norm(ΓN - Γ))
		push!(errsinf, norm(f.(xerr) - chebeval.(xerr, Ref(F̃)), Inf))
	end
	
	plot(NN, errs, lw=2, ms=4, m=:o, label = "Matrix Function",
		 size = (450, 250), yscale = :log10, xlabel = L"N", ylabel = "error", 
		 title = L"\beta = %$β", legend = :outertopright)
	plot!(NN, errsinf, lw=2, ms=4, m=:o, label = "Scalar Function")
	plot!(NN[5:end], 2*exp.( - π/β * NN[5:end] ), c = :black, lw=2, ls=:dash, label = "predicted")
	
end

# ╔═╡ 27a272c8-9695-11eb-3a5f-014392471e7a
md"""
### Rational Approximation of the Fermi-Dirac Function 

A rational function is a function of the form 
```math 
	r_{NM}(x) = \frac{p_N(x)}{q_M(x)},
```
were ``p_N, p_N`` are polynomials of, respectively, degrees ``N, M``. Note that ``p_N, p_M`` are both linear in its parameters, but ``r_{NM}`` is not. It is our first example of a **nonlinear approximation**. This makes both theory and practise significantly more challenging. In particular, there are many different techniques to construct and analyze rational approximants. Here, we will only give one example.
"""

# ╔═╡ a389d3a8-979a-11eb-0795-eb29979e1141
md"""
Recall that the Fermi-dirac function has poles at 
```math
	\zeta_n := i \pi/\beta (1 + 2n), \qquad n \in \mathbb{Z}.
```
"""

# ╔═╡ ce682a00-979a-11eb-1367-8780f4e400f9
_poles1 = let β = 10, f = x -> 1 / (1 + exp(β * x))
	xx = range(-1,1,length=30) 
	yy = range(-5,5, length = 200)
	X = xx * ones(length(yy))'; Y = ones(length(xx)) * yy'
	contour(xx, yy, (x, y) -> abs(f(x + im * y)), 
		    size = (200, 400), 
		    xlabel = L"{\rm Re} z", ylabel = "Poles of the Fermi-Dirac Function", 
	        colorbar = false, grid = false)
	plot!([-1, 1], [0,0], lw=2, c=:black, label= "")
end

# ╔═╡ 18d70404-979c-11eb-326f-25e63c23456c
md"""
The poles are given by 
```math
	f_\beta(z) \sim -\frac{1}{\beta (z-z_n)} \qquad \text{as } z \to z_n
```
so we can remove them by considering a new function
```math
	g(z) = f_\beta(z) - \frac{1}{\beta (z - z_n)}.
```
For example, let us remove the first three poles above and below the real axis, corresponding to the indices
```math
	n = -3, -2, -1, 0, 1, 2.
```
then we get the following picture:
"""

# ╔═╡ b758732e-979c-11eb-2a76-5ddc8c37a6c2
_poles2 = let β = 10, f = x -> 1 / (1 + exp(β * x)), nn = -3:2
	zz(n) = im * π/β * (1+2*n)
	pole(n, z) = -1 / (β * (z - zz(n)))
	xx = range(-1,1,length=30) 
	yy = range(-5,5, length = 200)
	contour(xx, yy, (x, y) -> (z = x+im*y; abs(f(z) - sum(pole(n,z) for n in nn))),
		    size = (200, 400), 
		    xlabel = L"{\rm Re} z", ylabel = "Poles of the Fermi-Dirac Function", 
	        colorbar = false, grid=false)
	plot!([-1, 1], [0,0], lw=2, c=:black, label= "")
	# plot!(zeros(6), imag.(zz.(nn)), lw=0, ms=1, m=:o, c=:red, label= "")
end;

# ╔═╡ 20cd7ccc-97a1-11eb-301a-41af065eb0a0
plot(_poles1, _poles2, size = (400, 500))

# ╔═╡ 349741a8-979d-11eb-35f2-5db54ef72ad0
md"""
Why is this useful? Remember the Bernstein ellipse! We have constructed a new function ``g(z) = f(z) - \sum_n (-1)/(\beta (z-z_n))`` with a much larger region of analyticity to fit a Bernstein ellipse into.  
"""

# ╔═╡ ffeb2d86-979f-11eb-21ea-176800eb4f5d
let β = 10, P1 = deepcopy(_poles1), P2 = deepcopy(_poles2)
	# b = 0.5*(ρ-1/ρ) ⇔ ρ^2 - 2 b ρ - 1 = 0
	b1 = π/β; ρ1 = b1 + sqrt(b1^2 + 1); a1 = 0.5 * (ρ1+1/ρ1)
	b2 = 7*π/β; ρ2 = b2 + sqrt(b2^2 + 1); a2 = 0.5 * (ρ2+1/ρ2)
	tt = range(0, 2π, length = 200)
	plot!(P1, a1*cos.(tt), b1*sin.(tt), lw=2, c=:red, label = "", 
		  title = L"\rho = %$(round(ρ1, digits=2))")
	plot!(P2, a2*cos.(tt), b2*sin.(tt), lw=2, c=:red, label = "", 
		  xlims = [-1.2, 1.2], title = L"\rho = %$(round(ρ2, digits=2))")
	plot(P1, P2, size = (400, 500))
	# @show a1, b1, π/β
	# @show a2, b2, π/β * 7
end 

# ╔═╡ 2f95c718-97a9-11eb-10b9-3d7751afe1ee
md"""
As the next step we construct a polynomial approximation of the Fermi-Dirac function with the poles remove, i.e., 
```math
	p_N(x) \approx g_\beta := f_\beta(x) + \sum_{n = -3}^2 \frac{1}{\beta (z - z_n)}
```
for which we know we have the *much improved rate* 
```math
	\|p_N - g_\beta\|_\infty \lesssim \rho_3^{-N},
```
where ``\rho_3 = 7\pi/\beta + \sqrt{1 + (7\pi/\beta)^2}``. This translates into a rational approximant, 
```math
	r_{N+6, 6}(x) := p_N(x) - \sum_{n = -3}^2 \frac{1}{\beta (z - z_n)},
```
with the same rate, 
```math
	\| r_{N+6, 6} - f_\beta \|_\infty \lesssim \rho_3^{-N}.
```
"""

# ╔═╡ c18d41ea-97aa-11eb-13fc-a9bf95807fd2
md"""
In general, we can remove ``M`` poles above and below the real axis to obtain 
```math
	r_{N+2M, 2M}(x) := p_N(x) - \sum_{n = -M}^{M-1} \frac{1}{\beta (z - z_n)}
```

For the following convergence test, we choose ``M = 0, 1, 3, 6, 10``.

Choose ``\beta``: $(@bind _betarat Slider(5:100, show_value=true))
"""

# ╔═╡ ca1c8836-97da-11eb-3498-e1b35b94c46d
let NN = 5:5:30, MM = [0, 1, 3, 6, 10], β = _betarat
	zz(n) = im * π/β * (1+2*n)
	pole(n, z) = -1 / (β * (z - zz(n)))
	xerr = range(-1, 1, length=2013)
	err(g, N) = norm(g.(xerr) - chebeval.(xerr, Ref(chebinterp(g, N))), Inf)

	P = plot(; size = (400, 250), yscale = :log10, 
			 xlabel = L"N", ylabel = L"\Vert f - r_{N'M'} \Vert_\infty",
			 title = L"\beta = %$β", legend = :outertopright, 
			  yrange = [1e-8, 1e0])
	for (iM, M) in enumerate(MM)
		g = (M == 0 ? x -> 1/(1+exp(β*x)) 
			        :  x -> ( 1/(1+exp(β*x)) - sum( pole(m, x) for m = -M:M-1 ) ))
		errs = err.(Ref(g), NN) 
		plot!(P, NN, errs, lw=2, c = iM, label = L"M = %$M")
		b1 = (2*M+1) * π/β
		ρ = b1 + sqrt(1 + b1^2)
		lab = M == MM[end] ? L"\rho^{-N}" : ""
		plot!(P, NN[4:end], (ρ).^(-NN[4:end]), c=:black, lw=2, ls=:dash, label = lab)
	end	
	P
end

# ╔═╡ ee8fa624-97dc-11eb-24d6-033395124bd6
md"""
This looks extremely promising: we were able to obtain very accurate rational approximation with much lower polynomial degrees. Can you use these to evaluate our matrix functions? What is ``r(H)``? Basically, we need to understand that ``(z - z_0)^{-1}|_{z = H} = (H - z_0)^{-1}`` i.e. this requires a matrix inversion. More generally, 
```math
	\Gamma \approx r(H) = p_N(H) - \sum_{m = -M}^{M-1} \beta^{-1} (H - z_m)^{-1}.
```
A general matrix inversion requires again ``O(P^3)`` operations, but for sparse matrices this can be significantly reduced. Moreover, one rarely need the entire matrix ``\Gamma`` but rather its action on a vector, i.e., ``\Gamma \cdot v`` for some ``v\in \mathbb{R}^{P}``.  In this case, we obtain 
```math
	\Gamma v = p_N(H) v - \sum_{m = -M}^{M-1} \beta^{-1} (H - z_m)^{-1} v.
```
If ``H`` is again sparse with ``O(P)`` entries then evaluating ``p_N(H) v`` would require ``O(NP)`` operations, while evaluating the rational contribution would require the solution of ``2M`` linear systems. Here the cost depends on the underlying dimensionality / connectivity of the hamiltonian and a detailed analysis goes beyond the scope of this course but typically the cost is ``O(P)`` in one dimension, ``O(P^{1/2})`` in 2D and ``O(P^2)`` in 3D. This already makes it clear that the tradeoff between the ``N, M`` parameters can be subtle and situation specific. 

We will leave this discussion here as open-ended.
"""

# ╔═╡ 3fb12378-9695-11eb-0766-9faf92928ad2
md"""
## §4.3 Best Approximation via IRLSQ

While best approximation in the least squares sense (``L^2`` norm) is relatively easy to characterise and to implement at least in a suitable limit that we can also understand (cf. Lecture 3). But for max-norm approximation we have always been satisfies with "close to best" approximations, or just "good approximations. For example, we have proven results such as
```math
	\| f -  I_N f\|_\infty 
    \leq 
	C \log N \inf_{t_N \in \mathcal{T}_N'} \| f - t_N \|_\infty,
```
where ``I_N`` denotes the trigonometric interpolant. A fully analogous result holds for Chebyshev interpolation: if ``I_N`` now denotes the Chebyshev interpolant, then 
```math 
	\| f -  I_N f\|_\infty 
    \leq 
	C \log N \inf_{t_N \in \mathcal{P}_N} \| f - p_N \|_\infty,


```

But now we are going to explore a way to obtain *actual* best approximations. What we do here applied both to trigonometric and algebraic polynomials, but since today's lecture is about algebraic polynomials (and rational functions) we will focus on those. Students are strongly encouraged to experiment also with best approximation with trigonometric polynomials.
"""

# ╔═╡ 049e4ff2-982c-11eb-3094-3520f14eb76b
md"""
Thus, the problem we aim to solve is to *find* ``p_N^* \in \mathcal{P}_N`` *such that*
```math
	\| p_N^* - f \|_\infty \leq \| p_N - f \|_\infty \qquad \forall p_N \in \mathcal{P}_N.
```
It turns out this problem has a unique solution; this is a non-trivial result which we won't prove here. 
"""


# ╔═╡ 43d44336-982c-11eb-03d1-f990c92d1832
md"""
Let us begin by highlighting how the Chebyshev interpolant *fails* to be optimal. Let 
```math
	f(x) = \frac{1}{1+25 x^2}
```
and choose ``N`` : $(@bind _N3 Slider(6:2:20, show_value=true))
"""

# ╔═╡ 5101a786-982c-11eb-2d10-53d5638ec977
let f = x -> 1 / (1 + 25 * x^2), N = _N3
	F̃ = chebinterp(f, N)
	xp = range(-1, 1, length = 300)
	err = abs.(f.(xp) - chebeval.(xp, Ref(F̃)))
	plot(xp, err, lw=2, label = "error", size = (350, 200), 
		 title = "Chebyshev interpolant")
end

# ╔═╡ c7cf9f9e-982c-11eb-072c-8367aaf306a0
md"""
We observe that the error is not equidistributed across the interval, this means that we could sacrifice some accuracy near the boundaries in return for lowering the error in the centre. 

We can observe the same with a least-squares fit:
"""

# ╔═╡ 5f561f82-98c8-11eb-1719-65c2a8aea11d
begin 
	function chebfit(f, X, N; W = ones(length(X)))
		F = W .* f.(X) 
		A = zeros(length(X), N+1)
		for (m, x) in enumerate(X)
			A[m, :] = W[m] * chebbasis(x, N) 	
		end
		return qr(A) \ F 
	end	
end

# ╔═╡ d52d8e70-98c8-11eb-354a-374d26f6252a
md"""
choose ``N`` : $(@bind _N4 Slider(6:2:20, show_value=true))
"""

# ╔═╡ 92eba25e-98c8-11eb-0e32-9da6e0f5173b
let f = x -> 1 / (1 + 25 * x^2), N = _N4
	F̃ = chebfit(f, range(-1, 1, length = 30 * N),  N)
	xp = range(-1, 1, length = 300)
	err = abs.(f.(xp) - chebeval.(xp, Ref(F̃)))
	plot(xp, err, lw=2, label = "error", size = (350, 200), 
		 title = "Least Squares Approximant")
end

# ╔═╡ 00478e4e-98c9-11eb-1553-816d22e7dd7d
md"""
So the idea is to modify the least squares loss function, 
```math
	L({\bf c}) = \sum_m \big|f(x_m) - p({\bf c}; x_m) \big|^2
```
by putting more weight where the error is large, 
```math
	L({\bf c}) = \sum_m w_m \big|f(x_m) - p({\bf c}; x_m) \big|^2
```
Now of course we don't know beforehand where the error is large and we wouldn't know what weight to put there. So instead we simply "learn" this: 
* perform a standard least squares fit
* estimate the error and put higher weights where the error is large
* iterate
"""

# ╔═╡ f88e49f2-98e3-11eb-1bf1-811331840a44
md"""
Let us look at one iteration of this idea.

STEP 1: Construct the initial LSQ solution
"""

# ╔═╡ 0f72a51e-98e4-11eb-3b5f-1728dd5c0e61
begin 
	f(x) = 1 / (1 + 25 * x^2) 
	N = 20
	xfit = range(-1,1, length=400)
	F̃ = chebfit(f, xfit, N)
end

# ╔═╡ f958b76a-98f1-11eb-3e8c-69136c5f8fac
plot(x -> abs(f(x) - chebeval(x, F̃)), -1, 1, lw=2, size = (300, 200), label = "")

# ╔═╡ 2946dbfa-98f2-11eb-2bcf-a1657769d9f1
md"""
**STEP 2:** 
Now let's introduce the weights. Our goal is to get a max-norm approximation, so let's choose the weights so that the ``\ell^2`` becomes a little more like a max-norm: 
```math
    \sum_{m}  |e(x_m)|^2 \qquad \leadsto \qquad 
	\sum_m \underset{=: W_m}{\underbrace{|e(x_m)|^\gamma}} \cdot |e(x_m)|^2.
```
This will put a little more weight on nodes ``x_m`` where the error is large. The parameter ``\gamma`` is a fudge parameter. Here we choose ``\gamma = 1``. 
"""

# ╔═╡ 62e8454e-98f3-11eb-0a12-6554b76e3e18
begin 
	e = f.(xfit) .- chebeval.(xfit, Ref(F̃))
	W = sqrt.(abs.(e))
	F̃2 = chebfit(f, xfit, N; W = W)
end

# ╔═╡ a5da6f94-98f3-11eb-098f-b779f14a69f5
begin
	plot(x -> abs(f(x) - chebeval(x, F̃)), -1, 1, lw=2, size = (300, 200), label = "iter-1")
	plot!(x -> abs(f(x) - chebeval(x, F̃2)), -1, 1, lw=2, label = "iter-2")
end 

# ╔═╡ b96c85a8-98f3-11eb-0d3b-45673e8b1e94
md"""
We see that the error has slightly increased near the edges and slightly decreased in the center. Let's do one more iteration.
"""

# ╔═╡ ccd6d074-98f3-11eb-1f74-71f3e87d3a29
begin 
	e2 = f.(xfit) .- chebeval.(xfit, Ref(F̃2))
	W2 = W .* sqrt.(abs.(e2))
	F̃3 = chebfit(f, xfit, N; W = W2)
end

# ╔═╡ e24421fc-98f3-11eb-32b0-3ff33d95fa18
begin
	plot(x -> abs(f(x) - chebeval(x, F̃)), -1, 1, lw=2, size = (300, 200), label = "iter-1")
	plot!(x -> abs(f(x) - chebeval(x, F̃2)), -1, 1, lw=2, label = "iter-2")
	plot!(x -> abs(f(x) - chebeval(x, F̃3)), -1, 1, lw=2, label = "iter-3")
end 

# ╔═╡ f063df16-98f3-11eb-0ea2-aff385524ac7
md"""
This looks extremely promising and suggests the following algorithm:

**Iteratively Reweighted Least Squares:**
1. Initialize ``W_m = 1``
2. Solve LSQ problem to obtain approximation ``p``
3. Update weights ``W_m \leftarrow W_m \cdot |f(x_m) - p(x_m)|^\gamma``
4. If weights have converged, stop, otherwise go to 2

**Remarks:**
* I've implemented this with a few tweaks for numerical stability in `tools.jl`. 
* This can be implemented quite effectively for rational approximation as well.
* As far as I am aware there is no general convergence result for this method.

I want to finish now with one final experiment quantitatively exploring Chebyshev interpolation against max-norm best approximation. But we make the problem a little harder by approximating 
```math
	f_1(x) = \frac{1}{1 + 100 x^2}, \qquad f_2(x) = |\sin(\pi x)|^2
```
"""

# ╔═╡ c9c2b106-98f4-11eb-279e-2b61f36e2fe5
let f1 = x -> 1 / (1+100*x^2), NN1 = 6:6:100, 
			f2 = x -> abs(sin(π*x)), NN2 = (2).^(2:7)
	xfit = range(-1,1, length = 3123)
	xerr = range(-1,1, length = 2341)
	interperr(f, N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp_naive(f, N))), Inf)
	besterr(f, N) = norm(f.(xerr) - chebeval.(xerr, Ref(IRLSQ.bestcheb(f, xfit, N))), Inf)
	
	ierrs1 = interperr.(f1, NN1)
	berrs1 = besterr.(f1, NN1)
	P1 = plot(NN1, ierrs1, lw=2, m=:o, ms=4, label = "Chebyshev")
	plot!(P1, NN1, berrs1, lw=2, m=:o, ms=4, label = "best", 
		 size = (350, 250), xlabel = L"N", ylabel = L"\Vert f - p_N \Vert", 
		  yscale = :log10, title = L"f_1(x) = 1/(1+100 x^2)")

	ierrs2 = interperr.(f2, NN2)
	berrs2 = besterr.(f2, NN2)
	P2 = plot(NN2, ierrs2, lw=2, m=:o, ms=4, label = "Chebyshev")
	plot!(P2, NN2, berrs2, lw=2, m=:o, ms=4, label = "best", 
		 size = (350, 250), xlabel = L"N", ylabel = L"\Vert f - p_N \Vert", 
		  xscale = :log10, yscale = :log10, title = L"f_2(x) = |\sin(\pi x)|^3")

	plot(P1, P2, size = (600, 250))
end

# ╔═╡ 90785a4e-9981-11eb-368e-cdf087a39217
md"""
Further reading:
* L. N. Trefethen, Approximation Theory and Approximation Practice
* Y. Nakatsukasa, O. Sète, L. N. Trefethen, The AAA algorithm for rational approximation, arXiv:1612.00337
* Lin, L. and Chen, M. and Yang, C. and He, L., Accelerating atomic orbital-based electronic structure calculation via pole expansion and selected inversion, J. Phys. Condens. Matter
"""

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
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.3.1"
DataFrames = "~1.3.2"
FFTW = "~1.4.5"
ForwardDiff = "~0.10.25"
LaTeXStrings = "~1.3.0"
Plots = "~1.25.10"
PlutoUI = "~0.7.34"
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
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

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
git-tree-sha1 = "25d90d444b608666143d7e276c17be6f5f3e9bb9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.10"

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
git-tree-sha1 = "d9c49967b9948635152edaa6a91ca4f43be8d24c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.10"

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
git-tree-sha1 = "95c6a5d0e8c69555842fc4a927fc485040ccc31c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.5"

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
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

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
# ╟─66772e00-9981-11eb-1941-d9132b99c780
# ╟─465e7582-95c7-11eb-0c7b-ed7dddc24b4f
# ╟─58d79c34-95c7-11eb-0679-eb9741761c10
# ╟─9c169770-95c7-11eb-125a-4f174da56d36
# ╟─05ec7c1e-95c8-11eb-176a-3372a765d4d7
# ╟─311dd26e-95ca-11eb-2ff5-03a53a662a62
# ╟─4428c460-95c8-11eb-16fc-6327acc80bb6
# ╟─3533717c-9630-11eb-3fc6-3f4b29eaa63a
# ╟─467c7f76-9630-11eb-1cdb-6dedae3c459f
# ╟─6cbcac96-95ca-11eb-05f9-c7a781b38061
# ╠═77dcea8e-95ca-11eb-2cf1-ad342f0f6d7d
# ╟─27f7e2c8-95cb-11eb-28ca-dfe90be89670
# ╟─cc8fe0f4-95cd-11eb-04ec-d3fc8cced91b
# ╟─dfc0866c-95cb-11eb-1869-7bd67468c233
# ╟─f3cd1326-95cd-11eb-192a-efc3371d17b6
# ╟─5181b8f4-95cf-11eb-30fe-cd93f4ed0012
# ╟─7661921a-962a-11eb-04c0-5b7e486aea05
# ╟─4ca65bec-962b-11eb-3aa2-d5500fd24677
# ╠═69dfa184-962b-11eb-0ed2-c17aa6c15a45
# ╟─3ef9e58c-962c-11eb-0172-a700d2f7e72c
# ╟─19509e38-962d-11eb-1f5c-811188a57261
# ╟─16591e8e-9631-11eb-1157-992452ed6514
# ╟─7b835804-9694-11eb-0692-8dab0a07203e
# ╟─c1e69df8-9694-11eb-2dde-673888b5f7e2
# ╟─5522bc18-9699-11eb-2d67-759be6fc8d62
# ╟─4fe01af2-9699-11eb-1fb7-9b43a947b51a
# ╠═b3bde6c8-9697-11eb-3ca2-1bacec3f2ad1
# ╟─0c74de06-9699-11eb-1234-eb64590644d7
# ╟─330aec36-9699-11eb-24de-794562156ef4
# ╠═53439934-975f-11eb-04f9-43dc19404d6b
# ╟─e23de7e6-98c5-11eb-1d25-df4887dab5f6
# ╠═e3466610-9760-11eb-1562-61a6a1bf76bf
# ╟─27a272c8-9695-11eb-3a5f-014392471e7a
# ╟─a389d3a8-979a-11eb-0795-eb29979e1141
# ╟─ce682a00-979a-11eb-1367-8780f4e400f9
# ╟─18d70404-979c-11eb-326f-25e63c23456c
# ╟─b758732e-979c-11eb-2a76-5ddc8c37a6c2
# ╟─20cd7ccc-97a1-11eb-301a-41af065eb0a0
# ╟─349741a8-979d-11eb-35f2-5db54ef72ad0
# ╟─ffeb2d86-979f-11eb-21ea-176800eb4f5d
# ╟─2f95c718-97a9-11eb-10b9-3d7751afe1ee
# ╟─c18d41ea-97aa-11eb-13fc-a9bf95807fd2
# ╟─ca1c8836-97da-11eb-3498-e1b35b94c46d
# ╟─ee8fa624-97dc-11eb-24d6-033395124bd6
# ╟─3fb12378-9695-11eb-0766-9faf92928ad2
# ╟─049e4ff2-982c-11eb-3094-3520f14eb76b
# ╟─43d44336-982c-11eb-03d1-f990c92d1832
# ╟─5101a786-982c-11eb-2d10-53d5638ec977
# ╟─c7cf9f9e-982c-11eb-072c-8367aaf306a0
# ╠═5f561f82-98c8-11eb-1719-65c2a8aea11d
# ╟─d52d8e70-98c8-11eb-354a-374d26f6252a
# ╟─92eba25e-98c8-11eb-0e32-9da6e0f5173b
# ╟─00478e4e-98c9-11eb-1553-816d22e7dd7d
# ╟─f88e49f2-98e3-11eb-1bf1-811331840a44
# ╠═0f72a51e-98e4-11eb-3b5f-1728dd5c0e61
# ╟─f958b76a-98f1-11eb-3e8c-69136c5f8fac
# ╟─2946dbfa-98f2-11eb-2bcf-a1657769d9f1
# ╠═62e8454e-98f3-11eb-0a12-6554b76e3e18
# ╟─a5da6f94-98f3-11eb-098f-b779f14a69f5
# ╟─b96c85a8-98f3-11eb-0d3b-45673e8b1e94
# ╠═ccd6d074-98f3-11eb-1f74-71f3e87d3a29
# ╟─e24421fc-98f3-11eb-32b0-3ff33d95fa18
# ╟─f063df16-98f3-11eb-0ea2-aff385524ac7
# ╟─c9c2b106-98f4-11eb-279e-2b61f36e2fe5
# ╟─90785a4e-9981-11eb-368e-cdf087a39217
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
