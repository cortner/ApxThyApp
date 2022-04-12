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

# ╔═╡ 74479a7e-8c34-11eb-0e09-73f47f8013bb
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
          PlutoUI, BenchmarkTools, ForwardDiff, Printf, Random, FFTW
	include("tools.jl")
end

# ╔═╡ 8728418e-8c34-11eb-3313-c52ecadbf252
md"""

## §3 Least Squares

The topic of the next set of lectures is least squares regression. Fitting the parameters of a model to general observations about that model is a ubiquitious problem occuring throughout the sciences, engineering and technology. The focus of this lecture will be a relatively simple setting where we can explore very precisely/rigorously how we can connect least squares methods and approximation theory. In the end we will also see how least squares methods can be tweaked in an unexpected way to solve the difficult max-norm best approximation problem!

* fitting a trigonometric polynomial to random data
* lsq fitting versus approxmation
* iteratively reweighted lsq for best approximation in the max-norm
"""

# ╔═╡ 9202b69e-9013-11eb-02a2-2f1c8e2fcc0c
md"""

## §3.2 Fitting to Point Values

We now consider the vastly simpler question of determining the coefficients (parameters) of a trigonometric polynomial ``t_N \in \mathcal{T}_N`` or ``t_N \in \mathcal{T}_N'`` by minimising the least squares functional
```math
	L(\boldsymbol{c}) := \frac12 \sum_{m = 1}^M \big| t_N(x_j) - f_j \big|^2,
```
where the tuples ``(x_j, f_j)`` are the "training data" and ``L`` is called the loss function.

Trigonometric interpolation can in fact be seen as a special case: if we take ``M = 2N`` and ``x_j`` the interpolation points, the loss can be minimised to achieve ``L(\boldsymbol{c}) = 0`` and the minimiser is precisely the solution of the linear system that defined the minimiser.

But we also want to explore the situation that we *cannot* choose the training data ``(x_j, f_j)`` but it is given to us. An interesting generic case that is in fact close to many real-world scenarios is that the training data is random. That is, we will take
```math
	x_j \sim U(-\pi, \pi), \qquad {\rm iid}
```
Moreover, we assume that the function values ``f_j`` are consistent, i.e. they arise from evaluation of a smooth function ``f(x)`` possibly subject to noise (e.g. due to measurement errors or model errors),
```math
	f_j = f(x_j) + \eta_j, \qquad \eta_j \sim N(0, \sigma), \quad {\rm iid}.
```
Assuming that the noise is normally distributed and iid is a particularly convenient scenario for analysis.
"""


# ╔═╡ 73983438-8c51-11eb-3142-03410d610022
md"""
#### Implementation

Before we can start experimenting with this scenario, we need to discuss how to implement least squares problems. We will only be concerned with the case when ``M \geq 2N`` i.e. there is sufficient data to determine the coefficients (at least in principle).

We begin by rewriting it in terms of the parameters, here for the case ``t_N \in \mathcal{T}_N``. Let ``A_{mk} = e^{i k x_j}`` be the value of the ``k``th basis function at the data point ``x_m``, then
```math
	\begin{aligned}
		L(\boldsymbol{c})
		&=
			\frac12 \sum_{m = 1}^M
			\bigg| \sum_{k = -N}^N c_k e^{i k x_m} - f_m \bigg|^2
		\\
		&=
			\frac12
			\frac12 \sum_{m = 1}^M
			\bigg| \sum_{k = -N}^N  A_{mk} c_k - f_m \bigg|^2
		\\
		&=
			\frac12 \sum_{m = 1}^M \Big| (A \boldsymbol{c})_m - f_m \Big|^2
		\\
		&=
			\frac12 \big\| A \boldsymbol{c} - \boldsymbol{f} \big\|^2,
	\end{aligned}
```
where ``\boldsymbol{f} = (f_m)_{m = 1}^M``.
This is a *linear least-squares system*. The matrix ``A`` is called the *design matrix*.

The first-order criticality condition, ``\nabla L(\boldsymbol{c}) = 0`` takes the form
```math
	A^* A \boldsymbol{c} = A^* \boldsymbol{f}
```
The equations making up this linear system are called the *normal equations*. 

**Lemma:** The least square problem has a unique minimizer if and only if the normal equations have a unique solution if and only if ``A`` has full rank.

We might be tempted to assemble the matrix ``A`` then form ``A^* A`` via matrix multiplication and then solve the system, e.g., using the Cholesky factorisation. This can go very badly since ``{\rm cond}(A^* A) = {\rm cond}(A)^2``, i.e. numerical round-off can become severe. Instead one should normally use the numerically very stable QR factorisation: there exist ``Q \in \mathbb{C}^{M \times 2N}`` and ``R \in \mathbb{C}^{2N \times 2N}`` such that 
```math 
		A = Q R 
```
With that in hand, we can manipulate ``A^* A = R^* Q^* Q R = R^* R`` and hence 
```math 
\begin{aligned} 
	& A^* A \boldsymbol{c} = A^* \boldsymbol{f} \\ 
	%
	\Leftrightarrow \qquad & 
	R^* R \boldsymbol{c} = R^* Q^* \boldsymbol{f} \\ 
	%
	\Leftrightarrow \qquad & 
	R \boldsymbol{c} = Q^* \boldsymbol{f}.
\end{aligned}
```
Moreover, since ``R`` is upper triangular the solution of this system can be performed in ``O(N^2)`` operations.
"""

# ╔═╡ d08fb928-8c53-11eb-3c35-574ef188de6b

# implementation of a basic least squares code
begin
	"""
	note that we now use k = -N,..., N; but we respect the ordering of the FFT	
	"""
	kgridproj(N) = [0:N; -N:-1]
	
	"""
	trigonometric basis consistent with `kgridproj`
	"""
	trigbasis(x, N) = [exp(im * x * k) for k = kgridproj(N)]

	function designmatrix(X, N)
		A = zeros(ComplexF64, length(X), 2*N+1)
		for (m, x) in enumerate(X)
			A[m, :] .= trigbasis(x, N)
		end
		return A
	end

	"""
	Fit a trigonometric polynomial to the data ``X = (x_m), F = (f_m)``.
	"""
	function lsqfit(X, F, N)
		A = designmatrix(X, N)
		return A \ F   # this performs A = Q*R, R \ (Q' * F) for us
	end

	
	trigprojeval(x, c) = real(sum(c .* trigbasis(x, (length(c)-1) ÷ 2)))
end

# ╔═╡ fc8495a6-8c50-11eb-14ac-4dbad6baa3c3
md"""
We can explore this situation with a numerical experiment:

Data $M$: $(@bind _M1 Slider(10:10:500; show_value=true))

Degree $N$: $(@bind _N1 Slider(5:100; show_value=true))

Noise $\eta = 10^{p}$; choose $p$: $(@bind _p1 Slider(-5:0; show_value=true))
"""
# $(@bind _eta Slider([0.0001, 0.001, 0.01, 0.1]))

# ╔═╡ 1ba8aa58-8c51-11eb-2d66-775d0fd31747
let N = _N1, M = _M1, σ = 10.0^(_p1), f = x -> 1 / (1 + exp(10*sin(x)))
	Random.seed!(2) # make sure we always produce the same random points
	if M < 2*N+1
		M = 2*N+1
		msg = "M must be >= 2N+1"
	end
	X = 2*π * rand(M)
	F = f.(X) + σ * randn(length(X))
	c = lsqfit(X, F, N)
	xp = range(0, 2π, length = 200)
	plot(xp, f.(xp), lw=4, label = L"f", size = (400, 200),
			title = "N = $N, M = $M", legend = :outertopright,
		 ylims = [-0.3, 1.3])
	P = plot!(xp, trigprojeval.(xp, Ref(c)), lw=2, label = "fit")
	plot!(P, X, F, lw=0, ms=2, m=:o, c=:black, label = "")
end

# ╔═╡ 3d14680a-98df-42e2-9734-8a5ba27f5f6d


# ╔═╡ 48175a0c-8e87-11eb-0f42-e9ca0f676e87
md"""
### WARNING

A proper treatment of this subject requires an in-depth computational statistics course. We cannot go into all the subtleties that are required here, such as cross-validation, regularisation, model selection. ... But we are in the age of data science and I *highly* recommend taking some advanced courses on these topics!

Here, we will only explore some approximation-theoretic perspectives on balancing available data with choice of model, i.e. polynomial degree. There is obviously a non-trivial relationship between these. Secondly we will explore what we can say about optimal choice of data points in order to learn about how to choose sampling points if we could choose as we wish.
"""

# ╔═╡ e6cf2c86-9043-11eb-04a7-f1367ad64b6b
md"""
## §3.3 Equispaced Data

As a warm-up we first explore the case when we get to *choose* the training datapoints. We already discussed that without *a priori* knowledge of the function to be fitted we should choose equi-spaced data points. Specifically, let us choose 
```math
	x_m = m\pi/M, \qquad m = 0, \dots, 2M-1.
```
We then evaluate the target function to obtain the training data, ``f_m := f(x_m)`` and minimize the loss 
```math
L(\boldsymbol{c}) = \sum_{m = 1}^{2M-1} \bigg| \sum_{k = -N}^N c_k e^{i k x_m} - f_m \bigg|^2
``` 
to obtain the parameters. With ``M = N`` this is equivalent to trigonometric interpolation (easy exercise), where we even have a fast solver available (FFT). 
"""

# ╔═╡ 82d74ec2-92a1-11eb-0d58-4bb674a8640e
md"""
### Analysis of the lsq system 

Let us write out the loss function explicitly, but weighted, 
```math
	L_M(c) = \frac{1}{2M} \sum_{m = 0}^{2M-1} \big| t_N(x_m) - f(x_m) \big|^2
```
where ``t_N \in \mathcal{T}_N`` has coefficients ``\boldsymbol{c} = (c_k)_{k = -N}^N``. Note that this is the periodic trapezoidal rule approximation of
```math
	L_\infty(c) = \int_{0}^{2\pi} \big|t_N(x) - f(x) \big|^2 \,dx.
```
We know that minimizing ``L_\infty(c)`` gives the best possible ``L^2`` approximation, i.e., if ``c = \arg\min L_\infty`` then 
```math 
	t_N = \Pi_N f = \sum_{k = -N}^N \hat{f}_k e^{i k x}
``` 
and 
```math
	\|t_N - f \|_{L^2} \leq \| t_N' - f  \|_{L^2} \qquad \forall t_N' \in \mathcal{T}_N'.
```
"""

# ╔═╡ 879e9596-90a8-11eb-23d6-935e367eeb17
md"""
Because ``L_M`` is an approximation ``L_\infty`` we can intuit that ``t_N = \arg\min L_M`` will be "close" to ``\Pi_N f`` in some sense. The following result is makes this precise:

**Proposition:** Let ``f \in C_{\rm per}``, and ``t_{NM} = \arg\min_{\mathcal{T}_N} L_M`` with ``M > N``, then ``t_N = \Pi_N I_M f``. In particular, 
```math
	\|t_N - f \|_{L^2} 
	 \leq 
	\| \Pi_N f - f \|_{L^2} + \| \Pi_N (I_M f - f) \|_{L^2}.
```

**Proof:** 
```math 
\begin{aligned}
	L_M(t_N) 
	&= 
	\frac{1}{2M} \sum_{m = 0}^{2M-1} |t_N(x_m) - f(x_m)|^2
	\\ &= 
	\frac{1}{2M} \sum_{m = 0}^{2M-1} |t_N(x_m) - I_M f(x_m)|^2
	\\ &= 
	\| t_N - I_M f \|_{L^2}^2
\end{aligned}
```
by applying for the discrete and then the semi-discrete Plancherel theorem. 
This means that minimising ``L_M`` actually minimizes the ``L^2``-distance to the trigonometric interpolant ``I_M f``, but with ``M > N``, i.e., 
```math
	t_N = \Pi_N I_M f.
```
The stated result now follows easily. ``\square``
"""

# ╔═╡ 3abcfea6-92a2-11eb-061a-d9752403eff8
md"""
**Remark:** We can study the "error term" ``\| \Pi_N (f - I_M f) \|_{L^2}`` in more detail, but it should be intuitive that for ``M \gg N`` it will be much smaller than the best-approximation term ``\| f - \Pi_N f \|_{L^2}``. Importantly, this gives us an overarching strategy to consider when we perform least-squares fits: find an approxmiation error concept, ``\| \bullet - f\|_{L^2}`` that is independent of the data ``(x_m, f_m)`` and which is minimized up to a higher-order term.
"""

# ╔═╡ c4b66e46-90a7-11eb-199e-cded424c7020
md"""
### Fast solver

We now turn to the implementation of the least squares system that we studied in the previous section. While the naive implementation requires ``O(M N^2)`` cost of the QR factorisation, we can use the orthogonality of the trigonometri polynomials to replace this with a matrix multiplication of ``O(MN)`` cost. But the representation 
```math 
	t_{NM} = \Pi_N I_M f 
```
gives us a clue for an even faster O(M \log M) algorithm: 
* Compute ``I_M f`` via the FFT; ``O(M \log M)`` operations
* Obtain ``\Pi_N I_M f`` by only retaining the coefficients ``k = -N, \dots, N``; ``O(N)`` operations.
"""

# ╔═╡ 69fa7e00-90ae-11eb-0681-2d9295ae5368
begin
	function approxL2proj(f, N, M)
		# generate the sample points
		X = range(0, 2π - π/M, length = 2M)
		# the k-grid we obtain from the
		# degree-M trigonometric interpolant
		Km = [0:M; -M+1:-1]
		# contruc the trigonometric interpolant I_M f
		F̂m = fft(f.(X)) / (2M)
		# and find the subset defining Π_N I_M f
		F̂n = [ F̂m[1:N+1]; F̂m[end-N+1:end] ]
	end 
	
	L2err(f, F̂; xerr = range(0, 2π, length=31 * length(F̂))) = 
		sqrt( sum(abs2, f.(xerr) - trigprojeval.(xerr, Ref(F̂))) / length(xerr) )
				
end

# ╔═╡ e3315558-90b5-11eb-3510-e327a6c2d209
md"""
We are now ready to run some numerical tests to confirm our theory. We pick two examples from Lecture 1: 
```math
\begin{aligned}
f_4(x) &= |\sin(2x)|^3 \\
f_7(x) &= \frac{1}{1 + 10*\sin(x)^2} \\
\end{aligned}
```
In truth there is little to explore here, the lsq solutions perform extremely well, even for a low number of training points. Indeed, the trigonometric interpolant itself already comes surprisingly close to the ``L^2``-best approximation. But let us remember that this was just a warm-up case!
"""

# ╔═╡ 8bd7b6fe-91d1-11eb-2d14-134257fa2878
begin
	f4(x) = abs(sin(x))^3
	f7(x) = 1 / (1.0 + 10*sin(x)^2)
	flabels = [L"f_4", L"f_7"]
end;

# ╔═╡ 7f9b5750-91d2-11eb-32ee-6dc5b74d9c0e
let f = f7, NN = 5:5:60, MM = 2*NN
	err = [ L2err(f, approxL2proj(f, N, M)) for (N, M) in zip(NN, MM) ]
	plot(NN, err, lw=3, label = L"f_7", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 yscale = :log10, size = (350, 230), title = L"f_7~~{\rm analytic}")
	plot!(NN[4:8], exp.(- 1/sqrt(10) * NN[4:8]), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 879bf380-91d5-11eb-0b46-85b15d8b2826
let f = f4, NN = (2).^(3:10), MM = 2 * NN
	err = [ L2err(f, approxL2proj(f, N, M)) for (N, M) in zip(NN, MM) ]
	plot(NN, err, lw=3, label = L"f_4", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 xscale = :log10, yscale = :log10, size = (350, 200), 
		 title = L"f_4 \in C^{2,1}")
	plot!(NN[4:end], NN[4:end].^(-3.5), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 8b7e2280-8e87-11eb-0448-9f3a5acf6032
md"""

## §3.3 Random training points

We now return to the case we experimented with at the beginning of this lecture, choose ``x_m \sim U([0, 2\pi])``, iid. While this appears to be a natural choice of random samples, specific applications might lead to different choices. However, it is crucial here. The reason is the following: 

We are trying to approximate 
```math
	f(x) \approx \sum_k c_k B_k(x)
```
where ``B_k`` is a basis of function on ``[0, 2\pi]``. Suppose we sample ``x_m \sim \rho dx`` where ``\rho`` is a general probability density on ``[0, 2\pi]``. The following theory requires that ``\{B_k\}`` is an orthonormal basis with respect to that measure, i.e., 
```math
	\int_{0}^{2\pi} B_k(x) B_{l}(x) \rho(x) dx = \delta_{kl}
```
Since the trigonometric polynomial basis is orthonormal w.r.t. the standard ``L^2``-inner product, i.e., ``\rho(x) = 1/(2\pi)`` we will also sample with respect to that measure. 

If ``x_m`` are distributed according to a different distribution then we need to adjust our basis. We can explore this in the assignment. 
"""

# ╔═╡ d8a55fd0-91dc-11eb-033c-0b10126c5ac7
md"""
The following results are taken from 

	Albert Cohen, Mark A Davenport, and Dany Leviatan. On the stability and accuracy of least squares approximations. Found. Comut. Math., 13(5):819–834, October 2013.

Our first result states stability of the least squares system with high probability:

**Theorem [Stability]:** Let ``x_m \sim U(0, 2\pi)``, idd, and ``A_{mk} = e^{i k x_m}`` then 
```math
	\mathbb{P}\big[ \| A^* A - I \|_{\rm op} \geq 1/2 \big] 
	\leq 2 N \exp\big( - 0.1 M N^{-1} \big)
```
This result is readily interpreted: if ``M \gg N`` then the normal equations are well-conditioned with high probability. In particular this also means that the design matrix ``A`` has full rank and that its ``R``-factor is also well-conditioned.

The second result states a resulting near best approximation error estimate: 

**Theorem [Error]:** Let ``x_m \sim U(0, 2\pi)``, iid and let ``t_{NM}`` denote the resulting degree ``N`` least squares approximant. There exists a constant ``c`` such that, if 
```math
	N \leq \frac{c}{1+r} \frac{M}{\log M} 
```
then 
```math
	\mathbb{E}\big[ \| f - t_{NM} \|_{L^2}^2 \big] 
	\leq
	(1+o(M)) \|f - \Pi_N f \|_{L^2}^2 + 2 \| f \|_{L^\infty}^2 M^{-r}.
```

Similarly as in our introductory example, this result gives us a best-approximation error up to an additional term that depends on how many training points we are given. To properly appreciate it we can show that it implies the following result: 
"""

# ╔═╡ fb0081a6-91df-11eb-00a9-a9deb7581813
md"""
* If ``f`` is continuous(ly differentiable) but not analytic then we expect that ``\|f - \Pi_N f \|_{L^2} \approx N^{-q}`` for some ``q``. In this case, choosing ``M \geq c N \log N`` with any ``c > `` we obtain that ``M^{-r} \lesssim (N \log N)^{-r} \ll N^{-q}``, i.e., 
```math 
	\mathbb{E}\big[ \| f - t_{NM} \|_{L^2}^2 \big]  \lesssim N^{-q}.
```

* If ``f`` is analytic then this is a little trickier: the idea is to choose ``N = c (M / \log M)^a`` for some ``a > 0`` which leads to ``r = c' (M / \log M)^{1-a}`` and hence 
```math
	M^{-r} = \exp\Big( - r \log M \Big) = 
	\exp\Big( - c' M^{1-a} (\log M)^{a} \Big)
```
To ensure this scales the same as 
```math
	\rho^{-N} = e^{-\alpha N} = \exp\Big( - \alpha c (M/\log M)^a \Big)
``` 
we must choose ``1-a = a`` i.e. ``a = 1/2``. That is, we obtain that for a suitable choice of ``c``, and ``N = c (M / \log M)^{1/2}`` we recover the optimal rate 
```math
	\mathbb{E}\big[ \| f - t_{NM} \|_{L^2}^2 \big] \lesssim \rho^{-N}.
```
"""

# ╔═╡ 97a2f8fe-91e0-11eb-2721-9395f949cc48
md"""
Let us again test these predictions numerically.
"""

# ╔═╡ b44e273a-91e0-11eb-1b99-3b20b207513d
begin 
	function lsqfit_rand(f::Function, N::Integer, M::Integer) 
		X = 2*π*rand(M)
		return lsqfit(X, f.(X), N) 
	end 

	L2err_rand(f, N, M; xerr = range(0, 2π, length=31*M)) = 
		sqrt( sum(abs2, f.(xerr) - trigprojeval.(xerr, Ref(lsqfit_rand(f, N, M)))) / (2*M) )
end

# ╔═╡ c479ca34-91e1-11eb-191e-2b8d5f2e8211
let f = f4, NN = (2).^(3:9), MM1 = 2 * NN .+ 1,
						     MM2 = 3 * NN, 
							 MM3 = 2 * ceil.(Int, NN .* log.(NN))
	
	err1 = L2err_rand.(f, NN, MM1)
	err2 = L2err_rand.(f, NN, MM2)
	err3 = L2err_rand.(f, NN, MM3)
	plot(NN, err1, lw=2, label = L"M = 2N + 1", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 xscale = :log10, yscale = :log10, size = (450, 250), 
		 title = L"f_4 \in C^{2,1}", legend = :outertopright)
	plot!(NN, err2, lw=2, label = L"M = 3N")
	plot!(NN, err3, lw=2, label = L"M = 2N \log N")		
	plot!(NN[4:end], NN[4:end].^(-3.5), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 96329376-91e0-11eb-0c0e-5f80723255f8
let f = f7, NN = 5:5:40, MM1 = 2 * NN .+ 1,
						MM2 = 3*NN,  
						MM3 = 2 * ceil.(Int, NN.^1.5)
	err1 = L2err_rand.(f, NN, MM1)
	err2 = L2err_rand.(f, NN, MM2)
	err3 = L2err_rand.(f, NN, MM3)
	plot(NN, err1, lw=2, label = L"M = 2N + 1", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 yscale = :log10, size = (450, 250), 
		 title = L"f_7 ~~{\rm analytic}", legend = :outertopright)
	plot!(NN, err2, lw=2, label = L"M = 3N")
	plot!(NN, err3, lw=2, label = L"M = 2 N^{3/2}")		
	plot!(NN[4:end], exp.(-1/sqrt(10) * NN[4:end]), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 20b6fcfe-93f2-11eb-1b8d-852aeb86d4f8
md"""
In this final example we see a clear gap between theory and practise. Is it just pre-asyptotics? Something about this specific example? Maybe the theory isn't sharp? Or maybe the specific function we are considering has additional properties?
"""

# ╔═╡ d9a37ca0-9b46-42b4-84ac-e012a8ec4b1e
md"""

## Regularisation (If we have time?)



"""

# ╔═╡ 60c2146e-d85c-435d-9262-c5dbc52fcd00
md"""
Let us extend our problem from the start of the lecture: 

Data $M$: $(@bind _Mr Slider(10:10:500; show_value=true))

Degree $N$: $(@bind _Nr Slider(5:100; show_value=true))

Noise $\eta = 10^{p}$; choose $p$: $(@bind _pr Slider(-5:0; show_value=true))

Regularisation $\lambda = 10^q$; choose $q$: $(@bind _qr Slider(-10:0.1:10; show_value=true))
"""

# ╔═╡ 89f791bf-c1ff-40a7-a4ce-66b5539efccd
begin
	function reglsqfit(X, F, N, Γ)
		A = [ designmatrix(X, N); Γ ]
		return qr(A) \ [F; zeros(size(A, 2))]   # this performs the  R \ (Q' * F) for us
	end
end

# ╔═╡ 18c034eb-fa16-486c-8415-46858d17f274
let N = _Nr, M = _Mr, σ = 10.0^(_pr), λ = 10.0^(_qr), f = x -> 1 / (1 + exp(10*sin(x)))
	Random.seed!(2) # make sure we always produce the same random points
	if M < 2*N+1
		M = 2*N+1
		msg = "M must be >= 2N+1"
	end
	X = 2*π * rand(M)
	F = f.(X) + σ * randn(length(X))
	c = lsqfit(X, F, N)
	cr = reglsqfit(X, F, N, λ*I)
	# cr = reglsqfit(X, F, N, λ * Diagonal((kgridproj(N)).^2))
	xp = range(0, 2π, length = 200)
	plot(xp, f.(xp), lw=4, label = L"f", size = (450, 200),
			title = "N = $N, M = $M",
		    ylims = [-0.3, 1.3], legend = :outertopright)
	plot!(xp, trigprojeval.(xp, Ref(c)), lw=2, label = "fit")
	plot!(xp, trigprojeval.(xp, Ref(cr)), lw=2, label = "regfit")
	plot!(X, F, lw=0, ms=2, m=:o, c=:black, label = "")
end

# ╔═╡ 4d98a913-c17d-4d9c-9708-5be743a4b944
md"""
A more quantitative approach: fit to a training set, but then measure error on a test set. 
"""

# ╔═╡ 73e88b3f-ed14-4919-bf61-a8b77510f7f9
md"""
Data $M$: $(@bind _Mr2 Slider(10:10:500; show_value=true))

Degree $N$: $(@bind _Nr2 Slider(5:100; show_value=true))

Noise $\eta = 10^{p}$; choose $p$: $(@bind _pr2 Slider(-5:0; show_value=true))
"""

# ╔═╡ 2fcce329-1ed3-4681-b571-b5d4c544dbf5
let N = _Nr2, M = _Mr2, σ = 10.0^(_pr2), f = x -> 1 / (1 + exp(10*sin(x)))
	
	Random.seed!(2) # make sure we always produce the same random points
	Xtrain = 2*π * rand(M)
	Ftrain = f.(Xtrain) + σ * randn(M)
	Xtest = 2*π * rand(M)   # this is atypical, normally we have fewer test points 
	Ftest = f.(Xtest)
	
	function testrmse(λ) 
		cr = reglsqfit(Xtrain, Ftrain, N, λ * Diagonal((kgridproj(N)).^2))
		fit = trigprojeval.(Xtrain, Ref(cr))
		prediction = trigprojeval.(Xtest, Ref(cr))
		rmsetest = norm(prediction - Ftest) / sqrt(length(Ftest))
		rmsetrain = norm(fit - Ftrain) / sqrt(length(Ftrain))
		return rmsetest, rmsetrain 
	end
	
	LAM = 0.1.^(-3:.1:10)
	RMSE = testrmse.(LAM)
	RMSE_test = getindex.(RMSE, 1)
	RMSE_train = getindex.(RMSE, 2)
	plot(LAM, RMSE_train, lw = 3, label = "train",
			xscale = :log10, yscale = :log10, size = (350, 250),
			legend = :bottomright)
	plot!(LAM, RMSE_test, lw = 3, label = "test")
end

# ╔═╡ 76f189b8-93f2-11eb-259d-d52fea279464
md"""

## Outlook: Algebraic Polynomials

Consider a non-periodic version of our favourite example 
```math
	f(x) = \frac{1}{1 + x^2}, \qquad x \in [-1, 1].
```
Since we are no longer on a periodic domain, let us use algebraic instead of trigonometric polynomials to approximate it, i.e. we seek a polynomial 
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

# ╔═╡ e611030a-93f2-11eb-3581-1b3f4b41360a
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

# ╔═╡ 530b2eb6-93f5-11eb-2c54-6317369a6b21
md"""
Next lecture will cover a range of "random topics" that I won't cover in much (or any) depth but which are both fun and important to have seen once. The first of these will be to explain how all our ideas from trigonometric approximation do carry over to algebraic approximation as long as we take the right perspective. 
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
FFTW = "~1.4.6"
ForwardDiff = "~0.10.25"
LaTeXStrings = "~1.3.0"
Plots = "~1.26.0"
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
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

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

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3f7cb7157ef860c637f3f4929c8ed5d9716933c6"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.7"

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
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "23d109aad5d225e945c813c6ebef79104beda955"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.26.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "de893592a221142f3db370f48290e3a2ef39998f"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.4"

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
git-tree-sha1 = "995a812c6f7edea7527bb570f0ac39d0fb15663c"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.1"

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
git-tree-sha1 = "74fb527333e72ada2dd9ef77d98e4991fb185f04"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.1"

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
# ╟─74479a7e-8c34-11eb-0e09-73f47f8013bb
# ╟─8728418e-8c34-11eb-3313-c52ecadbf252
# ╟─9202b69e-9013-11eb-02a2-2f1c8e2fcc0c
# ╟─73983438-8c51-11eb-3142-03410d610022
# ╠═d08fb928-8c53-11eb-3c35-574ef188de6b
# ╟─fc8495a6-8c50-11eb-14ac-4dbad6baa3c3
# ╟─1ba8aa58-8c51-11eb-2d66-775d0fd31747
# ╠═3d14680a-98df-42e2-9734-8a5ba27f5f6d
# ╟─48175a0c-8e87-11eb-0f42-e9ca0f676e87
# ╟─e6cf2c86-9043-11eb-04a7-f1367ad64b6b
# ╟─82d74ec2-92a1-11eb-0d58-4bb674a8640e
# ╟─879e9596-90a8-11eb-23d6-935e367eeb17
# ╟─3abcfea6-92a2-11eb-061a-d9752403eff8
# ╟─c4b66e46-90a7-11eb-199e-cded424c7020
# ╠═69fa7e00-90ae-11eb-0681-2d9295ae5368
# ╟─e3315558-90b5-11eb-3510-e327a6c2d209
# ╠═8bd7b6fe-91d1-11eb-2d14-134257fa2878
# ╠═7f9b5750-91d2-11eb-32ee-6dc5b74d9c0e
# ╠═879bf380-91d5-11eb-0b46-85b15d8b2826
# ╟─8b7e2280-8e87-11eb-0448-9f3a5acf6032
# ╟─d8a55fd0-91dc-11eb-033c-0b10126c5ac7
# ╟─fb0081a6-91df-11eb-00a9-a9deb7581813
# ╟─97a2f8fe-91e0-11eb-2721-9395f949cc48
# ╠═b44e273a-91e0-11eb-1b99-3b20b207513d
# ╠═c479ca34-91e1-11eb-191e-2b8d5f2e8211
# ╟─96329376-91e0-11eb-0c0e-5f80723255f8
# ╟─20b6fcfe-93f2-11eb-1b8d-852aeb86d4f8
# ╟─d9a37ca0-9b46-42b4-84ac-e012a8ec4b1e
# ╟─60c2146e-d85c-435d-9262-c5dbc52fcd00
# ╠═89f791bf-c1ff-40a7-a4ce-66b5539efccd
# ╠═18c034eb-fa16-486c-8415-46858d17f274
# ╟─4d98a913-c17d-4d9c-9708-5be743a4b944
# ╟─73e88b3f-ed14-4919-bf61-a8b77510f7f9
# ╠═2fcce329-1ed3-4681-b571-b5d4c544dbf5
# ╟─76f189b8-93f2-11eb-259d-d52fea279464
# ╟─e611030a-93f2-11eb-3581-1b3f4b41360a
# ╟─530b2eb6-93f5-11eb-2c54-6317369a6b21
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
