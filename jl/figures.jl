
texdir = @__DIR__()[1:end-2] * "tex/figs/"
jldir = @__DIR__() * "/figs/"

## FIGURE: illustration of the trapezoidal rule
using PyPlot
xkcd()
# http://nbviewer.jupyter.org/gist/cmundi/9041366
fig = figure(figsize=(5,5));
ax = axes()
x = linspace(0.5, 1.0, 100)
y = [0.55, 0.95]
p = plot(
         [y[1],y[1]], [0.0,1-y[1]^2], "k-",
         [y[2],y[2]], [0.0,1-y[2]^2], "k-",
         x,1-x.^2, "b",
         y, 1-y.^2, "r",
          )
xticks([])
yticks([])
ax[:set_xlim]([0.5, 1.0])
ax[:set_xlim]([0.25, 1.2])

annotate("error = area between curves",
         xy=[0.75, 1-0.75^2 - 0.03],
         arrowprops=Dict("arrowstyle"=>"->"),
         xytext=[0.1,0.1])

fig[:canvas][:draw]()
savefig(jldir * "/trapezoidal_rule.pdf")
