### Installation Instructions

* Download and install `Julia`, version 1.5 or later should work, though I recommend 1.7.x. [[julialang.org]](https://julialang.org)
* Install course material: In a terminal clone this repository, then change to the new directory
```
git clone git@github.com:cortner/ApxThyApp.git
cd ApxThyApp 
```
* Install `IJulia` to enable Jupyter based Julia notebooks. You can either do this manually: start a Julia REPL, then type `]` to open the package manager, then type `add IJulia`. Or alternatively with a one-line shell command
```
julia -e 'import Pkg; Pkg.add("IJulia")'
```
* Start the Julia REPL, and install the dependencies. This can again be done manually in the package manager via 
```
activate .
resolve
up
```
or again with a one-line shell script: 
```
julia --project=. -e "import Pkg; Pkg.resolve(); Pkg.update()"
```
The final lines of the terminal output should look something like this:
```
Status `~/gits/atshort/Project.toml`
  [a93c6f00] DataFrames v0.22.5
  [7a1cc6ca] FFTW v1.3.2
  [b964fa9f] LaTeXStrings v1.2.0
  [91a5bcdd] Plots v1.10.6
  [c3e4b0f8] Pluto v0.12.21
  [08abe8d2] PrettyTables v0.11.1
  [37e2e46d] LinearAlgebra
```
If you see this, then the installation was likely succesful. It is not unlikely that there will be error messages or warnings related to the installation of `Interact.jl`. More on this below. 

Once the installation is complete you can access the course material as
follows:  
```
julia --project=. -e "import IJulia; IJulia.notebook()"
```
Note that `--project=.` loads the environment specification for this course, which tells Julia which packages to use (the ones we downloaded and installed during the
installation process). Then `import IJulia` loads the `IJulia.jl` package and
`IJulia.notebook()` starts a Jupyter webserver on your local machine. It should automatically open a webbrowser, but if not, then it will print instructions in the terminal on how to open it.

* Finally open one of the notebooks and try to run the cells.