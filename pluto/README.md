
## Approximation Theory and Applications

This github repository contains the course material for a short-course on
approximation theory. I've put this course together primarily to learn the 
material myself, but have since enjoyed teaching it in various formats. The 
style and much of its content are inspired by Nick Trefethen's books on 
Spectral Methods in Matlab, and on Approximation Theory.

### Course Material

The best way to explore the course material is via `Pluto.jl` notebooks,
see more below. For a quick look, one can also read the static HTMLs:

* [00_intro](https://htmlpreview.github.io/?https://github.com/cortner/atashort/blob/main/html/ata_00_intro.jl.html)
* [01_trigpoly](https://htmlpreview.github.io/?https://github.com/cortner/atashort/blob/main/html/ata_01_trigpoly.jl.html)

### Assignments

Assignments are provided as Pluto notebooks in `assignments/Ax.jl`. To complete
an assignment:
* Open Pluto following the instructions below. Make sure that you
start Pluto after activating the `atashort` environment.
* From Pluto, open (e.g.) `assignments/A1.jl`
* At this point you may wish to rename the notebook, e.g. to `A1_studentid.jl` so that you don't overwrite the original (or so you can pull updates via `git pull`, but also so that when you submit the assignment your file can be identified.
* Complete the assignment following the instructions in the assignment notebook.
* Export the completed notebook as an HTML (click on the triangle/circle symbol in the upper right corner), and submit that HTML to your TA.

### Installation Instructions (Terminal)

* Download and install `Julia` version 1.5 or 1.6. [[julialang.org]](https://julialang.org)
* Install course material: (1) Git Option: In a terminal clone this repository, then change to the new directory
```
git clone https://github.com/cortner/atashort.git
cd atashort
```
(2) Alternative via zip-file: At the top of this webpage click on [↓ Code] (green button) then [Download Zip]. This will download a file `main.zip` which will contain the latest version of this repository. Unzip it somewhere on your harddrive, open a terminal, change to the directory where the files are and continue as below.
* Start the Julia REPL, and load the dependencies
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
If you see this, then the installation was likely succesful.

Once the installation is complete you can access the course material as
follows:  Change to directory `atashort`, then
```
julia --project=. -e "import Pluto; Pluto.run()"
```
Note that `--project=.` loads the environment specification which tells
Julia which packages to use (the ones we downloaded and installed during the
installation process). Then `import Pluto` loads the `Pluto.jl` package and
`Pluto.run()` starts a webserver on your local machine. It should automatically
open a webbrowser, but if not, then it will print instructions in the terminal,
something like this:
```
(base) Fuji-2:atshort ortner$ julia --project=. -e "import Pluto; Pluto.run()"

Opening http://localhost:1235/?secret=fKxO12of in your default browser... ~ have fun!

Press Ctrl+C in this terminal to stop Pluto
```
Simply copy-paste the web-address into a browser, then this should load the
Pluto front page. You can now open the sample notebooks to explore Pluto
or open one of the lecture notebooks, e.g. enter `ata_00_intro.jl` into
the text box and click on `Open`.

### Installation Instructions (Without Terminal)

* Download the course material to some local directory, e.g. `~/atashort`
* Open the Julia REPL
* Press `;` to switch to terminal mode and change directory to `~/atashort`
* Press `]` to switch to package manager mode and type
```
activate .
resolve
up
```
This should install all required packages. Once you have done this, you can access the course material as follows:
* Open the Julia REPL
* Press `;` to switch to terminal mode and change directory to `~/atashort`
* Press `]` to switch to package manager mode and type `activate .`
* Press `[backspace]` to switch back to Julia mode and type
```julia
import Pluto; Pluto.run()
```

### Some resources for learning about Julia and Pluto:

* https://julialang.org
* https://juliaacademy.com
* https://juliadocs.github.io/Julia-Cheat-Sheet/
* https://github.com/fonsp/Pluto.jl
* https://www.wias-berlin.de/people/fuhrmann/SciComp-WS2021/assets/nb01-first-contact-pluto.html
* https://computationalthinking.mit.edu/Spring21/

Although you won't need it for this course, I recommend VS Code for serious work with Julia. Atom is still slightly more convenient for some things but most development has now moved to VS Code so unless you are already committed to Atom, I recommend starting with VS Code.

### Required Background

* A first analysis course, i.e. concepts such as limits, continuous functions, limits of continuous functions, differentiable functions.
* A first course in linear algebra.
* Some basic skills in programming and reading code, any language will do. For assignments it should only be required to modify code from the lectures.
* Complex numbers, in particular the [complex exponential exp(i x)](https://en.wikipedia.org/wiki/Euler%27s_formula).
* For some parts of the course familiarity with analytic functions will be assumed.
* Familiarity with Fourier series and a first course in numerical analysis would be beneficial, but are not strictly required.
