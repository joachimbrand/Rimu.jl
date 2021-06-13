## Code testing

The script `runtest.jl` in the `test/` folder contains tests of the code. To run the test simply run the script from the Julia REPL or run
```
Rimu$ julia test/runtest.jl
```
from the command line.

More tests should be added over time to test core functionality of the code. To add new tests, directly edit the file `runtest.jl`.

### Automated testing with GitHub Actions

GitHub Actions are set up to run the test script automatically on the GitHub cloud server every time a new commit to the master branch is pushed to the server. The setup for this to happen is configured in the file
`actions.yml` in the `Rimu/.github/workflows` folder.

### Modifying the `Project.toml` file

In order for the testing code to be able to run on the cloud server, external packages that are accessed in the
code with `using` or `import` need to be installed first.
This is done in the script `actions.yml` via the package manager, based on the information contained in the file
`test/Project.toml`. More packages can be added to this file using the package manager in the following way: Say we want to install the package `DelimitedFiles`. At the Julia REPL, type the following:
```julia-repl
julia> cd("test")
julia> ]
(v1.0) pkg> activate .
(test) pkg> add DelimitedFiles
```
This will a new line to the file `Project.toml` with the name of the package and the corresponding uuid. When Pipelines now runs the commands in yml script, it will install the package `DelimitedFiles` before running the `runtest.jl` script.
