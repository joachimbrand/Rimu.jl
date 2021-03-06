## Documentation generation

We are using [`Documenter.jl`](https://github.com/JuliaDocs/Documenter.jl) to generate the documentation web site based on markdown files stored in `docs/src`. Please help keeping the documentation up-to-date by editing the markdown files! For instructions on how to write appropriate documentation please refer to the relevant chapter in the Julia [documentation](https://docs.julialang.org/en/v1/manual/documentation/) and the `Documenter.jl` [documentation](https://juliadocs.github.io/Documenter.jl/latest/).

### Generating the documentation web site

The documentation pages can be generated by running the build script by typing
```
Rimu$ julia docs/build.jl
```
on the shell prompt from the `Rimu/` folder. A complete image of the static documentation web site will be generated in the folder `docs/build/`. It can be viewed locally by pointing a web browser to file `docs/build/index.html`, or by deploying it to the Bitbucket pages web server.

### Automatic documentation generation and deployment

Our documentation is hosted on a Bitbucket pages web site. The documentation web site can be built and deployed automatically with BitBucket pipelines. This needs to be set up with an appropriate script in the file `bitbucket-pipelines.yml` in the project folder, where triggers for this to happen can be defined. In the current set up, a new documentation web site is generated and deployed whenever someone pushes to the master branch on the BitBucket server. The updated documentation can then be accessed at [https://joachimbrand.bitbucket.io/Rimu.jl/dev/](https://joachimbrand.bitbucket.io/Rimu.jl/dev/).
