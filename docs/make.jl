# script to be run from Bitbucket pipelines

include("build.jl") # generates the html pages

# For deploydocs, you need to set up the following TRAVIS_* environment variables.
# withenv(
#     # TRAVIS_REPO_SLUG is the only one that is repository-specific. This is because we are
#     # actually deploying the HTML pages into a different repository.
#     #"TRAVIS_REPO_SLUG" => "bitbucket.org/joachimbrand/joachimbrand.bitbucket.io",
#     "TRAVIS_REPO_SLUG" => "github.com/myang04/rimudoc",
#     "TRAVIS_PULL_REQUEST" => get(ENV, "BITBUCKET_BRANCH", nothing),
#     "TRAVIS_BRANCH" => "feature/github-actions",
#     "TRAVIS_TAG" => get(ENV, "BITBUCKET_TAG", nothing),
#     "TRAVIS_PULL_REQUEST" => ("BITBUCKET_PR_ID" in keys(ENV)) ? "true" : "false",
# ) do

gitcheckbranch = `git rev-parse --abbrev-ref HEAD`

branchname = chomp(read(gitcheckbranch,String))

if branchname == "develop"
    devbranch = "develop"
elseif startswith(branchname, "feature/doc") || branchname == "feature/github-actions"
    devbranch = branchname
else
    devbranch = "master"
end

deploydocs(
    # repo here needs to point to the BitBucket Pages repository
    # repo = "bitbucket.org/joachimbrand/joachimbrand.bitbucket.io.git",
    repo = "github.com/joachimbrand/Rimu.jl.git",
    # BitBucket pages are served from the master branch
    devbranch = devbranch, # default master
    # branch = "master"
    # As BitBucket Pages are shared between all the repositories of a user or a team,
    # it is best to deploy the docs to a subdirectory named after the package
    # dirname = "Rimu.jl",
    push_preview = true, # generate preview version
)
# end
# @info "Documentation updated on https://joachimbrand.bitbucket.io/Rimu.jl/dev/ "
