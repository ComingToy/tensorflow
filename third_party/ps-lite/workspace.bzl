load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "ps-lite",
        branch = "master",
        build_file = "//third_party/ps-lite:ps-lite.BUILD",
        remote= "https://git.conleylee.com/conley/ps-lite.git"
    )
