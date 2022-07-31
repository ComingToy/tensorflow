load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
   name = "zeromq_all_srcs",
   srcs = glob(["**"]),
   visibility = ["//visibility:public"],
)

cmake(
    name = "zeromq",
    generate_args=["-DBUILD_STATIC=ON", "-DBUILD_SHARED=OFF"],
    lib_source = ":zeromq_all_srcs",
    build_args = ['-j8'],
    out_static_libs = ["libzmq.a"],
    visibility = ["//visibility:public"],
    copts=["-fPIC", "-DGLIBCXX_USE_CXX11_ABI=0"],
)

