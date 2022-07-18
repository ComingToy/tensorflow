load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
   name = "all_srcs",
   srcs = glob(["**"]),
   visibility = ["//visibility:public"],
)

cmake(
    name = "rocksdb",
    generate_args=["-DWITH_BZ2=OFF -DWITH_LZ4=OFF -DWITH_SNAPPY=OFF -DWITH_ZLIB=ON -DWITH_GFLAGS=OFF -DWITH_ZSTD=OFF -DROCKSDB_BUILD_SHARED=OFF"],
    lib_source = ":all_srcs",
    build_args = ['-j8'],
    out_static_libs = ["librocksdb.a"],
    visibility = ["//visibility:public"],
    out_lib_dir="lib",
    copts=["-fPIC", "-DGLIBCXX_USE_CXX11_ABI=0"],
    linkopts = ['-lm'],
    deps=["@zlib"]
)

