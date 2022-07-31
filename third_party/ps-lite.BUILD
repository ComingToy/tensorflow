load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

tf_proto_library(
    name = "pslite_meta_proto",
    srcs = ["src/meta.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pslite",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/**/*.h", "include/**/**/*.h", "src/*.h"]),
    includes = ["include", "src"],
    deps = [":pslite_meta_proto_cc", "@zeromq//:zeromq"],
    #linkopts = ["-fexceptions"],
    copts = ["-fexceptions"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_kv_app",
    srcs = ["tests/test_kv_app.cc"],
    includes = ["include", "src"],
    deps = [":pslite"]
)
