load("@com_google_protobuf//:protobuf.bzl", "proto_gen")

proto_gen(
    name = "pslite_meta_proto_gen",
    srcs = ["src/meta.proto"],
    outs = ["src/meta.pb.cc", "src/meta.pb.h"],
    gen_cc = 1,
    protoc = "@com_google_protobuf//:protoc",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pslite_meta_proto_cc",
    srcs = ["src/meta.pb.cc"],
    hdrs = ["src/meta.pb.h"],
    deps = [
        "@com_google_protobuf//:protobuf_headers",
    ],
    visibility = ["//visibility:public"]
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
