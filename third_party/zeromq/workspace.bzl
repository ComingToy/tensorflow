load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
def repo():
    http_archive(
        name = "zeromq",
        build_file = "//third_party/zeromq:zeromq.BUILD",
        strip_prefix = "zeromq-4.3.4",
        urls = ["https://github.com/zeromq/libzmq/releases/download/v4.3.4/zeromq-4.3.4.tar.gz"],
        sha256 = "c593001a89f5a85dd2ddf564805deb860e02471171b3f204944857336295c3e5"
    )
