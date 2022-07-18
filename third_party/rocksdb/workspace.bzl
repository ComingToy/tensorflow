load("//third_party:repo.bzl", "tf_http_archive")
def repo():
    tf_http_archive(
        name = "rocksdb",
        build_file = "//third_party/rocksdb:rocksdb.BUILD",
        strip_prefix = "rocksdb-6.24.2",
        urls = [ "https://mirror.tensorflow.org/facebook/rocksdb/archive/refs/tags/v6.24.2.tar.gz","https://github.com/facebook/rocksdb/archive/refs/tags/v6.24.2.tar.gz"],
	sha256 = "cdecddd9dff271d8087b47f239e4f0dc96b6ff9b7da4dd7ccbbb109bd5db43d1"
    )
