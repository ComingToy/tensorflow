load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "feats_generated",
    srcs = ["comm_feats.fbs", "example.fbs", "feature.fbs", "vocab.fbs"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "read_from_db",
    srcs = ["read_from_db.cpp"],
    deps = [
		"@com_google_absl//absl/strings",
		"@com_google_absl//absl/flags:flag",
		"@com_google_absl//absl/flags:parse",
		":feats_generated",
		"@rocksdb//:rocksdb",
    ],
)

cc_binary(
    name = "write_to_db",
    srcs = ["write_to_db.cpp"],
    deps = [
		"@com_google_absl//absl/strings",
		"@com_google_absl//absl/flags:flag",
		"@com_google_absl//absl/flags:parse",
		":feats_generated",
		"@rocksdb//:rocksdb",
    ],
)

cc_binary(
    name = "stat_db",
    srcs = ["stat_db.cpp"],
    deps = [
		"@com_google_absl//absl/strings",
		"@com_google_absl//absl/flags:flag",
		"@com_google_absl//absl/flags:parse",
		":feats_generated",
		"@rocksdb//:rocksdb",
    ],
)

