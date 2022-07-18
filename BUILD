load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
exports_files([
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
		"//tensorflow/tools/pip_package:build_pip_package": "",
		"//tensorflow/tools/dataset:write_to_db": "",
		"//tensorflow/tools/dataset:read_from_db": "",
    },
)
