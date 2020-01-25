mkdir -p tensorflow_headers2
rsync bazel-bin/tensorflow/**/*.h --recursive --relative tensorflow_headers2
rsync bazel-bin/tensorflow/**/*.pb.h --recursive --relative tensorflow_headers2
mkdir -p tensorflow_headers
rsync tensorflow/./**/*.h --recursive --relative tensorflow_headers
mv tensorflow_headers2/bazel-bin tensorflow_headers/tensorflow

cp third_party/./eigen3/unsupported --recursive --relative tensorflow_headers/third_party
