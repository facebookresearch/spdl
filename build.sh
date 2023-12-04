(
    set -eux
    cmake -B .build -S . -DCMAKE_INSTALL_PREFIX=./artifacts
    cmake --build .build
    cmake --install .build
)
