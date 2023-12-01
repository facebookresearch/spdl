(
    set -eux
    mkdir -p .build
    cd .build
    cmake .. -DCMAKE_INSTALL_PREFIX=./install
    cmake --build .
    cmake --install .
)
