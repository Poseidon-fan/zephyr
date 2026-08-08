include_guard(GLOBAL)

# Version arguments are compatibility lower bounds. Toolchain selection and package locations belong to the caller.
find_package(CUDAToolkit REQUIRED)
find_package(NCCL REQUIRED)
