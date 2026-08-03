include_guard(GLOBAL)

# Version arguments are compatibility lower bounds. Toolchain selection and package locations belong to the caller.
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(NCCL 2.14.3 REQUIRED)
