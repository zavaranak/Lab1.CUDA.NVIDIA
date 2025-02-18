from numba import cuda
print(cuda.detect())  # Detects available CUDA devices
print(cuda.runtime.get_version())  # Checks CUDA runtime version