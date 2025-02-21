import numpy as np
from numba import cuda
import time

BLOCK_SIZE = 1024
MAX_VECTOR_SIZE = 1000000  # For variant B

@cuda.jit
def vector_subtraction(d_a, d_b, d_c, size):
    idx = cuda.grid(1)
    if idx < size:
        d_c[idx] = d_a[idx] - d_b[idx]

def cpu_vector_subtraction(a, b, c, size):
    for i in range(size):
        c[i] = a[i] - b[i]

def run_variant(n):
    ## Initialize vectors with random values
    vector1 = np.random.rand(n).astype(np.float32)
    vector2 = np.random.rand(n).astype(np.float32)
    result_cpu = np.zeros(n, dtype=np.float32)
    result_gpu = np.zeros(n, dtype=np.float32)

    ## CPU version
    start_cpu = time.time()
    cpu_vector_subtraction(vector1, vector2, result_cpu, n)
    cpu_time = (time.time() - start_cpu) * 1000

    start_gpu = time.time()
    ##copy to GPU memory 
    gpu_vector1 = cuda.to_device(vector1)
    gpu_vector2 = cuda.to_device(vector2)
    gpu_temp_result = cuda.device_array(n, dtype=np.float32)
    ## GPU operations timing
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    vector_subtraction[num_blocks, BLOCK_SIZE](gpu_vector1, gpu_vector2, gpu_temp_result, n) 
    cuda.synchronize()
    result_gpu = gpu_temp_result.copy_to_host()
    gpu_time = (time.time() - start_gpu) * 1000

    ## Verify and print results
    correct = np.allclose(result_cpu, result_gpu)
    print_results(cpu_time, gpu_time, correct)


    ## Free GPU memory
    cuda.close()

def print_results(cpu_time, gpu_time, correct):
    print("\nРезультаты:")
    print(f"CPU время: {cpu_time:.2f} мс")
    print(f"GPU время (включая копирование): {gpu_time:.2f} мс")
    print("Результат", "верный" if correct else "НЕВЕРНЫЙ")

if __name__ == "__main__":
    while True:
        print("\nВыберите вариант:")
        print("a - Фиксированный размер вектора (1024 элемента)")
        print("b - Произвольный размер вектора (до 1 млн элементов)")

        choice = input("Ваш выбор: ").lower()
    
        if choice == 'a':
            run_variant(BLOCK_SIZE)
            break
        elif choice == 'b':
            n = int(input(f"Введите размер векторов (до {MAX_VECTOR_SIZE}): "))
            if n > MAX_VECTOR_SIZE or n <= 0:
                print("Ошибка: неверный размер!")
            else:
                print(f"Вариант B: Произвольный размер вектора ({n} элементов)")
                run_variant(n)
            break
        else:
            print("Неверный выбор. Попробуйте снова.")