import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import time

BLOCK_SIZE=(32,32) #32^2 = 1024


# def generate_matrix_by_cpu(n, m):
#     """Генерация матрицы размером n x m со случайными значениями."""
#     return np.random.rand(n, m)

@cuda.jit
def generate_matrix(matrix,n,m,rng_states):
    row,col = cuda.grid(2)
    if row<n and col<m:
        # matrix[row,col] = row+col
        rand_val = cuda.random.xoroshiro128p_uniform_float32(rng_states, row * m + col)
        matrix[row,col]= rand_val

def multiply_matrices_cpu(m1, m2):
    """Умножение двух матриц."""
    print(np.dot(m1, m2))

@cuda.jit
def multiply_matrices(m1,m2,m_result,result_size,size2):
    row,col = cuda.grid(2)
    if row < result_size and col <result_size:
        temp = 0.0
        for i in range(size2):
            temp+=m1[row,i]*m2[i,col]     
        # print(temp)   #Check calculation
        m_result[row,col] = temp
def main():
    # Ввод размеров матриц
    n = int(input("Введите количество строк первой матрицы (n): "))
    m = int(input("Введите количество столбцов первой матрицы (m): "))

    ## Генерация матриц
    # matrix1 = generate_matrix(n, m)
    # matrix2 = generate_matrix(m, n)
    # print(matrix1)
    # print(matrix2)
    ## Copy to GPU 
    # d_m1 = cuda.to_device(matrix1)
    # d_m2 = cuda.to_device(matrix2)

    ## Generate matrix by GPU: grid => block(m,n) => each matrix == 1 thread
    rng_states = create_xoroshiro128p_states(n * m, seed=1)

    block_count_by_n = (n+BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0]
    block_count_by_m = (m+BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0]

    d_m2 = cuda.device_array((m,n),dtype=np.float32)
    d_m1 = cuda.device_array((n,m),dtype=np.float32)

    
    generate_matrix[(block_count_by_n,block_count_by_m),BLOCK_SIZE](d_m1,n,m,rng_states)
    generate_matrix[(block_count_by_m,block_count_by_n),BLOCK_SIZE](d_m2,m,n,rng_states)


    d_m_result = cuda.device_array((n,n),dtype=np.float32)
    ## GPU execution
    block_count = (block_count_by_n,block_count_by_n)
    start_time = time.time()
    multiply_matrices[block_count,BLOCK_SIZE](d_m1, d_m2, d_m_result,n,m)
    end_time = time.time()

    ## Copy result to CPU
    result = d_m_result.copy_to_host()
    print(f"Результат умножения матриц размером {n}x{m} и {m}x{n}:")
    print(result)
    print(f"Время выполнения: {end_time - start_time:.4f} секунд")

    ## Check with CPU
    # m1=d_m1.copy_to_host()
    # m2=d_m2.copy_to_host()
    # multiply_matrices_cpu(m1,m2)

if __name__ == "__main__":
    main()