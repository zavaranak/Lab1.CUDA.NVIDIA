import numpy as np
from numba import cuda
import time

input_file = "task2.input.txt"

@cuda.jit
def count_chars_kernel(text, freq):
    idx = cuda.grid(1)
    if idx < len(text):
        # print(idx)
        cuda.atomic.add(freq, text[idx], 1)

def count_chars(text):
    # Text to Byte (ASCII)
    text_array = np.frombuffer(text.encode('ascii'), dtype=np.uint8)
    
    # 256 символ из ASCII таблицы
    freq = np.zeros(256, dtype=np.uint32)
    
    # Copy to GPU
    d_text = cuda.to_device(text_array)
    d_freq = cuda.to_device(freq)
    
    # Запускать вычислении GPU 
    threads_per_block = 1024  
    blocks_per_grid = (len(text_array) + threads_per_block - 1) // threads_per_block  # Количество блоков
    count_chars_kernel[blocks_per_grid, threads_per_block](d_text, d_freq)
    
    # Copy обратно на CPU
    freq = d_freq.copy_to_host()
    return freq

# Функция для загрузки текста из файла
def load_text_from_file(filename):
    with open(filename, 'r', encoding='ascii') as file:
        return file.read()

# Функция для генерации текста
def generate_text(size):
    return ''.join(np.random.choice(list('ABCDEFGHJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz;.?1234567890'), size))

if __name__ == "__main__":
    # Выбор способа получения текста
    choice = input("Выберите способ загрузки текста (1 - из файла, 2 - сгенерировать): ")
    
    if choice == '1':
        print("Загрузка текста из файла ""task2.input.txt""")
        text = load_text_from_file(input_file)
    elif choice == '2':
        size = int(input("Введите размер текста (до 4 млн символов): "))
        if size > 4_000_000:
            print("Размер текста превышает 4 млн символов. Установлено значение 4 млн.")
            size = 4_000_000
        text = generate_text(size)
    else:
        print("Неверный выбор. Завершение программы.")
        exit()

    # Подсчёт частот символов
    start_time = time.time()
    freq = count_chars(text)
    end_time = time.time()

    # Вывод результата
    print("Частоты символов:")
    for i, count in enumerate(freq):
        if count > 0:
            print(f"Символ '{chr(i)}': {count} раз")

    print(f"Время выполнения: {end_time - start_time:.4f} секунд")