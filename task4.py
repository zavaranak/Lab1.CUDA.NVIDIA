import numpy as np
from numba import cuda
from PIL import Image
import time

def generate_palette(image):
    pixels = np.array(image).reshape(-1, 3)  
    palette = np.unique(pixels, axis=0)  # unique colors array (RGB)
    codes = np.arange(len(palette), dtype=np.uint8) # code (index) for each color
    return palette, codes


@cuda.jit
def replace_colors_with_codes(image, colors, codes, output):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        r = image[x, y, 0]
        g = image[x, y, 1]
        b = image[x, y, 2]
        for i in range(colors.shape[0]):
            if colors[i, 0] == r and colors[i, 1] == g and colors[i, 2] == b:
                output[x, y] = codes[i]
                break



def decode_image(coded_image, palette_colors):
    decoded_image = np.zeros((coded_image.shape[0], coded_image.shape[1], 3), dtype=np.uint8)
    for i in range(coded_image.shape[0]):
        for j in range(coded_image.shape[1]):
            decoded_image[i, j] = palette_colors[coded_image[i, j]]
    return decoded_image


def main():
    # Load Image
    image_path = "24_bit_image.bmp"
    image = Image.open(image_path)
    image_array = np.array(image)
    # Generate palette
    palette,color_codes = generate_palette(image)
    print(f"Палитра содержит {len(palette)} цветов.")
    # print(unique_colors)
    # print(color_codes)
    

    # copy arrays to GPU
    unique_colors_at_gpu = cuda.to_device(palette)
    color_codes_at_gpu = cuda.to_device(color_codes)    
    image_at_gpu = cuda.to_device(image_array)
    output_at_gpu = cuda.device_array((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

    # Define grid and block sizes x, y
    threads_per_block = (32, 32)
    blocks_per_grid_x = (image_array.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image_array.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    print(image_array.shape)
    print(blocks_per_grid)

    # Запуск ядра на GPU
    start_time = time.time()
    replace_colors_with_codes[blocks_per_grid, threads_per_block](image_at_gpu, unique_colors_at_gpu,color_codes_at_gpu, output_at_gpu)
    end_time = time.time()
    print(f"Замена цветов на коды выполнена за {end_time - start_time:.4f} секунд.")

    # Copy coded image from GPU
    coded_image = output_at_gpu.copy_to_host()

    # Decode coded image
    decoded_image = decode_image(coded_image, palette)
    print(coded_image.shape)

    # Compare coded_image and decoded_image
    if np.array_equal(image_array, decoded_image):
        print("Правильно.")
        # print(coded_image)
        # print(decoded_image)
    else:
        print("Неправильно.")

    # Сохранение результатов
    Image.fromarray(coded_image).save("coded_image.bmp")
    Image.fromarray(decoded_image).save("decoded_image.bmp")

if __name__ == "__main__":
    main()