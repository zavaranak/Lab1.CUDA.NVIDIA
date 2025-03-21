# Лабораторная по Nvidia CUDA 
## Ознакомительное задание 
### 1. Программа, поэлементно вычитающая векторы. 
    a. Константно заданный вектор, то есть вектор из известного на стадии компиляции количества элементов (как в примере, сгенерированном при создании CUDA-проекта в Visual Studio). Количество элементов не превышает максимальный размер блока потоков – 1024 элемента. 
    b. Вектор произвольного размера (протестировать до 1 млн. элементов). Размер вводится в качестве параметра во время работы программы. Для каждой лабораторной из перечня Основных заданий нужно сравнить с ЦП-реализацией по скорости выполнения, а также сравнить поэлементно полученные результаты, чтобы подтвердить корректность GPU-ускоренной реализации. Для видеокарты замеряется суммарное время копирования входных данных в видеопамять, выполнения функции-ядра и обратного копирования результирующих данных, но не замеряется время выделения и освобождения памяти. Основные задания 
### 2. Подсчёт количества вхождений каждого символа в тексте. 
    Текст подгружается из файла либо генерируется (на выбор студента). Каждый символ кодируется одним байтом. Программа должна корректно работать хотя бы до размера входных данных 4 млн. символов. Результирующий массив должен выводиться на экран. 
### 3. Перемножение матриц размером n * m и m * n. 
    Должна быть возможность варьирования n и m. Матрицы заданного размера генерируются автоматически. Программа должна корректно работать хотя бы до размера входных данных 2500 * 2500 элементов. 
### Задание повышенной сложности 4. Замена цветов на их коды (применение палитры цветов) на GPU. 
    Входные данные: изображение в формате RGB24, где общее количество цветов не превышает порога в 256 значений. Для получения такого рисунка рекомендуется открыть исходное изображение в редакторе Paint и сохранить его как «256-и цветный рисунок» (в этот момент произойдёт деградация качества изображения). Затем, не закрывая Paint, сохранить изображение как «24-разрядный рисунок». Этот рисунок и будет являться входными данными для вашей программы. Допускается генерация изображения в ходе выполнения программы.
    Предварительно на ЦП потребуется составить палитру, сопоставив каждому встреченному в изображении цвету 3-х байтовому цвету однобайтовый код. Функция-ядро должна сформировать изображение, где все пиксели будут заменены на их коды в соответствии с составленной палитрой. Потребуется также реализовать на ЦП декодирование изображения, то есть обратную замену кодов на соответствующие им пиксели. Для проверки корректности выполнения программы в конце должно выполняться побайтовое сравнение исходного и декодированного изображений.  
    Затем, после отладки базовой реализации, следует реализовать вариант, при котором палитра цветов помещается в константную память, и сравнить время При кодировании изображения с помощью палитры должен применяться более эффективный способ поиска элемента в палитре, нежели линейный поиск (то есть последовательный перебор элементов в массиве). 
## Методические рекомендации для преподавателей 
    Студент должен сначала выполнить ознакомительное задание и сдать его преподавателю, прежде чем приступать к основным заданиям. Выполнение задания повышенной сложности не является обязательным, но позволяет получить более высокую оценку на экзамене либо получить дополнительные баллы при балльной системе в вузе. 
## Критерии оценивания для преподавателей 
    Для каждой из основных задач и для задания повышенной сложности нужно проверить, что решение удовлетворяет следующим условиям: 
    ● содержит реализацию для GPU; 
    ● содержит реализацию для CPU; 
    ● побайтовое сравнение результирующих массивов, сформированных CPU и GPU, чтобы убедиться в корректности реализации; 
    ● реализация должна отрабатывать корректно как для входных данных небольшого объёма (например, 100 элементов), так и для максимального указанного в условиях задачи объёма. 
    Студент должен знать ответы на следующие вопросы по каждой лабораторной работе: 
    ● Какая часть кода выполняется на CPU, а какая на GPU? 
    ● Сколько потоков будут выполнять функцию-ядро? Где в коде высчитывается это 
    значение? 
