# Проект 0. Угадай число

## Оглавление  
[1. Описание проекта](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#Описание-проекта)  
[2. Какой кейс решаем?](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#Какой-кейс-решаем)  
[3. Краткая информация о данных](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#Краткая-информация-о-данных)  
[4. Этапы работы над проектом](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#Этапы-работы-над-проектом)  
[5. Результат](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#Результат)    
[6. Выводы](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#Выводы) 

### Описание проекта    
Угадать загаданное компьютером число за минимальное число попыток.

:arrow_up:[к оглавлению](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#оглавление)
:arrow_up::arrow_up:[к проектам](https://github.com/elvisandsivle/sf_data_science/README.md#Проекты)


### Какой кейс решаем?    
Нужно написать программу, которая угадывает число за минимальное число попыток

**Условия соревнования:**  
- Компьютер загадывает целое число от 0 до 100, и нам его нужно угадать. Под «угадать», подразумевается «написать программу, которая угадывает число».
- Алгоритм учитывает информацию о том, больше ли случайное число или меньше нужного нам.

**Метрика качества**     
Результаты оцениваются по среднему количеству попыток при 10000 повторений

**Что практикуем**     
Учимся писать хороший код на python


### Краткая информация о данных
Загадываем список чисел через функцию ``np.random.randint(1, 101, size=(10000))``
  
:arrow_up:[к оглавлению](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#оглавление)
:arrow_up::arrow_up:[к проектам](https://github.com/elvisandsivle/sf_data_science/README.md#Проекты)


### Этапы работы над проектом  
Для решения данной задачи было решено использовать метод двух указателей (low, high).

:arrow_up:[к оглавлению](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#оглавление)
:arrow_up::arrow_up:[к проектам](https://github.com/elvisandsivle/sf_data_science/README.md#Проекты)


### Результаты:  

Алгоритм "угадывает" число в среднем за 7 попыток при 10000 повторений.

:arrow_up:[к оглавлению](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#оглавление)
:arrow_up::arrow_up:[к проектам](https://github.com/elvisandsivle/sf_data_science/README.md#Проекты)


### Выводы:

Алгоритм работает в соответствии с условием. Метод с двух указателей показал себя достаточно успешно в данной задачи.

:arrow_up:[к оглавлению](https://github.com/elvisandsivle/sf_data_science/tree/main/project_0/README.md#оглавление)
:arrow_up::arrow_up:[к проектам](https://github.com/elvisandsivle/sf_data_science/README.md#Проекты)