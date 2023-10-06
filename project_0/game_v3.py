import numpy as np
 
def random_predict(number:int=1) -> int:
    count = 0
    low = 1
    high = 101

    while True:
        count += 1
        predict_number = np.random.randint(low, high) # предполагаемое число
        if predict_number > number:
            high = predict_number
        elif predict_number < number:
            low = predict_number + 1
        else:
            break # конец игры, выход из цикла
    return count
 
def score_game(random_predict) -> int:
    count_ls = [] # список для сохранения количества попыток
    np.random.seed(1) # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(10000)) # загадали список чисел
 
    for number in random_array:
        count_ls.append(random_predict(number))
 
    score = int(np.mean(count_ls)) # находим среднее количество попыток
    print(f"Ваш алгоритм угадывает число в среднем за: {score} попытки")
    return(score)
    
# RUN
if __name__ == '__main__':
    score_game(random_predict)