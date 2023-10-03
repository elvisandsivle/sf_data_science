import numpy as np

def random_predict(number: int = 1):
    count = 0
    min = 1
    max = 100
    
    while True:
        predict = round((min+max)/2)
        count += 1
        if number == predict:
            break
        elif number > predict:
            min = predict
        elif number < predict:
            max = predict

    return count

def score_game(random_predict) -> int:
    count_ls = [] # список для сохранения количества попыток
    np.random.seed(1) # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(10000)) # загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls)) # находим среднее количество попыток
    print(f"Ваш алгоритм угадывает число в среднем за: {score} попытки")
    
# RUN
if __name__ == '__main__':
    score_game(random_predict)