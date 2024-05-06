import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    prop = np.sum(y, axis=0)/y.shape[0]

    entropy = -np.sum(prop*np.log(prop + EPS))
    
    return entropy
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    prop = np.sum(y, axis=0)/y.shape[0]

    gini = 1-np.sum(prop ** 2)
    
    return gini
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    mean = (np.sum(y, axis=0)/y.shape[0])[0]
    
    return np.sum((y - mean) ** 2) / y.shape[0]

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    return np.sum(y - np.median(y)) / y.shape[0]


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        left_ind = []
        right_ind = []
        for i in range(len(X_subset)):
            if X_subset[i][feature_index] < threshold:
                left_ind.append(i)
            else:
                right_ind.append(i)
                
        X_left = X_subset[left_ind, :]
        y_left = y_subset[left_ind, :]
        X_right = X_subset[right_ind, :]
        y_right = y_subset[right_ind, :]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_ind = []
        right_ind = []
        for i in range(len(X_subset)):
            if X_subset[i][feature_index] < threshold:
                left_ind.append(i)
            else:
                right_ind.append(i)
        
        y_left = y_subset[left_ind, :]
        y_right = y_subset[right_ind, :]
        
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        crit_func, _ = self.__class__.all_criterions[self.criterion_name]
        
        n_objects, n_features = X_subset.shape
        best_criterion_value = -np.inf
        best_feature_index = None
        best_threshold = None
        
        for feature_index in range(n_features):
            # Сортировка значений по текущему признаку
            sorted_indices = np.argsort(X_subset[:, feature_index])
            sorted_X, sorted_y = X_subset[sorted_indices], y_subset[sorted_indices]
        
            # Рассмотрение всех возможных порогов разбиения для текущего признака
            for i in range(1, n_objects):
                if sorted_X[i-1, feature_index] == sorted_X[i, feature_index]:
                    continue  # Пропуск одинаковых значений
                   
                y_left = sorted_y[:i]
                y_right = sorted_y[i:]
        
                # Вычисление критерия разбиения
                h_yl = crit_func(y_left)
                h_yr = crit_func(y_right)
                h_y = crit_func(sorted_y)
                G = h_y - (len(y_right)/n_objects) * h_yr - (len(y_left)/n_objects) * h_yl
        
                # Обновление наилучших параметров разбиения
                if G > best_criterion_value:
                    best_criterion_value = G
                    best_feature_index = feature_index
                    best_threshold = (sorted_X[i-1, feature_index] + sorted_X[i, feature_index]) / 2
        
        return best_feature_index, best_threshold
    
    """
    class Node:
  
    This class is provided "as is" and it is not mandatory to it use in your code.
    
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
    """
    
    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        _, is_classification = self.__class__.all_criterions[self.criterion_name]
        
        if depth >= self.max_depth or len(X_subset) < self.min_samples_split:
            if is_classification:
                proba = np.mean(y_subset, axis=0)
            else:
                proba = np.mean(y_subset)
            return Node(feature_index=None, threshold=None, proba=proba)
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        
        if feature_index is None:
            
            if self.__class__.all_criterions[self.criterion_name][1]:  # Проверяем, является ли задача классификацией
               # Для классификации: вычисляем вероятности классов как долю каждого класса в подмножестве
               class_counts = np.sum(y_subset, axis=0)
               proba = class_counts / np.sum(class_counts)
            else:
               # Для регрессии: вычисляем среднее значение целевой переменной
               proba = np.mean(y_subset, axis=0)
            return Node(feature_index=None, threshold=None, proba=proba)

        
         # Индексы для левого и правого поддерева
        left_indices = [i for i, x in enumerate(X_subset[:, feature_index]) if x < threshold]
        right_indices = [i for i, x in enumerate(X_subset[:, feature_index]) if x >= threshold]

        # Рекурсивное создание дочерних узлов
        left_child = self.make_tree(X_subset[left_indices], y_subset[left_indices], depth + 1)
        right_child = self.make_tree(X_subset[right_indices], y_subset[right_indices], depth + 1)

        # Создание нового узла и назначение ему дочерних узлов
        new_node = Node(feature_index=feature_index, threshold=threshold)
        new_node.left_child = left_child
        new_node.right_child = right_child

        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    
    
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        predictions = []  # Инициализация списка для накопления предсказаний

        for obj in X:
            node = self.root
            while node.left_child or node.right_child:  # Пока не дойдем до листа
                if obj[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
            if self.__class__.all_criterions[self.criterion_name][1]:  # Если задача классификации
                predictions.append(np.argmax(node.proba))  # Для классификации добавляем класс с максимальной вероятностью
            else:
                predictions.append(node.proba)  # Для регрессии добавляем предсказанное значение

        y_predicted = np.array(predictions).reshape(-1, 1)  # Преобразование списка в массив с нужной формой
        return y_predicted
        
        
        
        
    def predict_proba(self, X):
        assert hasattr(self, 'root'), 'Model is not trained yet'
        crit_func, is_classification = self.__class__.all_criterions[self.criterion_name]
        assert is_classification, 'Available only for classification problem'

        # Инициализация списка для хранения предсказанных вероятностей
        predicted_probs = []

        # Проход по всем объектам в X
        for obj in X:
            # Начало от корня дерева
            node = self.root

            # Цикл до тех пор, пока не будет достигнут листовой узел
            while node.left_child or node.right_child:
                # Выбор направления движения в дереве на основе значения признака
                if obj[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child

            # Добавление вектора вероятностей из листового узла в список предсказаний
            predicted_probs.append(node.proba)  # предполагается, что node.proba - это уже вектор вероятностей

        # Преобразование списка предсказанных вероятностей в np.array
        y_predicted_probs = np.array(predicted_probs)
        return y_predicted_probs
