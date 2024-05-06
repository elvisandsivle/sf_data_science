import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

class SimplifiedBoostingRegressor:
    def __init__(self):
        self.models_list = []
        self.lr = 0.1
        self.loss_log = []
        
    @staticmethod
    def loss(targets, predictions):
        loss = np.mean((targets - predictions)**2)
        return loss
    
    @staticmethod
    def loss_gradients(targets, predictions):
        # Градиенты функции потерь (для MSE: 2 * среднее(предсказание - цель))
        gradients = 2 * (predictions - targets) / len(targets)  
        assert gradients.shape == targets.shape
        return gradients
        
        
    def fit(self, model_constructor, data, targets, num_steps=10, lr=0.1, max_depth=5, verbose=False):
        '''
        Fit sequence of models on the provided data.
        Model constructor with no parameters (and with no ()) is passed to this function.
        If 
        
        example:
        
        boosting_regressor = SimplifiedBoostingRegressor()    
        boosting_regressor.fit(DecisionTreeRegressor, X, y, 100, 0.5, 10)
        '''
        self.lr = lr
        self.lr = lr
        new_targets = targets.copy()
        for step in tqdm(range(num_steps)):
            model = model_constructor(max_depth=max_depth)
            model.fit(data, new_targets)
            self.models_list.append(model)

            predictions = self.predict(data)
            self.loss_log.append(self.loss(targets, predictions))

            gradients = self.loss_gradients(targets, predictions)
            new_targets = targets - predictions  # Обновляем цели на основе текущих предсказаний

        if verbose:
            print('Finished! Loss=', self.loss_log[-1])
            
    def predict(self, data):
        if not self.models_list:
            return np.zeros(len(data))
        
        predictions = np.zeros(len(data))
        for model in self.models_list:
            predictions += self.lr * model.predict(data)
        return predictions
    
    