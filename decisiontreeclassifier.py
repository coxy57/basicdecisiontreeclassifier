from sklearn.tree import DecisionTreeClassifier
import numpy as np

class BasicDecisionTreeClassifier:
    def __init__(self):
        # studytime, sleep
        self.x = np.array([[0,1],[6,7],[1,1],[7,7]])
        self.y = np.array(['fail','pass','pass','pass']).reshape(-1,1)
        self.dtc = DecisionTreeClassifier()
        self.dtc.fit(self.x,self.y)
    def predict(self,item):
        item_prediction = self.dtc.predict([item])
        return item_prediction
    def score_item(self,x,y):
        # input values - > x
        # target values/labels for x -> y
        item_scoring = self.dtc.score(x,y)
        return item_scoring
    
b = BasicDecisionTreeClassifier()
# predict using studytime and sleep
# using 0 studytime and 1 hour sleep
# it predicts fail because of the input data/labels we gave it
print(b.predict([0,1]))
