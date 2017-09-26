from sklearn.metrics import roc_auc_score
from sklearn import utils 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
import operator as op 
import pandas as pd
import os
import datetime

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot


path = "./preprocessed_dataset/"
filenames = [os.path.join(path, filename) for filename in os.listdir(path)]


data = [pd.read_csv(f) for f in filenames]
whole_data = pd.concat(data,axis=0)

label = np.array(whole_data["species"])
train = np.array(whole_data.drop("species", axis=1))


# dataset suffling 
train,label = utils.shuffle(train,label, random_state=0)

seed = 7
test_size = 0.33
train_data, test_data, train_label, test_label = train_test_split(train, label, test_size=test_size, random_state=seed)


print("Training_data :",train_data.shape)
print("Training_label :",train_label.shape)

print("Test_data :",test_data.shape)
print("Test_label :",test_label.shape)


gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)

result = []
nTreeList = range(50, 500, 10)

for iTrees in nTreeList:
    depth = None
    maxFeat = 4 #조정해볼 것
    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(train_data, train_label)

    #데이터 세트에 대한 MSE 누적
    y_pred = gbrt.predict(test_data)
    result.append(mean_squared_error(test_label, y_pred))
    

#트레이닝 테스트 오차 대비  앙상블의 트리 개수 도표 그리기
plot.figure(2)
plot.plot(nTreeList, result)
plt.title('Prediction error of gradient boosting ')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
#plot.ylim([0.0, 1.1*max(mseOob)])
plot.show()



# make predictions for test data
# y_pred = gbrt.predict(test_data)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(test_label, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



# print("테스트 세트 정확도: {:.2f}".format(gbrt.score(test_data, test_label)))
# print("테스트 세트 정확도: {:.2f*100}".format(gbrt.score(test_data, test_label)))
# print("테스트 세트 정확도: %.2f %%" % .format(gbrt.score(test_data, test_label))*100)

