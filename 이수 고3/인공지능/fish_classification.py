import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# data load
data = pd.read_csv('./Fish.csv')
print(data)
print('=============================================')

# data 에 결측값을 확인한다.
print(data.isnull().sum())
print('=============================================')

# target 데이터 분리
# 특정 column 값만 추출, 추출된 column 을 삭제
y = data['Species']
X = data.drop('Species', axis=1)
print(X)
print('=============================================')

# Scaler 를 적용
"""
StandardScaler 
평균이 0이고 분산이 1인 정규 분포로 만들어 준다.
수식: (Xi - (X의 평균)) / (X의 표준편차)

MinMaxScaler 
모든 값을 0~1 사시의 값으로 바꾸어주는 것. 이때 음수도 예외 없이 다 바꾼다.
수식: ( X- (X의 최솟값) ) / ( X의 최댓값 - X의 최솟값 )
"""
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_scale = scaler.fit_transform(X)
X = pd.DataFrame(x_scale, columns=X.columns)
print(X)
print('=============================================')

# 카테고리형 데이터를 수치형으로 변환
print(f'y original: {y}')
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(f'y after encoding: {y}')
y_mappings = {index: label for index, label in enumerate(encoder.classes_)}
print(y_mappings)

# train test data 로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
print(f'X_train: {X_train}')
print(f'X_test: {X_test}')
print(f'y_train: {y_train}')
print(f'y_test: {y_test}')
print('=============================================')

# 학습 and visualize
tr_scores = []
te_scores = []
for i in range(1, 16):
    # model 선언
    knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=i)
    # 학습
    knn.fit(X_train, y_train)

    # predict y value
    pred_train = knn.predict(X_train)
    pred_test = knn.predict(X_test)

    # accuracy 구하기
    tr_score = accuracy_score(pred_train, y_train)
    tr_scores.append(tr_score)

    te_score = accuracy_score(pred_test, y_test)
    te_scores.append(te_score)
    print(f'[{i} neighbors accuracy]: {te_score}')

plt.plot(range(1, 16), tr_scores, label='TRAIN set')
plt.plot(range(1, 16), te_scores, label='TEST set')
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()
