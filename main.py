# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn import datasets
from iris_classifier import IrisClassifier 

iris = datasets.load_iris() # train data 가져오기
X, y = iris.data, iris.target # Prediction Service class를 import

clf = svm.SVC(gamma='scale') # Model Training
clf.fit(X, y)

iris_classifier_service = IrisClassifier() # Prediction Service Class에 대한 객체 생성
iris_classifier_service.pack('model', clf) 
saved_path = iris_classifier_service.save() # Prediction Service를 BENTOML_HOME에 저장한다. 

# bentoml 폴더 아래에 학습한 모델정보가 저장되고 Dockerfile environment 등이 자동으로 생성
# IrisClassifier 폴더에 모델 피크 파일 저장.
# 폴더 이름은 클래스 이름으로 생성된다.

# bentoml serve IrisClassifier:latest 를 이용해 bentoML 서버를 실행한다.
# /bentoml/logs/ prediction.log 에서 예측 로그 확인 가능.