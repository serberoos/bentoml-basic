# -*- coding: utf-8 -*-
import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

# export BENTOML_HOME = '/root/jaegeun_project/bentoml-basic/bentoml' 로 BENTOML_HOME 변경 가능

@env(infer_pip_packages=True)
# @env 데코레이터를 사용해 pip 패키지를 추론해서 requirement.txt를 생성한다 또한, 직접 버전을 명시할 수 있고 이를 이용해 docker를 활용할 수 있다.
@artifacts([SklearnModelArtifact('model')])
# bentoML에서 이미 만든 Artifact를 사용하며 @artifacts를 사용한다.
# 여기서 model은 Prediction Service Class에서 부를 이름이고 predict 함수에서 self.artifacts.model.predict 를 의미한다.



# 사이킷 런 모델 이용한 prediction
# BentoService를 상속해 Prediction Service Class를 생성
class IrisClassifier(BentoService):

    @api(input=DataframeInput(), batch=True) # 데코레이터 @api를 설정한다.
    # API input, output, batch 유무를 인자로 받을 수 있다.
    def predict(self, df:pd.DataFrame):
    
        # 데이터프레임 입력 어댑터를 사용하여 '예측'이라는 이름의 추론 API
        # HTTP 요청 또는 CSV 파일이 Panda Dataframe 개체로 변환된다.
    
        return self.artifacts.model.predict(df)