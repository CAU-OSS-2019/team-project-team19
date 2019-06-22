# team-project-team19
team-project-team19 created by GitHub Classroom

 
o 참고
 홈페이지는 저희 팀 사정상 제외하게 되었으니 이미지 인식/분류 파이썬 코드만 보시면 됩니다.
 
o 구현물에 대한 간단한 설명

 우리 팀은 신경 네트워크. 구체적으로, 우리는 미리 훈련된 과일 이미지를 통해 '이미지 네트워크'에서 'tenserflow-keras'를 통해 이미지 분류를 했다. 




                                                       
           
 먼저 keras를 통해 5000개의 과일 이미지 데이터 훈련을 한다.시간이 많이 걸리지만 정확도를 97%까지 높이는 작업이다. 그 작업은 1시간 넘게 걸리며 데이터 훈련이 끝난 후 사용자한테 과일 데이터를 입력받는다. 그러면 인공지능을 통해 이미지 분류를 하고 가장 정확도가 높게 나온 결과 데이터를 출력한다.




o 핵심 기술에 대한 자세한 설명(조명,각도)

 데이터 집합은 깃헙에 있는 오픈소스 데이터를 통해 얻었다.
https://github.com/Horea94/Fruit-Images-Dataset/tree/master/test-multiple_fruits
 먼저 꼼꼼히 선별하여 여러 각도 조건하의 10가지 흔한 과일 그림을 골라내었다. 그걸로 각도 문제를 해결하였다. 과일 이미지를 분류하고 나서 convolutional neural network를 사용하여 과일 이미지를 훈련한다. 훈련은 그라디언트 하강의 방법을 사용했다. 먼저 목표 함수를 만든다. 다음 예측값과 실제값 사이의 오차 함수를 최적화함으로써 모델 파라메타를 훈련시키다. 마지막으로 파라미터를 모델에 넣어 예측함으로써 훈련 성과를 이루게 했다. 그런 다음 테스트 데이터 집합에서 검증하여 데이터 집합을 훈련 집합(train) 과 테스트 집합(test)으로 나눴다. 사진 비율은 train: test= 7:3 이다. 이미지 인식률을 높이기 위해 train은 test보다 조금 더 잘 표현된 사진을 사용했다.  overfitting현상은 나타나지 않았다. 동시에 이미지 데이터를 먼저 균등분포를 조작함으로써, 조명이 미치는 영향을 크게 줄일 수 있다. 마지막으로 전체적으로 과일 이미지 식별 정확도가 90% 이상이다.

