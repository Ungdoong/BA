# 191213(Fri)_ML\_Modeling\_Algorithm

## 지도 학습

___________________

- k-최근접 이웃 ( k-Nearest Neighbors )
- 선형 회귀 ( Linear Regression )
- 로지스틱 회귀 ( Logistic Regression )
- 서포트 벡터 머신 ( SVM : Support Vector Machines )
- 결정 트리 ( Decision Tree ) & 랜덤 포레스트 ( Random Forests )
- 신경망 ( Neural Networks 6 )



## 비지도 학습

________________

### 군집 ( Clustering )

- k-평균 ( k-means )
- 계층 군집 분석 ( HCA : Hierarchical Cluster Analysis )
- 기댓값 최적화 ( Expectation Maximization )

### 시각화 ( Visualization ) & 차원 축소 ( Dimensionality Reduction )

- 주성분 분석 ( PCA : Principal Component Analysis )
- 커널 ( kernel )
- 지역적 선형 인베딩 ( LLE : Locally-Linear Embedding )
- t-SNE ( t-distributed Stochastic Neighbor Embedding )

### 연관 규칙 학습 ( Association Rule Learning )

- 어프라이어리 ( Apriori )
- 이클렛 ( Eclat )



## 준지도 학습

_____________________

지도 학습 + 비지도 학습

- 심층 신뢰 신경망 ( DBN : Deep Belief Network )

  비지도 학습 방식으로 순차적으로 훈련된 다음 ㅈ던체 시스템이 지도 학습 방식으로 세밀하게 조정

  - 제한된 볼트만 머신 ( RBM : Restricted Boltzmann Machine )이라는 비지도 학습에 기초



## 강화 학습

______________

학습하는 시스템을 에이전트라고 부르며 환경을 관찰해서 행동을 실행하고 그 결과로 보상을 획득

시간이 지나면서 가장 큰 보상을 얻기 위해 정책이라고 부르는 최상의 전략으로 스스로 학습

정책은 주어진 상황에서 에이전트가 어떤 행동을 선택해야 할지 결정



## 출처

_______________

[https://tensorflow.blog/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-1%EC%9E%A5-2%EC%9E%A5/1-3-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%8B%9C%EC%8A%A4%ED%85%9C%EC%9D%98-%EC%A2%85%EB%A5%98/](https://tensorflow.blog/핸즈온-머신러닝-1장-2장/1-3-머신러닝-시스템의-종류/)