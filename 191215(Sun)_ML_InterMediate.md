# 191215(Sun)\_ML_Intermediate

## Missing Values

____________________

### Three Approaches

##### 1. A Simple Option: Drop Columns with Missing Values

- 값이 존재하지 않을 경우 해당 column을 제외

![image-20191215184050180](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191215184050180.png)

##### 2. A Better Option: Imputation

- 값이 존재하지 않을 경우 해당 column의 평균으로 채움
- column을 제외시키는 것보다 더 정확한 모델

![image-20191215184148548](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191215184148548.png)

##### 3. An Extension To Imputation

- Imputation 여부를 표시

![image-20191215184525418](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191215184525418.png)



## Categorical Variables

_________________

### Three Approaches

##### 1. Drop Categorical Variables

- Dataset에서 Categorical Variables를 삭제시킴
- 유용한 정보가 포함되지 않았을 경우 사용

##### 2. Label Encoding

- 서로 다른 Integer로 각 variable들을 표현
- value간의 명확한 정렬기준이 있을 때 사용(ordinal variables)

![image-20191215210042745](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191215210042745.png)

##### 3. One-Hot Encoding

- 가능한 값을 각각 column으로 만들어 상태체크
- value 간의 명확한 정렬기준이 없을 때 사용(nominal variables)
- categorical variables가 매우 다양한 value 값들을 가질 경우 효과적이지 못함(15가지 이하일 경우만 사용)

![image-20191215210408823](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191215210408823.png)



## Pipelines

________________

A pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

### Benefit

1. Cleaner Code

   With a pipeline, you won't need to manually keep track of your training and validation data at each step.

   매 단계마다 training과 validation을 수행할 필요가 사라짐

2. Fewer Bugs

   미적용 또는 전처리 과정을 생략할 가능성이 적어짐

3. Easier to Productionize

   프로토타입 모델을 다른 효율적인 모델로 전환하는 것이 쉬워짐

4. More Options for Model Validation

   cross-validation etc...



## Cross-Validation

__________________

Run modeling process on different subsets of the data to get multiple measures of model qualify

![image-20191216004359535](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191216004359535.png)

### When use?

- 추가 연산의 부담이 적은 작은 데이터셋
- 대용량의 데이터셋에서는 single validation set으로 충분



## XGBoost

____________________

### Gradient Boosting

A method that goes through cycles to iteratively add models into an ensemble.

![image-20191216012341810](C:\Users\Tacjin\AppData\Roaming\Typora\typora-user-images\image-20191216012341810.png)

- 자세한 설명

   https://brunch.co.kr/@snobberys/137 