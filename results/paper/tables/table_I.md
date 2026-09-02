TABLE I: Group representations, reference targets, and group OOD scores of the four methods compared in Experiment 1.

| 방법 | 그룹 표현 | 참조 대상 | 그룹 OOD 점수 |
|---|---|---|---|
| CNC-Local | 원본과 증강 입력의 활성 정보를 합친 질의 뉴런 군집 | 학습 클래스별 참조 뉴런 군집 | 최대 Jaccard 유사도에 음의 부호를 붙인 값 |
| GEM | 정규화하지 않은 특징의 평균 | 클래스 평균과 클래스 내 공유 공분산 | 클래스별 Mahalanobis 거리에 $-\frac{1}{2}$을 곱한 값의 log-sum-exp에 음의 부호를 붙인 값 |
| DN2 | 정규화하지 않은 특징의 평균 | 정규화하지 않은 학습 특징으로 구성한 참조 그룹 평균 | 가장 가까운 두 참조 표현까지의 제곱 Euclidean 거리 평균 |
| GMMD-Central | 정규화 특징의 평균 | 클래스별 참조 성분과 그룹 크기를 반영한 그룹 평균 분포 | 식 (4) |
