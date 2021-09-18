from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Estimator、Transformer、Param测试") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

# 使用Python的列表和元组构建训练集
# 标签、特征
training = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

# 创建逻辑回归实例，这其实就是一个Estimator
lr = LogisticRegression(maxIter=10, regParam=0.01)

# 打印逻辑回归参数、文档、默认值
print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# 训练逻辑回归模型，使用存储在lr实例中的参数，这其实就是一个Transformer
model1 = lr.fit(training)

# 查看模型1的参数，这些参数属于一个拥有唯一ID的逻辑回归实例
print("Model 1 was fit using parameters: ")
print(model1.extractParamMap())

# 直接使用Python的字段来创建ParamMap
paramMap = {lr.maxIter: 20}
paramMap[lr.maxIter] = 30  # Specify 1 Param, overwriting the original maxIter.
# 更新参数Map（这都是Python Dict的参数）
paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # type: ignore

# 合并参数
paramMap2 = {lr.probabilityCol: "myProbability"}  # type: ignore
paramMapCombined = paramMap.copy()
paramMapCombined.update(paramMap2)  # type: ignore

# 使用新的参数训练一个新的模型(model2)
model2 = lr.fit(training, paramMapCombined)
# 打印模型2吃的参数
print("Model 2 was fit using parameters: ")
print(model2.extractParamMap())

# 准备测试集
test = spark.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

# 使用model2进行与预测，得到一个新的DataFrame。虽然，test数据集有label，但模型会基于特征进行预测
prediction = model2.transform(test)
result = prediction.select("features", "label", "myProbability", "prediction") \
    .collect()

for row in result:
    print("features=%s, label=%s -> prob=%s, prediction=%s"
          % (row.features, row.label, row.myProbability, row.prediction))