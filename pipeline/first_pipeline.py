from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Pipeline组装") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

# 准备用于训练的数据集，分别为：id、文本、分类标签
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
# 配置一个ML管道，包含了以下几个阶段：分词、特征、逻辑回归模型
# 指定输入的是text类，输出的是words列，包含了所有分词后的结果
tokenizer = Tokenizer(inputCol="text", outputCol="words")
# 将单词转换为特征
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
# 创建逻辑回归模型
lr = LogisticRegression(maxIter=10, regParam=0.001)
# 创建Pipeline，并设置各个stage
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# 训练
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
# 准备测试数据集，这些数据仅有id、text，并没有分类标签
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# 使用之前的模型进行预测
prediction = model.transform(test)
# 输出预测结果
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row  # type: ignore
    print(
        "(%d, %s) --> prob=%s, prediction=%f" % (
            rid, text, str(prob), prediction   # type: ignore
        )
    )