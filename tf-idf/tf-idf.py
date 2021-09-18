from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TF-IDE") \
    .master("local[1]") \
    .getOrCreate()  # type: SparkSession

# 创建一个数据源
sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])

# 创建一个分词器
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
# 执行分词转换
wordsData = tokenizer.transform(sentenceData)
print("-" * 100)
print("分词后的数据:")
wordsData.show(truncate=False)

# 使用CountVectorizer创建文档词条计数特征
print("-" * 100)
print("使用CountVectorizer处理特征：")
countVector = CountVectorizer(inputCol="words", outputCol="countFeatures")
countModel = countVector.fit(wordsData)
countModel.transform(wordsData).show(truncate=False)

# 创建特征向量
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
# 特征转换
featurizedData = hashingTF.transform(wordsData)

print("-" * 100)
print("TF处理后的数据（计算词频）:")
featurizedData.select("words", "rawFeatures").show(truncate=False)

# 创建IDF进行特征转换
idf = IDF(inputCol="rawFeatures", outputCol="features")
# fit模型
idfModel = idf.fit(featurizedData)
# 使用IDF模型转换之前的特征数据
rescaledData = idfModel.transform(featurizedData)
# 再次查看标签和特征数据
rescaledData.select("label", "features").show(truncate=False)