from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

if __name__ == '__main__':
    data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
            (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
            (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
            (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),),
            (Vectors.sparse(4, [(1, 9.0), (2, 3.0)]),)
    ]

    spark = SparkSession.builder \
        .appName("机器学习数值相关性计算") \
        .master("local[1]") \
        .getOrCreate()  # type: SparkSession

    df = spark.createDataFrame(data, ["features"])
    df.show(truncate=False)

    r1 = Correlation.corr(df, "features").head()
    print("皮尔逊相关性矩阵为：\n" + str(r1[0]))

    r2 = Correlation.corr(df, "features", "spearman").head()
    print("斯皮尔曼相关性矩阵为：\n" + str(r2[0]))

