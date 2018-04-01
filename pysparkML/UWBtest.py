# coding=UTF-8
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType


if __name__ == "__main__":

    # sc = SparkContext('local', 'mlp')
    # sqlContext = SQLContext(sc)

    spark = SparkSession\
        .builder\
        .appName("UWBClassifier")\
        .getOrCreate()

    df = spark.read.csv('./data/uwb/Train_Test_Data_8000dB.csv', header=True)
    # 丢弃不用的数据
    df = df.drop('f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11')
    df = df.drop('rssiE', 'rssiH', 'h1', 'h2', 'eps1', 'eps2')

    df = df.withColumn("theta1", df["theta1"].cast(DoubleType()))
    df = df.withColumn("theta2", df["theta2"].cast(DoubleType()))
    df = df.withColumn("theta3", df["theta3"].cast(DoubleType()))
    df = df.withColumn("theta4", df["theta4"].cast(DoubleType()))
    df = df.withColumn("theta5", df["theta5"].cast(DoubleType()))
    df = df.withColumn("theta6", df["theta6"].cast(DoubleType()))
    df = df.withColumn("theta7", df["theta7"].cast(DoubleType()))
    df = df.withColumn("theta8", df["theta8"].cast(DoubleType()))
    df = df.withColumn("theta9", df["theta9"].cast(DoubleType()))
    df = df.withColumn("theta10", df["theta10"].cast(DoubleType()))
    df = df.withColumn("theta11", df["theta11"].cast(DoubleType()))
    df = df.withColumn("deletaPhi1", df["deletaPhi1"].cast(DoubleType()))
    df = df.withColumn("deletaPhi2", df["deletaPhi2"].cast(DoubleType()))
    df = df.withColumn("deletaPhi3", df["deletaPhi3"].cast(DoubleType()))
    df = df.withColumn("deletaPhi4", df["deletaPhi4"].cast(DoubleType()))
    df = df.withColumn("deletaPhi5", df["deletaPhi5"].cast(DoubleType()))
    df = df.withColumn("deletaPhi6", df["deletaPhi6"].cast(DoubleType()))
    df = df.withColumn("deletaPhi7", df["deletaPhi7"].cast(DoubleType()))
    df = df.withColumn("deletaPhi8", df["deletaPhi8"].cast(DoubleType()))
    df = df.withColumn("deletaPhi9", df["deletaPhi9"].cast(DoubleType()))
    df = df.withColumn("deletaPhi10", df["deletaPhi10"].cast(DoubleType()))
    df = df.withColumn("deletaPhi11", df["deletaPhi11"].cast(DoubleType()))
    df = df.withColumn("distance", df["distance"].cast(IntegerType()))
    df = df.withColumn("dis", df["dis"].cast(IntegerType()))

    print(df.show(2))
    print(df.count())
    # print(df.dtypes)

    df.createOrReplaceTempView("tempView1")
    df2 = spark.sql("SELECT distance FROM tempView1")
    outNums = df2.distinct().count()
    print(outNums)

    # assemble features
    assembler = VectorAssembler(
        inputCols=["theta1", "theta2", "theta3", "theta4", "theta5", "theta6",
                   "theta7", "theta8", "theta9", "theta10", "theta11",
                   "deletaPhi1", "deletaPhi2", "deletaPhi3", "deletaPhi4", "deletaPhi5", "deletaPhi6",
                   "deletaPhi7", "deletaPhi8", "deletaPhi9", "deletaPhi10", "deletaPhi11"],
        outputCol="features")
    df = assembler.transform(df)

    (trainingData, testData) = df.randomSplit([0.4, 0.1])
    # print(trainingData.first())
    print(trainingData.count())
    # print(testData.first())
    print(testData.count())

    # 就特么能二分类装毛儿还让我输出分类数啊！！！！
    layers = [22, 5, outNums]  # input: 22 features; output: 2 classes
    mlp = MultilayerPerceptronClassifier(maxIter=10, layers=layers,
                                         labelCol="distance", featuresCol="features",
                                         blockSize=128, seed=10)

    model = mlp.fit(trainingData)
    result = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(labelCol="distance", predictionCol="prediction",
                                                  metricName="accuracy")
    prediction_label = result.select("prediction", "distance")
    print("MLP test accuracy: " + str(evaluator.evaluate(prediction_label)))

