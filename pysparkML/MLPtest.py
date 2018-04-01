# coding=UTF-8
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType


if __name__ == "__main__":

    # sc = SparkContext('local', 'mlp')
    # sqlContext = SQLContext(sc)

    spark = SparkSession\
        .builder\
        .appName("MLPClassifier")\
        .getOrCreate()

    df = spark.read.csv('./data/titanic/test.csv', header=True)
    # 丢弃不用的数据
    df = df.drop('PassengerId', 'Name', 'Ticket', 'Cabin',)
    print(df.first())
    df = df.withColumn("Survived", df["Survived"].cast(IntegerType()))
    df = df.withColumn("Pclass", df["Pclass"].cast(IntegerType()))
    df = df.withColumn("Age", df["Age"].cast(DoubleType()))
    df = df.withColumn("SibSp", df["SibSp"].cast(IntegerType()))
    df = df.withColumn("Parch", df["Parch"].cast(IntegerType()))
    df = df.withColumn("Fare", df["Fare"].cast(DoubleType()))

    print(df.dtypes)

    # fill NaN
    ave_age = round(df.groupBy().avg("Age").collect()[0][0], 2)
    df = df.na.fill({'Age': ave_age})
    df = df.na.drop()

    # map categorical data
    indexer = StringIndexer(inputCol="Sex", outputCol="SexInd")
    df = indexer.fit(df).transform(df)

    indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedInd")
    df = indexer.fit(df).transform(df)

    # assemble features
    assembler = VectorAssembler(
        inputCols=["Age", "Pclass", "SexInd", "SibSp", "Parch", "Fare", "EmbarkedInd"],
        outputCol="features")

    df = assembler.transform(df)

    (trainingData, testData) = df.randomSplit([0.8, 0.2])
    print(trainingData.first())
    print(trainingData.count())
    print(testData.first())
    print(testData.count())

    layers = [7, 8, 4, 2]  # input: 7 features; output: 2 classes
    mlp = MultilayerPerceptronClassifier(maxIter=1000, layers=layers,
                                         labelCol="Survived", featuresCol="features",
                                         blockSize=128, seed=0)

    model = mlp.fit(trainingData)
    result = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction",
                                                  metricName="accuracy")
    prediction_label = result.select("prediction", "Survived")
    print("MLP test accuracy: " + str(evaluator.evaluate(prediction_label)))
