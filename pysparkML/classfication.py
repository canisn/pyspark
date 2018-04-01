#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import \
    LogisticRegression,\
    DecisionTreeClassifier,\
    DecisionTreeClassificationModel,\
    RandomForestClassifier,\
    RandomForestClassificationModel,\
    NaiveBayes,\
    NaiveBayesModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer


def data_explore():
    conf = SparkConf().setAppName('data_explore').setMaster('local')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile('./data/ml-100k/u.user')
    user_fiels = user_data.map(lambda line: line.split('|'))
    movie_data = sc.textFile('./data/ml-100k/u.item')
    movie_fields = movie_data.map(lambda line: line.split('|'))
    rating_data = sc.textFile('./data/ml-100k/u.data')
    rating_fields = rating_data.map(lambda line: line.split('\t'))
    num_movies = movie_fields.map(lambda field: field[0]).distinct().count()
    print('first line of user data:' + str(user_fiels.first()))
    print('num of movies is:' + str(num_movies))
    print('first line of movie data:' + str(movie_fields.first()))
    print('first line of rating data:' + str(rating_fields.first()))
    sc.stop()


def estimator_transformer():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    # Prepare training data from a list of (label, features) tuples.
    training = spark.createDataFrame([
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

    # Create a LogisticRegression instance. This instance is an Estimator.
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    # Print out the parameters, documentation, and any default values.
    print("\nLogisticRegression parameters:\n" + lr.explainParams() + "\n")
    lr.setMaxIter(10).setRegParam(0.01).setAggregationDepth(5)
    # Learn a LogisticRegression model. This uses the parameters stored in lr.
    model1 = lr.fit(training)

    # Since model1 is a Model (i.e., a transformer produced by an Estimator),
    # we can view the parameters it used during fit().
    # This prints the parameter (name: value) pairs, where names are unique IDs for this
    # LogisticRegression instance.
    print("\nModel 1 was fit using parameters: ")
    print(model1.extractParamMap())

    # We may alternatively specify parameters using a Python dictionary as a paramMap
    paramMap = {lr.maxIter: 20}
    paramMap[lr.maxIter] = 30  # Specify 1 Param, overwriting the original maxIter.
    paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # Specify multiple Params.

    # You can combine paramMaps, which are python dictionaries.
    paramMap2 = {lr.probabilityCol: "myProbability"}  # Change output column name
    paramMapCombined = paramMap.copy()
    paramMapCombined.update(paramMap2)

    # Now learn a new model using the paramMapCombined parameters.
    # paramMapCombined overrides all parameters set earlier via lr.set* methods.
    model2 = lr.fit(training, paramMapCombined)
    print("\nModel 2 was fit using parameters: ")
    print(model2.extractParamMap())

    # Prepare test data
    test = spark.createDataFrame([
        (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
        (0.0, Vectors.dense([3.0, 2.0, -0.1])),
        (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

    # Make predictions on test data using the Transformer.transform() method.
    # LogisticRegression.transform will only use the 'features' column.
    # Note that model2.transform() outputs a "myProbability" column instead of the usual
    # 'probability' column since we renamed the lr.probabilityCol parameter previously.
    prediction = model2.transform(test)
    result = prediction.select("features", "label", "myProbability", "prediction") \
        .collect()

    for row in result:
        print("features=%s, label=%s -> prob=%s, prediction=%s"
              % (row.features, row.label, row.myProbability, row.prediction))
    spark.stop()


def pipe_line():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0)
    ], ["id", "text", "label"])

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # Fit the pipeline to training documents.
    model = pipeline.fit(training)

    # Prepare test documents, which are unlabeled (id, text) tuples.
    test = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop")
    ], ["id", "text"])

    # Make predictions on test documents and print columns of interest.
    prediction = model.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        rid, text, prob, prediction = row
        print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
    spark.stop()


def logistic_regression():
    conf = SparkConf().setAppName('RF')
    sc = SparkContext(conf=conf)
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    # 双变量Logistic回归
    brdd = sc.parallelize([Row(label=1.0, weight=2.0, features=Vectors.dense(1.0)),
                          Row(label=0.0, weight=2.0, features=Vectors.sparse(1, [], []))])
    bdf = spark.createDataFrame(brdd)
    bdf.show()
    blor = LogisticRegression(maxIter=5, regParam=0.01, weightCol='weight')
    blorModel = blor.fit(bdf)
    # 双变量logistic回归的模型系数，如果是多元Logistic回归的话会有异常
    print("blorModel.coefficients:%s" % blorModel.coefficients)
    # 二变量logistic模型的截距
    print("blorModel.intercept:%s" % blorModel.intercept)

    # 多元Logistic回归
    mrdd = sc.parallelize([Row(label=1.0, weight=2.0, features=Vectors.dense(1.0)),
                          Row(label=0.0, weight=2.0, features=Vectors.sparse(1, [], [])),
                          Row(label=2., weight=2.0, features=Vectors.dense(3.0))])
    mdf = spark.createDataFrame(mrdd)
    mdf.show()
    mlor = LogisticRegression(maxIter=5, regParam=0.01, weightCol='weight', family='multinomial')
    mlorModel = mlor.fit(mdf)
    print("mlorModel.coefficientMatrix:%s" % mlorModel.coefficientMatrix)
    print("mlorModel.interceptVector:%s" % mlorModel.interceptVector)

    # 模型预测
    test0 = sc.parallelize([Row(features=Vectors.dense(-1.0))])
    test0df = spark.createDataFrame(test0)
    result = blorModel.transform(test0df).head()
    print("blrModel result")
    print("result.prediction:%s" % result.prediction)
    print("result.probability:%s" % result.probability)
    print("result.rawPrediction:%s" % result.rawPrediction)

    test1 = sc.parallelize([Row(features=Vectors.sparse(1, [0], [1.0]))])
    test1df = spark.createDataFrame(test1)
    blorModel.transform(test1df).head().prediction
    blorModel.transform(test1df).show()
    # 模型评估
    print("模型评估：")
    blorModel.summary.roc.show()
    blorModel.summary.pr.show()


def decision_tree_classifier():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    dt = DecisionTreeClassifier(maxDepth=2, labelCol="indexed")
    model = dt.fit(td)
    # model.numNodes
    # # 3
    # model.depth
    # # 1
    # model.featureImportances
    # # SparseVector(1, {0: 1.0})
    # model.numFeatures
    # # 1
    # model.numClasses
    # # 2
    print(model.toDebugString)
    # DecisionTreeClassificationModel (uid=...) of depth 1 with 3 nodes...
    test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    result = model.transform(test0).head()
    # result.prediction
    # # 0.0
    # result.probability
    # # DenseVector([1.0, 0.0])
    # result.rawPrediction
    # # DenseVector([1.0, 0.0])
    test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    # model.transform(test1).head().prediction
    # # 1.0
    temp_path = "."
    dtc_path = temp_path + "/dtc"
    dt.save(dtc_path)
    dt2 = DecisionTreeClassifier.load(dtc_path)
    # dt2.getMaxDepth()
    # # 2
    model_path = temp_path + "/dtc_model"
    model.save(model_path)
    model2 = DecisionTreeClassificationModel.load(model_path)
    # model.featureImportances == model2.featureImportances
    # # True


def random_forest_classifier():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([
        (1.0, Vectors.dense(1.0)),
        (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="indexed", seed=42)
    model = rf.fit(td)
    # model.featureImportances
    # # SparseVector(1, {0: 1.0})
    # allclose(model.treeWeights, [1.0, 1.0, 1.0])
    # # True
    test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    result = model.transform(test0).head()
    # result.prediction
    # # 0.0
    # numpy.argmax(result.probability)
    # # 0
    # numpy.argmax(result.rawPrediction)
    # # 0
    # test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    # model.transform(test1).head().prediction
    # # 1.0
    # model.trees
    # # [DecisionTreeClassificationModel (uid=...) of depth..., DecisionTreeClassificationModel...]
    temp_path = "."
    rfc_path = temp_path + "/rfc"
    rf.write().overwrite().save(rfc_path)
    rf2 = RandomForestClassifier.load(rfc_path)
    # rf2.getNumTrees()
    # # 3
    model_path = temp_path + "/rfc_model"
    model.write().overwrite().save(model_path)
    model2 = RandomForestClassificationModel.load(model_path)
    # model.featureImportances == model2.featureImportances
    # # True


def naive_bayes():
    conf = SparkConf().setAppName('RF')
    sc = SparkContext(conf=conf)
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([Row(label=0.0, weight=0.1, features=Vectors.dense([0.0, 0.0])),
                                Row(label=0.0, weight=0.5, features=Vectors.dense([0.0, 1.0])),
                                Row(label=1.0, weight=1.0, features=Vectors.dense([1.0, 0.0]))])

    nb = NaiveBayes(smoothing=1.0, modelType="multinomial", weightCol="weight")
    model = nb.fit(df)
    # model.pi
    # # DenseVector([-0.81..., -0.58...])
    # model.theta
    # # DenseMatrix(2, 2, [-0.91..., -0.51..., -0.40..., -1.09...], 1)
    test0 = sc.parallelize([Row(features=Vectors.dense([1.0, 0.0]))]).toDF()
    result = model.transform(test0).head()
    # result.prediction
    # # 1.0
    # result.probability
    # # DenseVector([0.32..., 0.67...])
    # result.rawPrediction
    # # DenseVector([-1.72..., -0.99...])
    test1 = sc.parallelize([Row(features=Vectors.sparse(2, [0], [1.0]))]).toDF()
    # model.transform(test1).head().prediction
    # # 1.0
    temp_path = "."
    nb_path = temp_path + "/nb"
    nb.save(nb_path)
    nb2 = NaiveBayes.load(nb_path)
    # nb2.getSmoothing()
    # # 1.0
    model_path = temp_path + "/nb_model"
    model.save(model_path)
    model2 = NaiveBayesModel.load(model_path)
    # model.pi == model2.pi
    # # True
    # model.theta == model2.theta
    # # True
    nb = nb.setThresholds([0.01, 10.00])
    model3 = nb.fit(df)
    result = model3.transform(test0).head()
    # result.prediction
    # # 0.0

if __name__ == '__main__':
    # estimator_transformer()
    # pipe_line()
    # logistic_regression()
    # decision_tree_classifier()
    # random_forest_classifier()
    naive_bayes()