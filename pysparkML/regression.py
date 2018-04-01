from pyspark.sql import SparkSession
from numpy import allclose
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeRegressionModel,\
    GBTRegressor, GBTRegressionModel,\
    GeneralizedLinearRegression, GeneralizedLinearRegressionModel,\
    LinearRegression, LinearRegressionModel, \
    RandomForestRegressor, RandomForestRegressionModel


def decision_tree_regressor():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    dt = DecisionTreeRegressor(maxDepth=2, varianceCol="variance")
    model = dt.fit(df)
    model.depth
    # 1
    model.numNodes
    # 3
    model.featureImportances
    # SparseVector(1, {0: 1.0})
    model.numFeatures
    # 1
    test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    model.transform(test0).head().prediction
    # 0.0
    test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    model.transform(test1).head().prediction
    # 1.0
    temp_path = "./"
    dtr_path = temp_path + "/dtr"
    dt.save(dtr_path)
    dt2 = DecisionTreeRegressor.load(dtr_path)
    dt2.getMaxDepth()
    # 2
    model_path = temp_path + "/dtr_model"
    model.save(model_path)
    model2 = DecisionTreeRegressionModel.load(model_path)
    model.numNodes == model2.numNodes
    # True
    model.depth == model2.depth
    # True
    model.transform(test1).head().variance
    # 0.0


def GBT_regressor():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    gbt = GBTRegressor(maxIter=5, maxDepth=2, seed=42)
    print(gbt.getImpurity())
    # variance
    model = gbt.fit(df)
    model.featureImportances
    # SparseVector(1, {0: 1.0})
    model.numFeatures
    # 1
    allclose(model.treeWeights, [1.0, 0.1, 0.1, 0.1, 0.1])
    # True
    test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    model.transform(test0).head().prediction
    # 0.0
    test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    model.transform(test1).head().prediction
    # 1.0
    temp_path = "./"
    gbtr_path = temp_path + "gbtr"
    gbt.save(gbtr_path)
    gbt2 = GBTRegressor.load(gbtr_path)
    gbt2.getMaxDepth()
    # 2
    model_path = temp_path + "gbtr_model"
    model.save(model_path)
    model2 = GBTRegressionModel.load(model_path)
    model.featureImportances == model2.featureImportances
    # True
    model.treeWeights == model2.treeWeights
    # True
    model.trees
    # [DecisionTreeRegressionModel (uid=...) of depth..., DecisionTreeRegressionModel...]


def generalized_linear_regression():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame(
        [(1.0, Vectors.dense(0.0, 0.0)), (1.0, Vectors.dense(1.0, 2.0)), (2.0, Vectors.dense(0.0, 0.0)),
         (2.0, Vectors.dense(1.0, 1.0)), ], ["label", "features"])
    glr = GeneralizedLinearRegression(family="gaussian", link="identity",)  # linkPredictionCol="p")
    model=glr.fit(df)
    transformed = model.transform(df)
    abs(transformed.head().prediction - 1.5) < 0.001
    # True
    abs(transformed.head().p - 1.5) < 0.001
    # True
    model.coefficients
    model.numFeatures
    # 2
    abs(model.intercept - 1.5) < 0.001
    # True
    temp_path = "./"
    glr_path = temp_path + "/glr"
    glr.save(glr_path)
    glr2 = GeneralizedLinearRegression.load(glr_path)
    glr.getFamily() == glr2.getFamily()
    # True
    model_path = temp_path + "/glr_model"
    model.save(model_path)
    model2 = GeneralizedLinearRegressionModel.load(model_path)
    model.intercept == model2.intercept
    # True
    model.coefficients[0] == model2.coefficients[0]
    # True


def linear_regression():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([(1.0, 2.0, Vectors.dense(1.0)), (0.0, 2.0, Vectors.sparse(1, [], []))],
                               ["label", "weight", "features"])
    lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal", weightCol="weight")
    model = lr.fit(df)
    test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    abs(model.transform(test0).head().prediction - (-1.0)) < 0.001
    # True
    abs(model.coefficients[0] - 1.0) < 0.001
    # True
    abs(model.intercept - 0.0) < 0.001
    # True
    test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    abs(model.transform(test1).head().prediction - 1.0) < 0.001
    # True
    lr.setParams("vector")
    # Traceback (most recent call last):
    #    ...
    # TypeError: Method setParams forces keyword arguments.
    temp_path = "./"
    lr_path = temp_path + "/lr"
    lr.save(lr_path)
    lr2 = LinearRegression.load(lr_path)
    lr2.getMaxIter()
    # 5
    model_path = temp_path + "/lr_model"
    model.save(model_path)
    model2 = LinearRegressionModel.load(model_path)
    model.coefficients[0] == model2.coefficients[0]
    # True
    model.intercept == model2.intercept
    # True
    model.numFeatures
    # 1


def RandomForestRegressor():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.createDataFrame([(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
    rf = RandomForestRegressor(numTrees=2, maxDepth=2, seed=42)
    model = rf.fit(df)
    model.featureImportances
    # SparseVector(1, {0: 1.0})
    allclose(model.treeWeights, [1.0, 1.0])
    # True
    test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
    model.transform(test0).head().prediction
    # 0.0
    model.numFeatures
    # 1
    model.trees
    # [DecisionTreeRegressionModel (uid=...) of depth..., DecisionTreeRegressionModel...]
    model.getNumTrees
    # 2
    test1 = spark.createDataFrame([(Vectors.sparse(1, [0], [1.0]),)], ["features"])
    model.transform(test1).head().prediction
    # 0.5
    temp_path = "./"
    rfr_path = temp_path + "/rfr"
    rf.save(rfr_path)
    rf2 = RandomForestRegressor.load(rfr_path)
    rf2.getNumTrees()
    # 2
    model_path = temp_path + "/rfr_model"
    model.save(model_path)
    model2 = RandomForestRegressionModel.load(model_path)
    model.featureImportances == model2.featureImportances
    # True
