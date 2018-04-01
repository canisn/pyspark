from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import BisectingKMeans,\
    KMeans, KMeansModel,\
    GaussianMixture, GaussianMixtureModel


def bisecting_k_means():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),(Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
    df = spark.createDataFrame(data, ["features"])
    bkm = BisectingKMeans(k=2, minDivisibleClusterSize=1.0)
    model = bkm.fit(df)
    centers = model.clusterCenters()
    len(centers)
    model.computeCost(df)
    model.hasSummary
    summary = model.summary
    summary.k
    summary.clusterSizes
    #预测
    transformed = model.transform(df).select("features", "prediction")
    rows = transformed.collect()
    rows[0].prediction == rows[1].prediction
    rows[2].prediction == rows[3].prediction


def k_means():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    from pyspark.ml.linalg import Vectors
    data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),), (Vectors.dense([9.0, 8.0]),),
            (Vectors.dense([8.0, 9.0]),)]
    df = spark.createDataFrame(data, ["features"])
    kmeans = KMeans(k=2, seed=1)
    model = kmeans.fit(df)
    centers = model.clusterCenters()
    len(centers)
    # 2
    model.computeCost(df)
    # 2.000...
    transformed = model.transform(df).select("features", "prediction")
    rows = transformed.collect()
    rows[0].prediction == rows[1].prediction
    # True
    rows[2].prediction == rows[3].prediction
    # True
    model.hasSummary
    # True
    summary = model.summary
    summary.k
    # 2
    summary.clusterSizes
    # [2, 2]
    temp_path = "./"
    kmeans_path = temp_path + "/kmeans"
    kmeans.save(kmeans_path)
    kmeans2 = KMeans.load(kmeans_path)
    kmeans2.getK()
    # 2
    model_path = temp_path + "/kmeans_model"
    model.save(model_path)
    model2 = KMeansModel.load(model_path)
    model2.hasSummary
    # False
    model.clusterCenters()[0] == model2.clusterCenters()[0]
    # array([ True,  True], dtype=bool)
    model.clusterCenters()[1] == model2.clusterCenters()[1]
    # array([ True,  True], dtype=bool)


def gaussian_mixture():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    data = [(Vectors.dense([-0.1, -0.05]),), (Vectors.dense([-0.01, -0.1]),), (Vectors.dense([0.9, 0.8]),),
            (Vectors.dense([0.75, 0.935]),), (Vectors.dense([-0.83, -0.68]),), (Vectors.dense([-0.91, -0.76]),)]
    df = spark.createDataFrame(data, ["features"])
    gm = GaussianMixture(k=3, tol=0.0001, maxIter=10, seed=10)
    model = gm.fit(df)
    model.hasSummary
    # True
    summary = model.summary
    summary.k
    # 3
    summary.clusterSizes
    # [2, 2, 2]
    weights = model.weights
    len(weights)
    # 3
    model.gaussiansDF.show()
    transformed = model.transform(df).select("features", "prediction")
    rows = transformed.collect()
    rows[4].prediction == rows[5].prediction
    # True
    rows[2].prediction == rows[3].prediction
    # True
    temp_path = "./"
    gmm_path = temp_path + "/gmm"
    gm.save(gmm_path)
    gm2 = GaussianMixture.load(gmm_path)
    gm2.getK()
    # 3
    model_path = temp_path + "/gmm_model"
    model.save(model_path)
    model2 = GaussianMixtureModel.load(model_path)
    model2.hasSummary
    # False
    model2.weights == model.weights
    # True
    model2.gaussiansDF.show()


if __name__ == '__main__':
    gaussian_mixture()
