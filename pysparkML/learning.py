#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wicle Qian
# 2015.11.19
# test the python in Spark without pyspark


from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import numpy as np


def linear_regression():
    conf = SparkConf().setAppName('RF')
    sc = SparkContext(conf=conf)
    raw_data = sc.textFile("./data/hour_noheader.csv")
    # raw_data = spark.read.format("csv").option("header", "true").csv("./data/hour.csv")

    num_data = raw_data.count()
    records = raw_data.map(lambda x: x.split(","))
    first = records.first()
    print(first)
    print(num_data)

    #cache data
    def get_mapping(rdd, idx):
        return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

    print("maping first categorical feature conlumn: %s" % get_mapping(records, 2))

    mappings = [get_mapping(records, i) for i in range(2, 10)]
    print("mappings is:" + str(mappings))
    cat_len = sum(map(len, mappings))
    num_len = len(records.first()[10:14])
    total_len = num_len + cat_len
    print("Feature vector length for categorical features: %d" % cat_len)
    print("Feature vector length for numerical features: %d" % num_len)
    print("Total feature vector length: %d" % total_len)

    # 提取特征
    def extract_features(record):
        cat_vec = np.zeros(cat_len)
        i = 0
        step = 0
        for field in record[2:9]:
            m = mappings[i]
            idx = m[field]
            cat_vec[idx + step] = 1
            i = i + 1
            step = step + len(m)
        num_vec = np.array([float(field) for field in record[10:14]])
        return np.concatenate((cat_vec, num_vec))

    # 提取标签
    def extract_label(record):
        return float(record[-1])

    data = records.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

    first_point = data.first()
    print("Raw data: " + str(first[2:]))
    print("Label: " + str(first_point.label))
    print("Linear Model feature vector:\n" + str(first_point.features))
    print("Linear Model feature vector length: " + str(len(first_point.features)))

    # 创建决策树模型特征向量
    def extract_features_dt(record):
        return np.array(map(float, record[2:14]))

    data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))

    first_point_dt = data_dt.first()
    print("Decision Tree feature vector: " + str(first_point_dt.features))
    print("Decision Tree feature vector length: " + str(len(first_point_dt.features)))

    #训练线性模型并测试预测效果
    linear_model = LinearRegressionWithSGD.train(data, iterations=10000, step=0.1, intercept=False)
    true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
    print("Linear Model predictions: " + str(true_vs_predicted.take(5)))

    #训练决策树模型并测试预测效果
    dt_model = DecisionTree.trainRegressor(data_dt, {})
    preds = dt_model.predict(data_dt.map(lambda p: p.features))
    actual = data.map(lambda p: p.label)
    true_vs_predicted_dt = actual.zip(preds)

    print("Decision Tree predictions: " + str(true_vs_predicted_dt.take(5)))
    print("Decision Tree depth: " + str(dt_model.depth()))
    print("Decision Tree number of nodes: " + str(dt_model.numNodes()))

    #评估回归模型的方法：
    """
        均方误差(MSE, Mean Sequared Error)
        均方根误差(RMSE, Root Mean Squared Error)
        平均绝对误差(MAE, Mean Absolute Error)
        R-平方系数(R-squared coefficient)
        均方根对数误差(RMSLE)
    """

    # 均方误差&均方根误差
    def squared_error(actual, pred):
        return (pred - actual) ** 2

    mse = true_vs_predicted.map(lambda t, p: squared_error(t, p)).mean()
    mse_dt = true_vs_predicted_dt.map(lambda t, p: squared_error(t, p)).mean()

    cat_features = dict([(i - 2, len(get_mapping(records, i)) + 1) for i in range(2, 10)])

    # train the model again
    dt_model_2 = DecisionTree.trainRegressor(data_dt, categoricalFeaturesInfo=cat_features)
    preds_2 = dt_model_2.predict(data_dt.map(lambda p: p.features))
    actual_2 = data.map(lambda p: p.label)
    true_vs_predicted_dt_2 = actual_2.zip(preds_2)

    # compute performance metrics for decision tree model
    mse_dt_2 = true_vs_predicted_dt_2.map(lambda t, p: squared_error(t, p)).mean()

    print("Linear Model - Mean Squared Error: %2.4f" % mse)
    print("Decision Tree - Mean Squared Error: %2.4f" % mse_dt)
    print("Categorical feature size mapping %s" % cat_features)
    print("Decision Tree [Categorical feature]- Mean Squared Error: %2.4f" % mse_dt_2)

    #  均方根对数误差
    def squared_log_error(pred, actual):
        return (np.log(pred + 1) - np.log(actual + 1)) ** 2

    rmsle = np.sqrt(true_vs_predicted.map(lambda t, p: squared_log_error(t, p)).mean())
    rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda t, p: squared_log_error(t, p)).mean())
    rmsle_dt_2 = np.sqrt(true_vs_predicted_dt_2.map(lambda t, p: squared_log_error(t, p)).mean())

    print("Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle)
    print("Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt)
    print("Decision Tree [Categorical feature]- Root Mean Squared Log Error: %2.4f" % rmsle_dt_2)


    # 改进和调优
    # targets = records.map(lambda r: float(r[-1])).collect()
    # hist(targets, bins=40, color='lightblue', normed=True)
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(16, 10)

def random_forest():
    conf = SparkConf().setAppName('RF')
    sc = SparkContext(conf=conf)
    # print("\npyspark version:" + str(sc.version) + "\n")

    data = MLUtils.loadLibSVMFile(sc, './data/sample_libsvm_data.txt')
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    model = RandomForest.trainClassifier(trainingData, numClasses=2,
                                         categoricalFeaturesInfo={}, numTrees=3,
                                         featureSubsetStrategy="auto", impurity='gini',
                                         maxDepth=4, maxBins=32)

    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda v, p: v != p).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    print(model.toDebugString())
    # Save and load model
    model.save(sc, ".model/myRandomForestClassificationModel")
    sameModel = RandomForestModel.load(sc, "./model/myRandomForestClassificationModel")


if __name__ == '__main__':
    random_forest()
