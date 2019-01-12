#!/usr/bin/env bash

PREDICT_LOCAL_PATH='/home/cloudera/practice/task20a/predict_iris.csv'
hadoop fs -put -f ${PREDICT_LOCAL_PATH} predict_iris_hdfs.csv
hbase org.apache.hadoop.hbase.mapreduce.ImportTsv -Dimporttsv.separator="," -Dimporttsv.columns=HBASE_ROW_KEY,cf:Iris setosa,cf:Iris versicolor,cf:Iris virginica iris_predict predict_iris_hdfs.csv
