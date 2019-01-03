#!/usr/bin/env bash

RAW_PATH='/home/cloudera/practice/task20a/data.json'
hadoop fs -put ${RAW_PATH} insert_data_hdfs.json
hadoop fs -cat insert_data_hdfs.json