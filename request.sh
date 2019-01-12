#!/usr/bin/env bash

RAW_DATA="$(hadoop fs -cat insert_data_hdfs.json)"
PREDICTION= curl -H "Content-type: application/json" -d "${RAW_DATA}" 'http://localhost:9999/prediction'
