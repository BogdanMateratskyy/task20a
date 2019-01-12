import pandas as pd
import iris_model.serving.data_science as model
from flask import Flask, request, abort


app = Flask(__name__)
model_path = "/home/cloudera/practice/task20a/iris_model/model"
result_csv = "/home/cloudera/practice/task20a/predict_iris.csv"


@app.route("/prediction", methods=["GET"])
def get_prediction():
    iris_prediction = pd.read_csv(result_csv)
    return iris_prediction.to_json(orient="records")


@app.route("/prediction", methods=["POST"])
def post_prediction():
    try:
        new_record = pd.read_json(request.data)
        result = model.predict(model_path, new_record)
        df = pd.DataFrame.from_records(result, columns=["Iris setosa", "Iris versicolor", "Iris virginica"])
        print(df)
        df.to_csv('predict_iris.csv')
        return "Prediction was successfully created to the next path: {}".format(result_csv)
    except Exception as ex:
        abort(400, ex)
    # return "Prediction was successfully created to the next path: {}".format(result_csv)


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = "9999"
    app.run(host=HOST, port=PORT)
