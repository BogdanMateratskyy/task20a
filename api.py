import pandas as pd
import iris_model.serving.data_science as model
from flask import Flask

app = Flask(__name__)

model_path = "/home/cloudera/practice/task20a/iris_model/model"
data_frame = "/home/cloudera/practice/task20a/data.json"


@app.route("/prediction")
def prediction():
    result = model.predict(model_path, data_frame)
    df = pd.DataFrame.from_records(result, columns=["Iris setosa", "Iris versicolor", "Iris virginica"])
    print(df)
    return df
    # with open(model_path, "rb") as pickle_file:
    #     loaded_model = joblib.load(pickle_file)
    #     result = loaded_model.predict_proba(data_frame)
    #     df = pd.DataFrame.from_records(result, columns=["Iris setosa","Iris versicolor","Iris virginica"])
    #     OutputDataSet = df


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = "9999"
    app.run(host=HOST, port=PORT)
