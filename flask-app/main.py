import pandas as pd
import pickle

from flask import Flask, flash, jsonify, request, render_template
from werkzeug.utils import redirect

from forms import JsonForm


# Загружаем модель в память
with open('./model.pkl', 'rb') as model_pkl:
   knn = pickle.load(model_pkl)

# Инициализируем приложение Flask
app = Flask(__name__)
app.config.update(
    ENV="development",
    SECRET_KEY="qwertytrewsupersecret",
)

@app.get("/hello/")
def hello_name():
    # print_request()

    # name = "World"
    name = request.args.get("name", "")
    name = name.strip()
    if not name:
        name = "World"
    return {"message": f"Hello, {name}!"}


@app.route('/', methods=['POST'])
def apicall():
    try:
        text_json = request.get_json()
        unseen_data = pd.read_json(text_json)
        diag_ids = unseen_data['id']
    except Exception:
        raise Exception

    predictions = knn.predict(unseen_data)
    final_predictions = pd.DataFrame(list(zip(diag_ids, list(pd.Series(predictions)))))
    # res = {}
    # for i in range(len(diag_ids)):
    #     res[diag_ids[i]] = predictions[i]
    #


    responses = jsonify(predictions=final_predictions.to_json())
    responses.status_code = 200

    return (responses)



@app.route('/predict', methods=['GET','POST'], endpoint="predict")
def predict_diagnosis():
    form = JsonForm()
    if request.method == "GET":
        return render_template("predict.html", form=form)
    text = form.input_data.data
    try:
        unseen_data = pd.read_json(text)
    except ValueError:
        return redirect('/predict')

    diag_ids = unseen_data['id']
    predictions = knn.predict(unseen_data)

    for i in range(len(diag_ids)):
        if predictions[i]:
            flash(f"Predicted result for observation {diag_ids[i]} is: MALIGNANT", "danger")
        else:
            flash(f"Predicted result for observation {diag_ids[i]} is: BENIGN", "success")

    return redirect('/predict')

if __name__ == "__main__":
    app.run(debug=True)