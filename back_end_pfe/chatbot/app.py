# from flask import Flask, render_template, request, jsonify
# from chat import get_response

# app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.post("/predict")
# def predict():
#     text= request.get_json().get("message")
#     #check if text is valid 
#     response= get_response(text)
#     message={"answer":response}
#     return jsonify(message)

# if __name__=="__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request,jsonify
# # from flask_cors import CORS
# app = Flask(__name__)
# # CORS(app, origins='http://localhost:4200')
# # CORS(app)

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict')
# def predict():
#     text= request.get_json().get("message")
#     #check if text is valid 
#     response= get_response(text)
#     message={"answer":response}
#     return jsonify(message)
# if __name__=="__main__":
#     app.run(debug=True)




from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from chat import get_response
app = Flask(__name__)
# CORS(app, origins='http://localhost:4200')
CORS(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text= request.get_json().get("message")
    #check if text is valid 
    response= get_response(text)
    message={"answer":response}
    return jsonify(message)
if __name__=="__main__":
    app.run(debug=True)