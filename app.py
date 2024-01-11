from flask import Flask , render_template, request
import numpy as np
import pickle

app=Flask(__name__)

model = pickle.load(open('water.pkl','rb'))
model2 = pickle.load(open('air.pkl','rb'))
@app.route("/")
def home():
    return render_template('main.html')

@app.route('/water', methods=['GET'])
def water_form():
    return render_template('water.html')

@app.route('/water',methods=['POST'])
def water():
    ph = request.form["ph"]
    hard = request.form["hard"]
    Solids = request.form["Solids"]
    chloro = request.form["chloro"]
    Sulphate = request.form["Sulphate"]
    Conductivity = request.form["Conductivity"]
    org = request.form["org"]
    trihal= request.form["trihal"]
    Turb = request.form ["Turb"]

    arr = np.array([[ph,hard,Solids,chloro,Sulphate,Conductivity,org,trihal,Turb]])
    pred = model.predict(arr)
    return render_template('waterfinal.html',data = pred)

@app.route('/air', methods=['GET'])
def air_form():
    return render_template('air.html')

@app.route('/air',methods=['POST'])
def air():
    so2 = request.form["SO2"]
    no2 = request.form["NO2"]
    rspm = request.form["RSPM"]
    spm = request.form["SPM"]
    pm2_5 = request.form["PM2_5"]

    arr1 = np.array([[so2,no2,rspm,spm,pm2_5]],dtype= float)
    pred1 = model2.predict(arr1)
    return render_template('airfinal.html',air = pred1)


     

if __name__ == "__main__":
      app.run(debug=True)