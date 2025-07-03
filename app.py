from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            feature1 = request.form.get('CGPA')
            feature2 = request.form.get('Internships')
            feature3 = request.form.get('Projects')
            feature4 = request.form.get('Workshops/Certifications')
            feature5 = request.form.get('AptitudeTestScore')
            feature6 = request.form.get('SoftSkillsRating')
            feature7 = request.form.get('ExtracurricularActivities')
            feature8 = request.form.get('PlacementTraining')
            feature9 = request.form.get('SSC_Marks')
            feature10 = request.form.get('HSC_Marks')
           
            
            data = CustomData(
                feature1=float(feature1),
                feature2=int(feature2),
                feature3=int(feature3),
                feature4=int(feature4),
                feature5=int(feature5),
                feature6=float(feature6),
                feature7=str(feature7),
                feature8=str(feature8),
                feature9=int(feature9),
                feature10=int(feature10)
                
            )

            # Convert to DataFrame
            df = data.to_dataframe()

            # Call model prediction
            pipeline = PredictPipeline()
            result = pipeline.predict(df)
            
            placement = "Placed" if result[0] == 1 else "Not Placed"
            return render_template('index.html', prediction_text=f"Prediction: {placement}")

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80,debug=True)
    



