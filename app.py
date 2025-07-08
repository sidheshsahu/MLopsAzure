# from flask import Flask, render_template, request
# from src.pipeline.predict_pipeline import CustomData, PredictPipeline
# import os


# app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if request.method == 'POST':
#             feature1 = request.form.get('CGPA')
#             feature2 = request.form.get('Internships')
#             feature3 = request.form.get('Projects')
#             feature4 = request.form.get('Workshops/Certifications')
#             feature5 = request.form.get('AptitudeTestScore')
#             feature6 = request.form.get('SoftSkillsRating')
#             feature7 = request.form.get('ExtracurricularActivities')
#             feature8 = request.form.get('PlacementTraining')
#             feature9 = request.form.get('SSC_Marks')
#             feature10 = request.form.get('HSC_Marks')
           
            
#             data = CustomData(
#                 feature1=float(feature1),
#                 feature2=int(feature2),
#                 feature3=int(feature3),
#                 feature4=int(feature4),
#                 feature5=int(feature5),
#                 feature6=float(feature6),
#                 feature7=str(feature7),
#                 feature8=str(feature8),
#                 feature9=int(feature9),
#                 feature10=int(feature10)
                
#             )

#             # Convert to DataFrame
#             df = data.to_dataframe()

#             # Call model prediction
#             pipeline = PredictPipeline()
#             result = pipeline.predict(df)
            
#             placement = "Placed" if result[0] == 1 else "Not Placed"
#             return render_template('index.html', prediction_text=f"Prediction: {placement}")

#     except Exception as e:
#         return str(e)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=80,debug=True)
    







from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    CGPA: float = Form(...),
    Internships: int = Form(...),
    Projects: int = Form(...),
    Workshops_Certifications: int = Form(...),
    AptitudeTestScore: int = Form(...),
    SoftSkillsRating: float = Form(...),
    ExtracurricularActivities: str = Form(...),
    PlacementTraining: str = Form(...),
    SSC_Marks: int = Form(...),
    HSC_Marks: int = Form(...)
):
    try:
        data = CustomData(
            feature1=CGPA,
            feature2=Internships,
            feature3=Projects,
            feature4=Workshops_Certifications,
            feature5=AptitudeTestScore,
            feature6=SoftSkillsRating,
            feature7=ExtracurricularActivities,
            feature8=PlacementTraining,
            feature9=SSC_Marks,
            feature10=HSC_Marks
        )

        df = data.to_dataframe()
        pipeline = PredictPipeline()
        result = pipeline.predict(df)

        placement = "Placed" if result[0] == 1 else "Not Placed"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction_text": f"Prediction: {placement}"
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction_text": f"Error: {str(e)}"
        })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=80, reload=True)

