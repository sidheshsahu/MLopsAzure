
# End-to-End MLOps Pipeline for Student Placement Prediction

An end-to-end MLOps pipeline that automates the entire lifecycle of a machine learning model for student placement prediction — from data ingestion and preprocessing to model training, experiment tracking, Dockerization, and deployment on Azure using ACR and Web App services.





##  Project Overview
This project demonstrates a production-ready MLOps workflow for building, tracking, and deploying a student placement prediction model using:

✅ MLflow for experiment tracking and model management

🐳 Docker for containerizing the ML application

☁️ Azure Container Registry (ACR) for hosting the image

🌐 Azure Web App for cloud-based model serving

🧪 Custom logging & exception handling for observability and debugging




## Key Features

📥 Data Ingestion & Preprocessing – Clean, transform, and prepare input data

🔍 Model Training – Trains a classification model with hyperparameter support

📈 MLflow Tracking – Logs model parameters, metrics, and artifacts

🐳 Dockerized Application – Fully containerized with a lightweight Flask server

☁️ Azure Deployment – Pushes image to Azure Container Registry and deploys on Azure Web App

📋 Custom Logger & Exception Handling – For better traceability
## Tech Stack

| Tool          | Purpose                                |
| ------------- | -------------------------------------- |
| Python        | Core programming language              |
| Flask         | API framework for serving the model    |
| MLflow        | Experiment tracking and model registry |
| Docker        | Containerization of the application    |
| Azure ACR     | Hosting container images               |
| Azure Web App | Deploying ML container on the cloud    |
| HTML/CSS      | Simple front-end for input & output    |
