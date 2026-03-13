# Used Car Price Prediction – FOR EXPERIMENTATION

## Project Overview

  #### This project implements a complete machine learning pipeline for predicting used car prices.
  <li>
  Modular pipeline architecture
</li>
<li>Experiment tracking with DVC</li>
<li>Live metric logging using DVCLive</li>
<li>Reproducible experiments</li>
<li>Parameterized training via params.yaml</li>
<li>Demonstrated model using Streamlit</li>


##### The entire workflow is production-oriented and reproducible.


## ML Pipeline Workflow

The pipeline is orchestrated using DVC and consists of:

1. Data Ingestion  
2. Data Validation  
3. Data Preprocessing  
4. Model Training  
5. Model Evaluation  
6. Best Model Selection  
7. Artifact Storage in AWS S3


## Docker Integration

The project now includes a Dockerfile that sets up the entire environment with all dependencies.
Allows running the ML pipeline and Streamlit app in a containerized setup:

### Build Docker image
docker build -t usercarprediction-app .

### Run container
docker run -p 5000:5000 usercarprediction-app

Ensures reproducibility across different machines and avoids environment-related issues.
Compatible with CI/CD pipelines for automated deployment.

## ML-Pipeline flow

<img width="648" height="753" alt="image" src="https://github.com/user-attachments/assets/c310042b-db44-48c6-9816-2b6aec777242" />


## Application
<img width="655" height="908" alt="image" src="https://github.com/user-attachments/assets/b2de8113-1395-4ed7-9ad6-f17a9e787b2a" />


  


