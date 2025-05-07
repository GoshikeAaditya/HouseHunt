# House Hunt: House Recommendation and Price Prediction System

## Overview

House Hunt is a machine learning-powered application that provides:
- Personalized house recommendations based on user preferences
- House price prediction using property features

It features a REST API (FastAPI), a Streamlit web app, and user authentication.

---

## Technical Features

- **Languages & Frameworks**: Python, FastAPI, Streamlit, Pydantic
- **Machine Learning**: RandomForestRegressor (price prediction), NearestNeighbors (recommendation)
- **Preprocessing**: StandardScaler, LabelEncoder
- **Data Handling**: pandas, numpy
- **Model Serialization**: joblib
- **API Endpoints**: `/recommend`, `/predict_price`
- **Frontend**: Streamlit app with login, registration, recommendations, and price prediction pages
- **User Authentication**: Credentials stored in a JSON file
- **Data Storage**: CSV for dataset, JSON for credentials

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt