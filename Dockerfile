# write some code to build your image
FROM python:3.8.12-bullseye

COPY model.joblib /model.joblib
COPY api /api
COPY requirements.txt /requirements.txt
COPY find_your_inner_gamer /find_your_inner_gamer
# COPY /Users/laurabonnet/Documents/GITHUBK/main-cyclist-337816-8df14917206d.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt



CMD uvicorn api.gamer:app --host 0.0.0.0 --port $PORT
