FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r app/requirements.txt
EXPOSE 8080
CMD ["uvicorn", "app.serving:app", "--host", "0.0.0.0", "--port", "8080"]
