FROM python:3.8.6-slim
WORKDIR /Bert-app
ADD . /Bert-app
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN python load_models.py
CMD ["python","app.py"]