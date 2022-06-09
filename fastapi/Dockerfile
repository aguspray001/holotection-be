FROM tiangolo/uvicorn-gunicorn:python3.9-slim

RUN mkdir /app

COPY requirements.txt /app

COPY . /app

WORKDIR /app

CMD ["uvicorn", "main:app" "--reload"]