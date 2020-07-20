FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get update && apt-get install build-essential -y

WORKDIR /code
ENV PYTHONPATH /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /code

EXPOSE 5000

CMD ["/bin/bash"]