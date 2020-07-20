# fligoo-tech-interview
Take Home

##Download data and unzip it into data directory
url: https://s3-us-west-2.amazonaws.com/fligoo.data-science/TechInterviews/BankChurn/data.zip

## Docker Build
docker-compose build

## Docker run
docker run -v $(pwd):/code/ -p 5000:5000 -it backchurn:dev bash

## Run jupyter
jupyter notebook --allow-root --ip=0.0.0.0 --port 5000

## Train
make train

## Command help
make help

## load API with the model exposed
docker-compose up

## Request example
import requests
url = 'http://0.0.0.0:5000/predict'
data = {'pid':"asd", 'default': 0}
resp = requests.get(url, json=data)
print(resp.json())