# Tech-Interview
Take Home: Churn Prediction

## Download data and unzip it into data directory
url: https://s3-us-west-2.amazonaws.com/fligoo.data-science/TechInterviews/BankChurn/data.zip

## Docker Build
```sh
$ docker-compose build
```

## [Optional] Run jupyter notebooks
```sh
$ docker run -v $(pwd):/code/ -p 5000:5000 -it bank-churn:dev bash
```

```sh
$ jupyter notebook --allow-root --ip=0.0.0.0 --port 5000
```

- notebooks:
    -  EDA.ipynb
    -  Baseline.ipynb


## Train Model & Exposed into an API.
```sh
$ docker-compose up
```

### Request example
```python
import requests
url = 'http://0.0.0.0:5000/predict'
data = {'pid':"IS0002", 'default': 0}
resp = requests.get(url, json=data)
print(resp.json())
```