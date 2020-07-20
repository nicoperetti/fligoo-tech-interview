#!/usr/bin/env python
# coding: utf-8
import click
import pickle
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from features.transformers import ItemSelector
from sklearn.linear_model import LogisticRegression


@click.command()
@click.option('--input_path', type=click.STRING, help='Path to input file')
@click.option('--output_path', type=click.STRING, help='Path to output dir')
def train_script(input_path, output_path):
    df = pd.read_csv(input_path)

    X = df[["product_id", "default"]]
    y = df["cancelled"]

    transformer_list = [("pid", Pipeline(steps=[
                            ('pid_selector', ItemSelector(key="product_id")),
                            ('pid_one_hot', OneHotEncoder(categories='auto',
                                                          handle_unknown='ignore'))])),
                        ('default', ItemSelector(key="default"))]

    model = Pipeline([
        ("features", FeatureUnion(transformer_list=transformer_list)),
        ("clf", LogisticRegression(C=1e-05, class_weight='balanced'))
    ])

    model.fit(X, y)

    pickle.dump(model, open(output_path, 'wb'))


if __name__ == "__main__":
    train_script()
