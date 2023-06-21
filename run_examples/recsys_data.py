#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script preprocesses Criteo dataset tsv files to binary (npy) files.

import pandas as pd
import numpy as np
import glob
import math
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from category_encoders import *
import os
import sys
import argparse
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="recsys preprocessing script."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing Recsys tsv files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to store files.",
    )
    return parser.parse_args(argv)


def get_df_from_filepath(data_path):
    all_files = glob.glob(data_path)
    df = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)        
    return df


def get_train_test_df(input_dir, test_date):
    train_data_path = f'{input_dir}/train/*.csv'    
    test_data_path = f'{input_dir}/test/*.csv'    

    train_df = get_df_from_filepath(train_data_path)
    test_df = get_df_from_filepath(test_data_path)
    test_df['is_clicked'] = test_df['is_installed'] = test_df['f_0']
    total_df = pd.concat((train_df, test_df), ignore_index=True)

    test_df = total_df.loc[total_df['f_1'] == test_date]
    train_df = total_df.loc[total_df['f_1'] < test_date]

    print('before process shape')
    print(train_df.shape, test_df.shape)
    return train_df, test_df


def add_rating(df_train):
    def f(df):
        label_dict = {
            '00': 0, '10': 1,
            '01': 2, '11': 3
        }
        df = df.copy()
        df['rating'] = df['is_clicked'].astype(str) + df['is_installed'].astype(str)
        df['rating'] = df['rating'].map(label_dict)
        return df    
    
    df_train = f(df_train)
    return df_train


def filter_df(df_train):
    df_train = df_train[df_train['rating'] > 0]
    return df_train


def fit_transform(encoder, is_combine, ori_columns, 
                  after_columns, df_train, df_test=None):
    if is_combine and df_test is not None:
        df_all = pd.concat((df_train, df_test), ignore_index=True)
        encoder.fit(df_all[ori_columns])
        del df_all
    else:
        encoder.fit(df_train[ori_columns])
    
    df_train[after_columns] = encoder.transform(df_train[ori_columns])
    if df_test is not None:
        df_test[after_columns] = encoder.transform(df_test[ori_columns])

    return df_train, df_test


def categorify_cat_feat(df_train, df_test=None):
    def get_cat_id(cat_columns, df_train, is_combine=True, df_test=None):
        if not cat_columns:
            return df_train, df_test
        fill_value = 0
        cat_enc = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=fill_value)),
            (
                "categorify-1", 
                 preprocessing.OrdinalEncoder(
                     dtype=np.int64,
                     # encoded_missing_value=-1,
                     handle_unknown='use_encoded_value',
                     unknown_value=-1,
                    #  min_frequency=1,
                 )
            ),
        ])

        df_train, df_test = fit_transform(cat_enc, is_combine, cat_columns, 
                  cat_columns, df_train, df_test)
        
        if not is_combine:
            for column in cat_columns:
                if -1 in df_test[column].unique():
                    df_train[column] += 1
                    df_test[column] += 1
        return df_train, df_test
        
    basic_cat_columns = BASIC_CAT_COLUMNS
    df_train, df_test = get_cat_id(basic_cat_columns, df_train, is_combine=IS_COMBINE, df_test=df_test)
    
    return df_train, df_test


def get_processed_df(df_train, df_test=None):

    df_train = add_rating(df_train)

    if IS_FILTER:
        df_train = filter_df(df_train)

    df_train, df_test = categorify_cat_feat(df_train, df_test)

    return df_train, df_test


def save_output_df(df_train, df_test, output_dir):
    train_save_cols = BASIC_CAT_COLUMNS + ['rating', 'f_1']
    export_save_cols = BASIC_CAT_COLUMNS + ['f_0', 'f_1']

    train_save_path = f'{output_dir}/u.data'
    export_save_path = f'{output_dir}/export'

    df_export = pd.concat((df_train, df_test), ignore_index=True)[export_save_cols]
    for col in export_save_cols:
        print(f'col: {col}, max value: {df_export[col].max()}')
    df_train = df_train[train_save_cols]
    
    print('after process shape')
    print(df_train.shape)
    print(df_train.head())

    df_train.to_csv(train_save_path, sep='\t', header=False, index=False)
    df_export.to_csv(export_save_path, sep='\t', header=False, index=False)
    

def main(argv: List[str]) -> None:
    """
    This function preprocesses the raw Criteo tsvs into the format (npy binary)
    expected by InMemoryBinaryCriteoIterDataPipe.

    Args:
        argv (List[str]): Command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.system(f'mkdir -p {output_dir}')

    train_df, test_df = get_train_test_df(input_dir, TEST_DATE)

    train_df, test_df = get_processed_df(train_df, test_df)

    save_output_df(train_df, test_df, output_dir)

    return


if __name__ == "__main__":
    TEST_DATE = 67
    IS_FILTER = False
    IS_COMBINE = False
    BASIC_CAT_COLUMNS = ['f_15', 'f_6']
    main(sys.argv[1:])


# python recsys_data.py \
#    --input_dir '/home/vmagent/app/data/sharechat_recsys2023_data' \
#    --output_dir 'data/ml-100k'