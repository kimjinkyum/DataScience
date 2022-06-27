import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def change_hour(x):
    return x["Timestamp"].split()[1].split(":")[0]


def change_ip_class(x):
    parts = x.split(".")
    one_gram = int(parts[0])
    if 0 <= one_gram <= 127:
        class_ip = "A"
    elif 128 <= one_gram <= 191:
        class_ip = "B"
    elif 192 <= one_gram <= 223:
        class_ip = "C"
    elif 224 <= one_gram <= 239:
        class_ip = "D"
    else:
        class_ip = "E"
    return class_ip


# data, strategy = "mean", "median"
def fill_missing_value(data, strategy):
    imputer = SimpleImputer(strategy=strategy)
    data[["Protocol", 'Total Length of Fwd Packet']] = imputer.fit_transform(
        data[["Protocol", 'Total Length of Fwd Packet']])
    return data


def scale_value(data, strategy):
    if strategy == "standard":
        std = StandardScaler()
    if strategy == "normalize":
        std = Normalizer()
    if strategy == "minmax":
        std = MinMaxScaler()
    select_ = data.select_dtypes(exclude='object').columns
    data[select_] = std.fit_transform(data.select_dtypes(exclude='object'))
    return data


def train_model(X_train, y_train, X_test, y_test, model):
    if model == "logistic":
        reg = LogisticRegression()

    if model == "SVM":
        reg = SVC()

    if model == "rf":
        reg = RandomForestClassifier()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), reg


if __name__ == "__main__":
    # Get data
    df = pd.read_csv("data/Darknet.csv", error_bad_lines=False)
    df = reduce_mem_usage(df)

    # EDA
    # Correlation 확인
    df.corr()

    # Type of traffic on Darknet 등등 확인
    sns.histplot(data=df, x="Label.1", stat="probability")
    plt.xlabel('')
    plt.show()

    # 1) Column 한개만 가지고 있는 column 삭제
    unique_column = []
    for k in df.columns:
        if len(df[k].unique()) == 1:
            # print("Column name:", k, df[k].unique())
            unique_column.append(k)

    df = df.drop(unique_column, axis=1)

    # 2) Timestamp -> 시간
    df["hour"] = df.apply(change_hour, axis=1).astype("int")

    # 3) Label.1 중복 값 대처
    df["Label.1"].loc[df["Label.1"] == "AUDIO-STREAMING"] = "Audio-Streaming"
    df["Label.1"].loc[df["Label.1"] == "File-transfer"] = "File-Transfer"
    df["Label.1"].loc[df["Label.1"] == "Video-streaming"] = "Video-Streaming"

    # 4) IP 값을 -> IP Class 값으로 (A, B, C, D, E class)
    df["Src IP class"] = df["Src IP"].apply(change_ip_class)
    df["Dst IP class"] = df["Dst IP"].apply(change_ip_class)

    # 5) 사용하지 않은 column 삭제
    df = df.drop(
        ["Flow ID", "Timestamp", "Src IP", "Dst IP", "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Timestamp"],
        axis=1)

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2)

    X_train_fill = fill_missing_value(X_train, "mean")
    X_test_fill = fill_missing_value(X_test, "mean")
    X_train_fill_s = scale_value(X_train, "standard")
    X_test_fill_s = scale_value(X_test, "standard")

    max_f1 = 0
    max_reg = ""
    max_c_m = ""
    # 모델 트레이닝
    for al in ["rf", "logistic", "SVM"]:
        c_m, f1, reg = train_model(X_train_sale, y_train, X_test_scale, y_test, al)
        if f1 > max_f1:
            max_f1 = f1
            max_reg = al
            max_c_m = c_m

    # 평가
    # Classification Report
    print(classification_report(y_test, max_reg.predict(X_test_fill_s)))


