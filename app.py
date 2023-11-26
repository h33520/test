import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, plot_confusion_matrix, confusion_matrix, roc_curve, roc_auc_score, classification_report,accuracy_score,auc
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 导入数据库
df = pd.read_csv('winequalityN.csv')

# Streamlit 应用
st.title("Wine Quality Prediction App")

# 用户输入
fixed_acidity = st.slider("Fixed Acidity", float(df["fixed acidity"].min()), float(df["fixed acidity"].max()))
volatile_acidity = st.slider("Volatile Acidity", float(df["volatile acidity"].min()), float(df["volatile acidity"].max()))
citric_acid = st.slider("Citric Acid", float(df["citric acid"].min()), float(df["citric acid"].max()))
residual_sugar = st.slider("Residual Sugar", float(df["residual sugar"].min()), float(df["residual sugar"].max()))
chlorides = st.slider("Chlorides", float(df["chlorides"].min()), float(df["chlorides"].max()))
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", float(df["free sulfur dioxide"].min()), float(df["free sulfur dioxide"].max()))
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", float(df["total sulfur dioxide"].min()), float(df["total sulfur dioxide"].max()))
density = st.slider("Density", float(df["density"].min()), float(df["density"].max()))
pH = st.slider("pH", float(df["pH"].min()), float(df["pH"].max()))
sulphates = st.slider("Sulphates", float(df["sulphates"].min()), float(df["sulphates"].max()))
alcohol = st.slider("Alcohol", float(df["alcohol"].min()), float(df["alcohol"].max()))


# 添加按钮
if st.button("Predict"):
    # 在按钮被点击时进行预处理
    # 首先删除空缺值
    df = df.dropna()

    # 设定分类区间对quality进行二分类
    bins = (2, 5,  9)

    # 将quality分为bad和good两类
    group_names = ['bad','good']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)

    # 将quality的标签换为数字
    label_quality = LabelEncoder()
    df['quality'] = label_quality.fit_transform(df['quality'])

    # 把数据分为X和Y两类
    Y = df['quality']
    X = df.drop('quality', axis=1)

    # 获取用户输入
    user_input = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
               free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]

    # 将用户输入转换为DataFrame
    user_df = pd.DataFrame(user_input, columns=X.columns)

    # 将用户输入与原始数据集合并
    combined_df = pd.concat([X, user_df], ignore_index=True)

    # 定义标签编码函数
    def Label_enc(feat):
        LabelE = LabelEncoder()
        LabelE.fit(feat)
        return LabelE.transform(feat)

    # 遍历数据集，对数据特征进行编码
    for col in combined_df.columns:
        combined_df[str(col)] = Label_enc(combined_df[str(col)])

    # 通过去除均值并按比例缩放到单位方差来标准化特征
    scaler = StandardScaler()

    # 在合并后的数据上进行归一化处理
    combined_df_normalized = pd.DataFrame(scaler.fit_transform(combined_df), columns=combined_df.columns)

    # 获取用户输入所在行的索引
    user_index = combined_df.index[-1]

    # 执行 train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(combined_df_normalized.iloc[:-1, :], Y, test_size=0.25, random_state=42)

    #通过交叉验证来确定标准 KNN 和权重 KNN 分别对应的最优 K 值
    cv_scores = []
    candidate_k_values = range(1, 21)

    #使用权重 KNN 对不同的 K 值进行精确度的计算
    for n in candidate_k_values:                                           
        knn_weight = KNeighborsClassifier(n_neighbors = n,weights='distance')
        knn_weight.fit(X_train, Y_train)
        Y_pred_weight = knn_weight.predict(X_test)
    
        #设置 10 折交叉验证
        scores = cross_val_score(knn_weight, X, Y, cv=10)                   
        cv_scores.append(np.mean(scores))

    # 选择性能最好的 K 值
    best_weight = candidate_k_values[np.argmax(cv_scores)] 

    # 计算准确率
    model = KNeighborsClassifier(n_neighbors=best_weight, weights='distance')
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    # 输出准确率
    st.write(f"模型准确率：{accuracy}")

    # 预测结果
    prediction_result = model.predict(combined_df.iloc[user_index, :].values.reshape(1, -1))
    
    # 显示预测结果
    st.subheader("Prediction:")
    
    # 根据预测结果输出文本
    if prediction_result[0] == 0:
        prediction_text = "红酒质量较差"
    else:
        prediction_text = "红酒质量较好"

    # 显示预测结果
    st.subheader("Prediction:")
    st.write(f"The predicted wine quality is: {prediction_result[0]}，{prediction_text}")

