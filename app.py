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

#导入数据库
df = pd.read_csv('/homework/homework2/winequalityN.csv')

#检查数据库中是否有空值
df.isnull().sum()

# 删除包含缺失值的行
df = df.dropna()  

#设定分类区间对quality进行二分类
bins = (2, 5,  9)

#将quality分为bad和good两类
group_names = ['bad','good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

#将quality的标签换为数字
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])

#统计quality编码后的数字
df['quality'].value_counts()

#定义标签编码函数
def Label_enc(feat):
    LabelE = LabelEncoder()
    LabelE.fit(feat)
    print(feat.name,LabelE.classes_)
    return LabelE.transform(feat)

#遍历数据集，对数据特征进行编码
for col in df.columns:
    df[str(col)] = Label_enc(df[str(col)])

#把数据分为X和Y两类
Y = df['quality']
X = df.drop('quality', axis=1)

#通过去除均值并按比例缩放到单位方差来标准化特征
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#将数据集分为训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 42)

#通过交叉验证来确定标准 KNN 和权重 KNN 分别对应的最优 K 值
cv_scores = []
candidate_k_values = range(1, 21)

#使用权重 KNN 对不同的 K 值进行精确度的计算
for n in candidate_k_values:                                           
    knn_weight = KNeighborsClassifier(n_neighbors = n,weights='distance')
    knn_weight.fit(X_train, Y_train)
    Y_pred_weight = knn_weight.predict(X_test)
    print('KNeighborsClassifier: n = {} , Accuracy is: {}'.format(n,knn_weight.score(X_test,Y_test)))
    
    #设置 10 折交叉验证
    scores = cross_val_score(knn_weight, X, Y, cv=10)                   
    cv_scores.append(np.mean(scores))

# 选择性能最好的 K 值
best_weight = candidate_k_values[np.argmax(cv_scores)] 

model = KNeighborsClassifier(n_neighbors=best_weight, weights='distance')
model.fit(X_train, Y_train)


# Streamlit 应用
st.title("Wine Quality Prediction App")

# 用户输入
fixed_acidity = st.slider("Fixed Acidity", float(X["fixed acidity"].min()), float(X["fixed acidity"].max()))
volatile_acidity = st.slider("Volatile Acidity", float(X["volatile acidity"].min()), float(X["volatile acidity"].max()))
citric_acid = st.slider("Citric Acid", float(X["citric acid"].min()), float(X["citric acid"].max()))
residual_sugar = st.slider("Residual Sugar", float(X["residual sugar"].min()), float(X["residual sugar"].max()))
chlorides = st.slider("Chlorides", float(X["chlorides"].min()), float(X["chlorides"].max()))
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", float(X["free sulfur dioxide"].min()), float(X["free sulfur dioxide"].max()))
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", float(X["total sulfur dioxide"].min()), float(X["total sulfur dioxide"].max()))
density = st.slider("Density", float(X["density"].min()), float(X["density"].max()))
pH = st.slider("pH", float(X["pH"].min()), float(X["pH"].max()))
sulphates = st.slider("Sulphates", float(X["sulphates"].min()), float(X["sulphates"].max()))
alcohol = st.slider("Alcohol", float(X["alcohol"].min()), float(X["alcohol"].max()))

# 添加按钮
if st.button("Predict"):
    # 预测结果
    prediction_input = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
    prediction_result = model.predict(prediction_input)

    # 显示预测结果
    st.subheader("Prediction:")
    st.write(f"The predicted wine quality is: {prediction_result[0]}")

    # 如果需要，你还可以在这里显示其他信息，例如模型的准确性等
