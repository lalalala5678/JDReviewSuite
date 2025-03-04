import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import logging
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(file_path: str) -> pd.DataFrame:
    """从 Excel 文件中加载数据。"""
    try:
        data = pd.read_excel(file_path)
        logging.info("数据加载成功。")
        return data
    except Exception as e:
        logging.error("加载数据失败: %s", e)
        raise


def preprocess_data(data: pd.DataFrame, target: str, drop_cols: list) -> (pd.DataFrame, pd.Series):
    """
    数据预处理：
      - 删除无关列
      - 将目标变量映射为二分类 (A.愿意 -> 1, 其它 -> 0)
      - 分离特征和目标变量
    """
    data.drop(columns=drop_cols, inplace=True, errors='ignore')
    data[target] = data[target].apply(lambda x: 1 if x == "A.愿意" else 0)
    X = data.drop(columns=[target])
    y = data[target]
    return X, y


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    根据数据类型构造预处理器：
      - 数值型特征：中位数填充 + 标准化
      - 类别型特征：先将每个元素转换为字符串（缺失值替换为 "missing"），再进行常量填充和 One-Hot 编码
    """
    # 分别获取数值型和类别型特征（基于当前 X）
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logging.info("数值型特征：%s", numeric_features)
    logging.info("类别型特征：%s", categorical_features)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 通过 FunctionTransformer 强制将数据转换为字符串（缺失值替换为 "missing"）
    categorical_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda X: X.applymap(lambda x: "missing" if pd.isnull(x) else str(x)))),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """构造包含预处理器和多层感知机分类器的完整流水线。"""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(random_state=42))
    ])
    return pipeline


def tune_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """使用 GridSearchCV 对模型进行交叉验证和超参数调优。"""
    param_grid = {
        'classifier__hidden_layer_sizes': [(50, 25), (100, 50), (50, 50, 25)],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__max_iter': [500]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info("调参完成。最佳参数：%s", grid_search.best_params_)
    logging.info("交叉验证最佳准确率：%.4f", grid_search.best_score_)
    return grid_search


def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """在测试集上评估模型性能。"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info("测试集准确率：%.4f", acc)
    logging.info("分类报告：\n%s", report)
    print("测试集准确率：", acc)
    print("分类报告：\n", report)

def get_feature_names(ct: ColumnTransformer):
    """
    从 ColumnTransformer 中提取特征名称。对于 Pipeline 类型的转换器，
    如果最后一步实现了 get_feature_names_out，则调用该方法；否则直接返回原始列名。
    """
    feature_names = []
    for name, trans, cols in ct.transformers_:
        if trans == 'drop' or trans is None:
            continue
        if isinstance(trans, Pipeline):
            # 对于 Pipeline，我们尝试从最后一步提取特征名称
            last_step = trans.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
            else:
                names = cols
        elif hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        else:
            names = cols
        feature_names.extend(names)
    return np.array(feature_names)


def shap_explain(model: GridSearchCV, X_test: pd.DataFrame, num_samples: int = 10) -> None:
    """
    利用 SHAP 值对模型进行解释：
      - 采用 KernelExplainer 解释 predict_proba
      - 使用部分测试样本进行解释
      - 绘制 SHAP summary plot（包含正负贡献）
      - 另外计算每个特征的平均原始 SHAP 值（可能为负）和平均绝对 SHAP 值，
        并利用 Plotly 绘制条形图显示原始 SHAP 值（负值将反映为负贡献）
    """
    # 从最佳模型中获取已拟合的预处理器和分类器
    fitted_preprocessor = model.best_estimator_.named_steps['preprocessor']
    best_classifier = model.best_estimator_.named_steps['classifier']
    
    # 使用预处理器转换测试数据
    X_test_transformed = fitted_preprocessor.transform(X_test)
    background = X_test_transformed[:100] if X_test_transformed.shape[0] >= 100 else X_test_transformed

    explainer = shap.KernelExplainer(best_classifier.predict_proba, background)
    shap_values = explainer.shap_values(X_test_transformed[:num_samples])
    
    # 针对二分类问题，选择其中一个类的解释
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # 计算每个特征的平均原始 SHAP 值（可能为负）和平均绝对 SHAP 值
    mean_raw_shap = np.mean(shap_vals, axis=0).flatten()
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0).flatten()
    
    # 获取转换后特征的名称，使用自定义函数处理不支持 get_feature_names_out 的情况
    feature_names = get_feature_names(fitted_preprocessor)
    
    # 使用 SHAP 自带的交互式 summary plot（会展示正负贡献情况）
    shap.initjs()
    shap.summary_plot(shap_values, X_test_transformed[:num_samples], feature_names=feature_names)
    
    # 构造 DataFrame 用于 Plotly 绘图，同时显示原始 SHAP 值（包含负值）
    df_shap = pd.DataFrame({
        "Feature": feature_names,
        "Mean Raw SHAP": mean_raw_shap,
        "Mean Abs SHAP": mean_abs_shap
    })
    # 按平均绝对值排序
    df_shap.sort_values(by="Mean Abs SHAP", ascending=False, inplace=True)
    
    import plotly.express as px
    fig = px.bar(df_shap, x="Mean Raw SHAP", y="Feature", orientation='h',
                 title="Mean Raw SHAP Values (含正负贡献)",
                 labels={"Mean Raw SHAP": "平均原始 SHAP 值", "Feature": "特征"})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.write_html("shap_raw_summary_plot.html")
    logging.info("已将原始 SHAP 值条形图导出为 shap_raw_summary_plot.html")





def main():
    # 参数配置
    file_path = './《正大杯》药食同源滋补汤正式数据955.xlsx'
    target = "Q25_25.如果正大食品推出药食同源滋补汤，您有意愿购买吗？"
    drop_cols = ['答题序号', '来源', '开始时间', '提交时间', '答题时长',
                 'IP省份', 'IP城市', 'IP地址', '浏览器', '操作系统']

    # 1. 数据加载
    data = load_data(file_path)
    # 删除全缺失的列，避免后续 imputer 和编码器报错
    cols_to_drop_if_all_missing = data.columns[data.isnull().all()].tolist()
    data.drop(columns=cols_to_drop_if_all_missing, inplace=True, errors='ignore')
    logging.info("删除全缺失的列：%s", cols_to_drop_if_all_missing)
    logging.info("数据基本信息：\n%s", data.info())
    logging.info("数据预览：\n%s", data.head())

    # 2. 数据预处理：构造目标变量和特征
    X, y = preprocess_data(data, target, drop_cols)

    # 3. 构造预处理器（基于 X 进行特征选择）
    preprocessor = create_preprocessor(X)

    # 4. 构建完整流水线
    pipeline = build_pipeline(preprocessor)

    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. 模型调参
    grid_search = tune_model(pipeline, X_train, y_train)
    print("最佳参数：", grid_search.best_params_)
    print("交叉验证最佳准确率：", grid_search.best_score_)

    # 7. 在测试集上评估模型
    evaluate_model(grid_search, X_test, y_test)

    # 8. 模型解释：利用 SHAP 进行特征重要性分析
    shap_explain(grid_search, X_test)   


if __name__ == '__main__':
    main()
