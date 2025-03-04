import pandas as pd
import numpy as np
import matplotlib
import shap
import logging
import os
import plotly.express as px  # 用于生成交互式图表

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

# 1. 将日志写入文件 model_training.log（如需同时在控制台打印，可添加 handlers）
logging.basicConfig(
    filename='model_training.log',
    filemode='w',  # 'w' 每次运行时覆盖日志；如需追加可改为 'a'
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logging.info("数值型特征：%s", numeric_features)
    logging.info("类别型特征：%s", categorical_features)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 修改：设置 OneHotEncoder 返回 dense 数组（新版 scikit-learn 使用 sparse_output=False）
    categorical_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda X: X.applymap(lambda x: "missing" if pd.isnull(x) else str(x)))),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
    """
    在测试集上评估模型性能，并将分类报告保存为 HTML。
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    logging.info("测试集准确率：%.4f", acc)
    logging.info("分类报告：\n%s", report)
    print("测试集准确率：", acc)
    print("分类报告：\n", report)

    # 将分类报告转为 DataFrame 后导出 HTML
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_html("classification_report.html", encoding='utf-8')
    logging.info("已将分类报告导出为 classification_report.html")

def get_feature_names(ct: ColumnTransformer):
    """
    从 ColumnTransformer 中提取特征名称。
    对于 Pipeline 类型的转换器，尝试从最后一步提取特征名称，
    否则直接返回原始列名。
    """
    feature_names = []
    for name, trans, cols in ct.transformers_:
        if trans == 'drop' or trans is None:
            continue
        if isinstance(trans, Pipeline):
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
      - 计算每个特征的平均原始 SHAP 值（可能为负）和平均绝对 SHAP 值
      - 利用 Plotly 绘制条形图（显示原始 SHAP 值，包括负贡献），结果以 HTML 文件保存
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
    
    # 尝试获取转换后特征名称
    feature_names = get_feature_names(fitted_preprocessor)
    # 检查名称数量是否与转换后特征数一致
    if len(feature_names) != X_test_transformed.shape[1]:
        logging.warning("特征名称数 (%d) 与转换后特征数 (%d) 不匹配，采用通用名称。",
                        len(feature_names), X_test_transformed.shape[1])
        feature_names = np.array([f"feature_{i}" for i in range(X_test_transformed.shape[1])])
    
    # 如果 SHAP 值数组长度是特征名称数组长度的两倍，则只保留前半部分
    if len(mean_raw_shap) == 2 * len(feature_names):
        logging.info("检测到 SHAP 值长度为特征名称长度的2倍，取前半部分。")
        mean_raw_shap = mean_raw_shap[:len(feature_names)]
        mean_abs_shap = mean_abs_shap[:len(feature_names)]
    
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

    # 将结果输出为 Excel 文件（不包含索引）
    df_shap.to_excel("shap_raw_summary_plot.xlsx", index=False)
    logging.info("已将原始 SHAP 值数据导出为 shap_raw_summary_plot.xlsx")

    # 将结果输出为 JSON 文件（以 records 形式保存，并保留中文字符）
    df_shap.to_json("shap_raw_summary_plot.json", orient="records", force_ascii=False)
    logging.info("已将原始 SHAP 值数据导出为 shap_raw_summary_plot.json")





def export_neuron_parameters(classifier, output_excel='neuron_parameters.xlsx'):
    """
    将 MLPClassifier 中每层神经元的权重和偏置导出到 Excel 文件，
    每层的权重和偏置分别存放在不同的工作表中。
    """
    with pd.ExcelWriter(output_excel) as writer:
        for i, (weights, biases) in enumerate(zip(classifier.coefs_, classifier.intercepts_)):
            # 权重：行为上一层神经元，列为当前层神经元
            df_weights = pd.DataFrame(weights)
            # 偏置：每个神经元对应一个偏置值
            df_biases = pd.DataFrame(biases, columns=['Bias'])
            sheet_name_weights = f'Layer{i+1}_Weights'
            sheet_name_biases = f'Layer{i+1}_Biases'
            df_weights.to_excel(writer, sheet_name=sheet_name_weights, index=True)
            df_biases.to_excel(writer, sheet_name=sheet_name_biases, index=True)
    logging.info("已将神经元参数导出至 %s", output_excel)


def main():
    # 设置中文字体（示例为 SimHei），使得 SHAP 图中能显示中文
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

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

    # 7. 在测试集上评估模型（同时输出分类报告为 HTML）
    evaluate_model(grid_search, X_test, y_test)

    # 新增：导出最佳模型中 MLPClassifier 每层神经元的参数到 Excel 文件
    best_classifier = grid_search.best_estimator_.named_steps['classifier']
    export_neuron_parameters(best_classifier, output_excel='neuron_parameters.xlsx')

    # 8. 模型解释：利用 SHAP 进行特征重要性分析，输出交互式 HTML 图表
    shap_explain(grid_search, X_test)


if __name__ == '__main__':
    main()
