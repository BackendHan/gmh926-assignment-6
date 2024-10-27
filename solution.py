import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack, csr_matrix
import gc
import warnings
warnings.filterwarnings('ignore')

# 读取 test.csv
print('Reading test data...')
test_df = pd.read_csv('test.csv')

# 准备测试数据
print('Preparing test data...')
test_ids = set(test_df['Id'].values)

# 从 train.csv 中提取测试数据
print('Extracting test data from train.csv...')
cols = ['Id', 'Summary', 'Text']
test_data_list = []

chunk_size = 50000  # 减小块大小以节省内存
reader = pd.read_csv('train.csv', usecols=cols, chunksize=chunk_size)

for chunk in reader:
    chunk_test_data = chunk[chunk['Id'].isin(test_ids)]
    if not chunk_test_data.empty:
        test_data_list.append(chunk_test_data)

test_data = pd.concat(test_data_list, ignore_index=True)

# 确保 test_data 的行顺序与 test_df 一致
test_data = test_data.set_index('Id').loc[test_df['Id']].reset_index()

# 合并 'Summary' 和 'Text' 为 'Combined_Text'
print('Combining test text data...')
test_data['Combined_Text'] = test_data['Summary'].fillna('') + ' ' + test_data['Text'].fillna('')
test_data['Combined_Text'] = test_data['Combined_Text'].fillna('')

# 初始化 TfidfVectorizer
print('Initializing TfidfVectorizer...')
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))

# 对测试数据进行矢量化
print('Vectorizing test text data...')
X_test = vectorizer.fit_transform(test_data['Combined_Text'])

# 初始化分类器
print('Initializing classifier...')
clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, n_jobs=-1)

classes = np.array([1, 2, 3, 4, 5])

# 读取 train.csv 并逐块训练模型
print('Training model incrementally...')
chunk_size = 50000  # 减小块大小
sampling_rate = 0.3  # 使用30%的数据
reader = pd.read_csv('train.csv', usecols=['Score', 'Summary', 'Text'], chunksize=chunk_size)

for chunk in reader:
    # 仅保留 'Score' 不为空的行
    chunk = chunk[chunk['Score'].notnull()]
    if chunk.empty:
        continue

    # 对块进行采样
    chunk = chunk.sample(frac=sampling_rate, random_state=42)

    # 准备数据
    chunk['Combined_Text'] = chunk['Summary'].fillna('') + ' ' + chunk['Text'].fillna('')
    chunk['Combined_Text'] = chunk['Combined_Text'].fillna('')

    # 矢量化文本数据
    X_train_chunk = vectorizer.transform(chunk['Combined_Text'])

    y_train_chunk = chunk['Score'].astype(int).values

    # 部分拟合模型
    clf.partial_fit(X_train_chunk, y_train_chunk, classes=classes)

    # 清理内存
    del chunk, X_train_chunk, y_train_chunk
    gc.collect()

# 分块预测
print('Predicting on test data...')
y_pred = []

test_chunk_size = 50000
for i in range(0, X_test.shape[0], test_chunk_size):
    X_test_chunk = X_test[i:i+test_chunk_size]
    y_pred_chunk = clf.predict(X_test_chunk)
    y_pred.extend(y_pred_chunk)

# 准备提交文件
print('Preparing submission...')
submission = pd.DataFrame({'Id': test_data['Id'], 'Score': y_pred})
submission.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')
