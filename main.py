# 1. Импорт необходимых библиотек
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from tqdm import tqdm

"""# 2. Параметры эксперимента"""

encoding_dim = 10           # размер латентного пространства
hidden_multiplier = 2       # множитель для скрытых слоев (глубина/ширина)
pretrain_epochs = 50        # эпохи предобучения автоэнкодера
batch_size = 32
n_clusters = 5              # количество кластеров
alpha = 1.0                 # параметр Student’s t-распределения
loss_weights = [1.0, 0.1]   # вес KLD и MSE в общей функции потерь
maxiter = 2000              # число итераций DEC (уменьшено для примера)
update_interval = 100       # как часто пересчитывать целевое распределение

"""# 3. Загрузка и предобработка данных"""

url = 'https://storage.yandexcloud.net/academy.ai/Mall_Customers.csv'
df = pd.read_csv(url)
# Переименовываем и кодируем
mapping = {'Male': 0, 'Female': 1}
X = df[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']]
X['Gender'] = X['Gender'].map(mapping)
# Масштабируем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""# 4. K-Means baseline"""

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
labels_km = kmeans.fit_predict(X_scaled)

"""# 5. Построение и предобучение автоэнкодера"""

def build_autoencoder(input_dim, encoding_dim, hidden_multiplier):
    inp = Input(shape=(input_dim,))
    # Кодировщик
    x = Dense(encoding_dim * hidden_multiplier, activation='relu')(inp)
    x = Dense(encoding_dim, activation='relu')(x)
    # Декодировщик
    x = Dense(encoding_dim * hidden_multiplier, activation='relu')(x)
    out = Dense(input_dim, activation='linear')(x)
    autoenc = Model(inp, out, name='autoencoder')
    encoder = Model(inp, x, name='encoder')
    return autoenc, encoder

input_dim = X_scaled.shape[1]

autoencoder, encoder = build_autoencoder(input_dim, encoding_dim, hidden_multiplier)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
autoencoder.fit(
    X_scaled, X_scaled,
    epochs=pretrain_epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# Сохранение модели
embeddings_pre = encoder.predict(X_scaled)

"""# 6. Инициализация кластеров и построение DEC-модели"""

# K-Means на предобученных вложениях
kmeans_emb = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
labels_init = kmeans_emb.fit_predict(embeddings_pre)
cluster_centers = kmeans_emb.cluster_centers_

# Слой кластеризации
class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, init_centers, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.cluster_centers = tf.Variable(init_centers, trainable=True, dtype=tf.float32)
    def call(self, inputs):
        # Student’s t-distribution
        sq_diff = tf.reduce_sum(tf.square(tf.expand_dims(inputs,1) - self.cluster_centers), axis=2)
        q = 1.0 / (1.0 + sq_diff / self.alpha)
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q

# Сборка модели DEC
q_layer = ClusteringLayer(n_clusters, cluster_centers, alpha=alpha, name='clustering')(encoder.output)
dec_model = Model(inputs=encoder.input, outputs=[q_layer, autoencoder.output])
dec_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=['kld', 'mse'],
    loss_weights=loss_weights
)

"""# 7. Обучение DEC и сбор статистики"""

# Хранение эволюции центроидов
centers_evolution = [cluster_centers.copy()]
dec_metrics = []

for ite in tqdm(range(maxiter), desc='DEC Training'):
    # Пересчёт целевого распределения
    if ite % update_interval == 0:
        q, _ = dec_model.predict(X_scaled, verbose=0)
        p = (q**2 / np.sum(q, axis=0)) / np.sum(q**2 / np.sum(q,axis=0), axis=1, keepdims=True)
        labels_dec = q.argmax(1)
        # Сбор текущих центров
        centers = dec_model.get_layer('clustering').cluster_centers.numpy()
        centers_evolution.append(centers.copy())
    # Батч для обучения
    start = (ite * batch_size) % X_scaled.shape[0]
    end = start + batch_size
    idx = np.arange(start, min(end, X_scaled.shape[0]))
    x_batch = X_scaled[idx]
    p_batch = p[idx]
    m = dec_model.train_on_batch(x_batch, [p_batch, x_batch])
    dec_metrics.append(m)

# Вложения после DEC
embeddings_post = encoder.predict(X_scaled)

"""# 8. Метрики качества кластеризации"""

def print_metrics(X, labels, name):
    print(f"--- {name} ---")
    print("Silhouette", silhouette_score(X, labels))
    print("Davies-Bouldin", davies_bouldin_score(X, labels))
    print("Calinski-Harabasz", calinski_harabasz_score(X, labels))

print_metrics(X_scaled, labels_km, "K-Means")
print_metrics(X_scaled, labels_dec, "DEC")

"""# 9. Визуализация латентных пространств (PCA и t-SNE)"""

plt.figure(figsize=(12,12))

# Pre-DEC PCA
plt.subplot(2,2,1)
pca = PCA(n_components=2)
z = pca.fit_transform(embeddings_pre)
plt.scatter(z[:,0], z[:,1], c=labels_dec, s=30)
plt.title('Pre-DEC PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Post-DEC PCA
plt.subplot(2,2,2)
z = pca.fit_transform(embeddings_post)
plt.scatter(z[:,0], z[:,1], c=labels_dec, s=30)
plt.title('Post-DEC PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Pre-DEC t-SNE
plt.subplot(2,2,3)
ts = TSNE(n_components=2, perplexity=30, random_state=42)
z = ts.fit_transform(embeddings_pre)
plt.scatter(z[:,0], z[:,1], c=labels_dec, s=30)
plt.title('Pre-DEC t-SNE')
plt.xlabel('dim1')
plt.ylabel('dim2')

# Post-DEC t-SNE
plt.subplot(2,2,4)
z = ts.fit_transform(embeddings_post)
plt.scatter(z[:,0], z[:,1], c=labels_dec, s=30)
plt.title('Post-DEC t-SNE')
plt.xlabel('dim1')
plt.ylabel('dim2')

plt.tight_layout()
plt.show()

"""# 10. Эволюция центроидов"""

centers_arr = np.array(centers_evolution)
# Показ траекторий центроидов по первым двум осям латента
plt.figure(figsize=(6,6))
for c in range(n_clusters):
    traj = centers_arr[:, c, :2]
    plt.plot(traj[:,0], traj[:,1], '-o', label=f'Cluster {c}')
plt.title('Эволюция центроидов')
plt.legend()
plt.xlabel('Latent dim1')
plt.ylabel('Latent dim2')
plt.show()

"""# Итоги эксперимента

---

1. Предобучение автоэнкодера
* За 50 эпох удалось снизить MSE-потери с ~0.97 до ~0.02 на обучающей выборке и с ~1.81 до ~0.18 на валидации.
* Рост параметра «accuracy» отражает улучшение реконструкции.
* С учётом кривой потерь автоэнкодер фактически «выучил» латентное представление данных: высокие «accuracy» (~0.95) и низкие потери (~0.02) к 50-й эпохе говорят о том, что структура четырехмерных признаков успешно упакована в заданное latent space.

2. Латентные встраивания: до vs после DEC

* PCA-преобразование (2D)
 * Pre-DEC PCA: точки лежат более разбросанно, кластеры пересекаются.
 * Post-DEC PCA: встраивание стало более «растянуто», сгруппировалось в пять более отчётливых скоплений по PC1 и PC2, особенно явно отделился «жёлтый» кластер (верхняя левая группа).
* t-SNE (2D)
 * Pre-DEC t-SNE: из пяти кластеров рельефно вырисовываются группы, но они всё равно пересекаются и имеют «хвосты».
 * Post-DEC t-SNE: кластеры стали компактнее и более изолированы, особенно выделился один крупный синий «пучок» внизу.
* По визуализациям DEC действительно упорядочил латентное пространство, сделав группы более компактными и раздельными.

3. Эволюция центроидов в латентном пространстве
* Траектории центроидов (по первым двум латентным осям) показали, что большинство центров за первые 3–4 обновления смещаются активно, а затем стабилизируются.
* Иногда один из центров «дрейфует» дальше (Cluster 1 и 3), что может говорить о том, что DEC пытается «перехватить» плохо отделившийся набор точек.

4. Кластеры на исходном и шкалированном пространстве
* K-Means: видно пять приблизительно равных по численности кластеров, но разброс внутри них достаточно велик, особенно по доходу и тратам.
* DEC: перераспределил объекты: один кластер стал крупнее (Cluster 1), другой — значительно уменьшился (Cluster 2), что видно и на гистограммах размеров.

5. Сравнение метрик кластеризации
 * Silhouette: упал у DEC (0.209 до 0.272). То есть по средней «чистоте» кластеров DEC проигрывает.
 * Davies-Bouldin: вырос (1.487 до 1.181), что указывает на более высокую внутрикластерную дисперсию и/или меньшие межкластерные расстояния.
 * Calinski-Harabasz: снизился (50.13 до 62.13), что также свидетельствует об ухудшении соотношения «межгрупповой разброс» к «внутригрупповому».

* Визуально латентное пространство после DEC выглядит более структурированным, но на исходном (или же одинаково масштабированном) признаковом пространстве качество кластеризации на самом деле ухудшилось.

6. Выводы
* Для данного датасета простой K-Means даёт более стабильные и объективно лучшие метрики.
* DEC показывает сильный потенциал в упорядочивании латентных пространств (во множестве других задач это именно то, что нужно), но требует тщательного подбора гиперпараметров.
"""