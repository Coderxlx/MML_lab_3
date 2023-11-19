import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

# Завантаження даних
file_path = 'IrisData_full.csv'
iris_data = pd.read_csv(file_path, header=None)
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data.columns = columns

# Перемішування записів
iris_data_shuffled = iris_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Нормалізація параметрів
scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_data_shuffled.iloc[:, :-1])

# Розділення на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, iris_data_shuffled['species'], test_size=0.3, random_state=42)

# Навчання KNN-класифікатора з різними значеннями K і вибір найкращого K
k_values = range(1, 26)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Визначення найкращого K
best_k = k_values[np.argmax(accuracy_scores)]
best_accuracy = max(accuracy_scores)
print (f'The best k = {best_k} , score = {best_accuracyD}')

# Генерація та візуалізація власних даних
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
np.random.seed(2021)
X_D2, y_D2 = make_blobs(n_samples=300, n_features=2, centers=8, cluster_std=1.3, random_state=4)
y_D2 = y_D2 % 2

plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:, 0], X_D2[:, 1], c=y_D2, marker='o', s=30, cmap=cmap_bold)
plt.show()

# Подготовка к визуализации границы решения
def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    """
    # Fit model
    model = model_class(**model_params)
    model.fit(X, y)

    # Set up the mesh of points
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict classifications for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Define color map
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision boundaries with KNN (k={})".format(model_params['n_neighbors']))

    plt.show()

# Visualizing decision boundaries for KNN
plot_decision_boundaries(X=X_D2, y=y_D2, model_class=KNeighborsClassifier, n_neighbors=best_k)
