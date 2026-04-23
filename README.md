


# EXP 1: Anaconda / Google Colab demo
import numpy as np
print("ML notebook ready")
a = np.array([1, 2, 3])
print(a * 2)

# EXP 2: ML libraries demo
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print(np.array([1,2,3]) + 1)
print(pd.DataFrame({"A":[1,2], "B":[3,4]}))
X, y = load_iris(return_X_y=True)
print(X.shape, y.shape)



# EXP 3: Linear Regression
from sklearn.linear_model import LinearRegression
import numpy as np, matplotlib.pyplot as plt
X = np.array([[1],[2],[3],[4],[5]]); y = np.array([2,4,6,8,10])
m = LinearRegression().fit(X, y)
print(m.coef_[0], m.intercept_, m.predict([[6]]))
plt.scatter(X, y); plt.plot(X, m.predict(X)); plt.show()



# EXP 4: Logistic Regression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
m = LogisticRegression(max_iter=200).fit(Xtr, ytr)
p = m.predict(Xte)
print(accuracy_score(yte, p))
print(confusion_matrix(yte, p))



# EXP 5: SVM
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
m = SVC(kernel='linear').fit(Xtr, ytr)
print(accuracy_score(yte, m.predict(Xte)))



# EXP 6: Hebbian Learning
import numpy as np
X = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
y = np.array([1,0,0,0])
w = np.zeros(2)
for x, t in zip(X, y): w += x * t
print(w)



# EXP 7: EM Algorithm
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=2, random_state=0)
m = GaussianMixture(n_components=2, random_state=0).fit(X)
print(m.predict(X)[:10])
print(m.lower_bound_)



# EXP 8: McCulloch-Pitts Neuron
def mp(x, w, t): return int(sum(i*j for i,j in zip(x,w)) >= t)
print("AND:", mp([1,1],[1,1],2))
print("OR :", mp([1,0],[1,1],1))
print("NOT:", int(1 < 1))



# EXP 9: Single Layer Perceptron
from sklearn.linear_model import Perceptron
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
m = Perceptron().fit(X, y)
print(m.predict(X))



# EXP 11: PCA
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X, y = load_iris(return_X_y=True)
Z = PCA(n_components=2).fit_transform(X)
print(PCA(n_components=2).fit(X).explained_variance_ratio_)
plt.scatter(Z[:,0], Z[:,1], c=y); plt.show()


