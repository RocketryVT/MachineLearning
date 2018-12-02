import csv
import numpy as np
import pandas
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import matplotlib.font_manager

np.random.seed()

path = "motor_data.csv"
data = np.array(pandas.read_csv(path))

time = data[:,0]
thrust = data[:,1]
noisy_thrust = data[:,2]
dist_thrust = data[:,3]
pres = data[:,4]
noisy_pres = data[:,5]
dist_pres = data[:,6]
temp = data[:,7]
noisy_temp = data[:,8]
dist_temp = data[:,9]

X_train = np.column_stack((noisy_thrust - thrust, noisy_pres - pres))
X_test = np.column_stack((np.random.normal(scale=1.5, size=(50,)),
    np.random.normal(scale=1, size=(50,))))
X_outliers = np.column_stack((dist_thrust - thrust, dist_pres - pres))
X_outliers = np.vstack([X_outliers,
    np.random.uniform(low=-30, high=30, size=(200, 2))])

# fit the model
svm = svm.OneClassSVM(kernel="rbf", gamma=0.1, nu=0.1)
svm.fit(X_train)
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)
y_pred_outliers = svm.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-30, 30, 500), np.linspace(-10, 10, 500))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.rc('axes', axisbelow=True)
plt.figure(1)
plt.title("Measured Deviation from Nominal")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 6
markers = np.array(['.', 'D'])
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], s=s, color='green')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], s=s, color='blue')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], s=s, color='red')
plt.axis('tight')
plt.xlim((-30, 30))
plt.ylim((-10, 10))
plt.grid()
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "regular observations", "abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel("Thrust (lbf)")
plt.ylabel("Pressure (bar)")

plt.figure(2)
plt.subplot(121)
plt.title("Thrust")
plt.xlabel("Time (sec)")
plt.ylabel("Thrust (lbf)")
plt.grid()
plt.xlim((0, 5))
plt.ylim((360, 500))
th = plt.scatter(time, thrust, s=s, color='green')
thn = plt.scatter(time, noisy_thrust, s=s, color='blue')
thd = plt.scatter(time, dist_thrust, s=s, color='red')
plt.legend([th, thn, thd], ["Nominal", "Measured", "Anomalous"],
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.subplot(122)
plt.title("Pressure")
plt.xlabel("Time (sec)")
plt.ylabel("Pressure (bar)")
plt.grid()
plt.xlim((0, 5))
pr = plt.scatter(time, pres, s=s, color='green')
prn = plt.scatter(time, noisy_pres, s=s, color='blue')
prd = plt.scatter(time, dist_pres, s=s, color='red')
plt.legend([pr, prn, prd], ["Nominal", "Measured", "Anomalous"],
           prop=matplotlib.font_manager.FontProperties(size=11))

X_inliers = np.column_stack((time, noisy_thrust))
X_outliers = np.column_stack((time, noisy_thrust +
    np.random.normal(scale=15, size=noisy_thrust.shape)))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

lof = LocalOutlierFactor(n_neighbors=4, contamination=0.5)
y_pred = lof.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = lof.negative_outlier_factor_

plt.figure(3)
plt.title("Uncertainty Assessment with LOF")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Measurement')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=radius*700, edgecolors='r',
            facecolors='none', label='Outlier Scores')
plt.axis('tight')
plt.grid();
plt.xlim((0, 5))
plt.ylim((360, 500))
plt.xlabel("Time (sec)")
plt.ylabel("Thrust (lbf)")
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
