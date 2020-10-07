from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

# generate 2 class dataset by ouselves
X, y = make_classification(n_samples=2000, n_classes=2, random_state=1)
# split datasets into train/test sets (80%~20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# generate a no model prediction
nm_probs = [0 for _ in range(len(y_test))]
# fit a model
model = RandomForestRegressor()
model.fit(X_train, y_train)
# predict probabilities
rf_probs = model.predict(X_test)
# calculate scores
nm_auc = roc_auc_score(y_test, nm_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
# summarize scores
print('No model: ROC AUC=%.2f' % (nm_auc))
print('Random forest: ROC AUC=%.2f' % (rf_auc))
# calculate roc curves
nm_fpr, nm_tpr, _ = roc_curve(y_test, nm_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
# plot the roc curve for the model
plt.plot(nm_fpr, nm_tpr, linestyle='--', label='No model AUC = %0.2f'% nm_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random forest AUC = %0.2f'% rf_auc)
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


