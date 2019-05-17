from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

#training data
train_data = iris.data
train_target = iris.target

#testing data
test_data = [[1, 2, 3, 4]]

#classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#prediction
print(clf.predict(test_data))