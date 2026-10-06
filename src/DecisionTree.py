from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import warnings
import graphviz
warnings.filterwarnings('ignore')

def main():
    # DecisionTree Classifier
    #dt_clf = DecisionTreeClassifier(random_state=156)
    dt_clf = DecisionTreeClassifier(random_state=42) # tree2 #-> 해당 예제에선 차이없는듯...

    # Load data, split into train/test sets
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=11)

    # Training the model
    dt_clf.fit(X_train, y_train)

    # Visualizing the decision tree
    #export_graphviz(dt_clf, out_file="tree.dot", feature_names=iris.feature_names, class_names=iris.target_names, impurity=True, filled=True)
    export_graphviz(dt_clf, out_file="tree2.dot", feature_names=iris.feature_names, class_names=iris.target_names, impurity=True, filled=True)
    # Visualizing the decision tree using graphviz
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    return

if __name__ == "__main__":
    main()