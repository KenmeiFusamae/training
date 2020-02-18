# common
# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
# SVM
from sklearn.svm import LinearSVC
# 決定木
from sklearn.tree import  DecisionTreeClassifier
# k-NN
from sklearn.neighbors import  KNeighborsClassifier

# データ分割
from sklearn.model_selection import train_test_split

#分析対象データ
from sklearn.datasets import load_iris

# data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, stratify = iris.target, random_state=0)

# initial value
best_score = 0
best_method = ""
# working place. everything
def homework(X_train, X_test, y_train, y_test,best_score,best_method):
        # ロジスティック回帰

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
    print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))

    #決定木
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    model.fit(X_train, y_train)

    print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
    print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))
    best_score = model.score(X_test, y_test)
    best_method = model.__class__.__name__
    print(best_method)

    #k-NN
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(X_train, y_train)
    print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
    print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))
    if (best_score < model.score(X_test, y_test)):
        best_score = model.score(X_test, y_test)
        best_method = model.__class__.__name__


    model = LinearSVC()
    model.fit(X_train, y_train)

    # 訓練データとテストデータのスコア
    print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
    print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))
    if (best_score < model.score(X_test, y_test)):
        best_score = model.score(X_test, y_test)
        best_method = model.__class__.__name__

    my_result = best_method
    return my_result
