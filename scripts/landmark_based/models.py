from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


def get_classifiers():
    """Return initialized landmark-feature classifiers."""
    return {
        "SVM_RBF": svm.SVC(kernel="rbf", probability=True, random_state=0),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=0, n_estimators=50),
        "Random_Forest": RandomForestClassifier(random_state=0, n_estimators=100),
        "Naive_Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=300, random_state=0),
        "XGBoost": xgb.XGBClassifier(
            eval_metric="mlogloss",
            random_state=0,
            n_estimators=50,
            max_depth=3,
        ),
    }
