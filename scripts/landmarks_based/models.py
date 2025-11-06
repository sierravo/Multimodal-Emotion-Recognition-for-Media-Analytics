from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def get_classifiers():
    """
    Returns a dictionary of initialized classifiers.
    """
    classifiers = {
        "SVM_RBF": svm.SVC(kernel='rbf', probability=True),
        "Gradient_Boosting": GradientBoostingClassifier(),
        "Random_Forest": RandomForestClassifier(),
        "Naive_Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=500),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    return classifiers
