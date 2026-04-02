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
        "SVM_RBF": svm.SVC(kernel='rbf', probability=True, random_state=0),

        "Gradient_Boosting": GradientBoostingClassifier(
            random_state=0
        ),

        "Random_Forest": RandomForestClassifier(
            random_state=0
        ),

        "Naive_Bayes": GaussianNB(),  # no random_state

        "MLP": MLPClassifier(
            max_iter=500,
            random_state=0
        ),

        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=0
        )
    }

    return classifiers