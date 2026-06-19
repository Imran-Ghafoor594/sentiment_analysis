import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_train, X_test, y_train, y_test, run_name, results_dir):
    """
    Performs empirical evaluation, latency measurement, and plots the confusion matrix heatmap.
    """
    # Training time latency track karna
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Inference time latency track karna
    t1 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t1

    # Metrics calculate karna
    acc = accuracy_score(y_test, y_pred)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    _, _, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_w,
        "train_time": train_time,
        "inference_time": inference_time
    }

    # Confusion Matrix as a Heatmap plotting
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'])
    plt.title(f"Confusion Matrix: {run_name}")
    plt.ylabel('Actual Classes')
    plt.xlabel('Predicted Classes')

    # Chart image ko save karna
    fig_path = f"{results_dir}/{run_name.replace(' ', '_').lower()}_cm.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return metrics, fig_path
