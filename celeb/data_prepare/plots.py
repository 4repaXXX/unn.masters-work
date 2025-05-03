import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc

def plot_history(history_df):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['auc'], label='Train AUC')
    plt.plot(history_df['val_auc'], label='Val AUC')
    plt.title('AUC over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Precision-Recall
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['precision'], label='Train Precision')
    plt.plot(history_df['val_precision'], label='Validation Precision')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['recall'], label='Train Recall')
    plt.plot(history_df['val_recall'], label='Validation Recall')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.savefig('results/training_metrics.png')  # Сохраняем графики
    plt.show()
    
def confusion_matrix_plot(y_pred, y_true, precision=0.5, recall=0.5):
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix\nTest Precision: {precision:.2f}, Recall: {recall:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def precision_recall_plot(y_true, y_pred, recall=0.5, precision=0.5):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
    plt.fill_between(recall, precision, alpha=0.2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/precision_recall_curve.png', dpi=300)
    plt.show()

def roc_plot(y_true, y_pred):

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/roc_curve.png', dpi=300)
    plt.show()