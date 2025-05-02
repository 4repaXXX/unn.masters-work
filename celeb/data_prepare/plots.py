import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    plt.savefig('training_metrics.png')  # Сохраняем графики
    plt.show()
    
def confusion_matrix(trainedy_pred_model, precision, recall):
    y_pred = trained_model.predict(test_gen)
    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = test_gen.labels
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix\nTest Precision: {precision:.2f}, Recall: {recall:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()