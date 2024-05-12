import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Example confusion matrix data
true_labels = [0, 1, 0, 1, 2, 0, 1, 2, 2]
predicted_labels = [0, 1, 0, 1, 2, 1, 1, 2, 0]

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save confusion matrix as image
plt.show()




# Example classification report data
y_true = np.array([0, 1, 2, 2, 0])
y_pred = np.array([0, 1, 1, 2, 1])
target_names = ['class 0', 'class 1', 'class 2']

# Generate classification report
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

# Convert classification report to a table
table_data = []
for key, value in report.items():
    if key in target_names:
        table_data.append([key, value['precision'], value['recall'], value['f1-score'], value['support']])

# Plot classification report as a table
plt.figure(figsize=(8, 6))
plt.table(cellText=table_data, colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'], loc='center')
plt.axis('off')  # Hide axes
plt.title('Classification Report')
plt.tight_layout()
plt.savefig('classification_report.png')  # Save classification report as image
plt.show()
