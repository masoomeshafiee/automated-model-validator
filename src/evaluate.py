from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    #improve it later
    y_pred = model.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
