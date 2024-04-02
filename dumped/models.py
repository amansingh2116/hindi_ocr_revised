# Define the models to be used to classify images using python libraries

def randomforest(X_train, Y_train, X_test, Y_test):
    from sklearn.ensemble import RandomForestClassifier
    # Define the models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the models
    rf_model.fit(X_train, Y_train)
    # Predictions
    y_pred = rf_model.predict(X_test)
    # Model Evaluation
    accuracy = rf_model.score(X_test, Y_test)
    return (y_pred, accuracy)


def knn(X_train, Y_train, X_test, Y_test):
    from sklearn.neighbors import KNeighborsClassifier
    # Define the models
    knn_model = KNeighborsClassifier(n_neighbors=5)
    # Fit the models
    knn_model.fit(X_train, Y_train)
    # Predictions
    y_pred = knn_model.predict(X_test)
    # Model Evaluation
    accuracy = knn_model.score(X_test, Y_test)
    return (y_pred, accuracy)


def svm(X_train, Y_train, X_test, Y_test):
    from sklearn.svm import SVC
    # Define the models
    svm_model = SVC(kernel='rbf', C=1.0)
    # Fit the models
    svm_model.fit(X_train, Y_train)
    # Predictions
    y_pred = svm_model.predict(X_test)
    # Model Evaluation
    accuracy = svm_model.score(X_test, Y_test)
    return (y_pred, accuracy)

def Logistic_regression(X_train, Y_train, X_test, Y_test):
    from sklearn.linear_model import LogisticRegression
    # Define the models
    logit_model = LogisticRegression(multi_class='multinomial', max_iter=1000, C=1, solver='sag')
    # Fit the models
    logit_model.fit(X_train, Y_train)
    # Predictions
    y_pred = logit_model.predict(X_test)
    # Model Evaluation
    accuracy = logit_model.score(X_test, Y_test)
    return (y_pred, accuracy)



def cnn(X_train, Y_train, X_test, Y_test):
    import tensorflow as tf
    # Define the CNN architecture
    model = tf.keras.Sequential([ tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Reshape the data if needed (e.g., for image data)
    X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    # Train the model
    model.fit(X_train_reshaped, Y_train, epochs=5, batch_size=64)

    # Evaluate the model
    test_loss, accuracy = model.evaluate(X_test_reshaped, Y_test)
    
    # Make predictions
    y_pred = model.predict_classes(X_test_reshaped)

    return (y_pred, accuracy)
