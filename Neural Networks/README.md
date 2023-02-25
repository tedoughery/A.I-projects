A program meant to get familiar with pytorch by building a simple deep-learning model for predicting labels for the MNIST dataset.

1) get_data_loader(training=true) - returns Dataloader for the training set (if training = True) or the test set.

2) build_model() - returns an untrained model.

3) train_model(model, train_loader, criterion, T) - trains a model for T epochs.

4) evaluate_model(model, test_loader, criterion, show_loss=True) - prints the evaluation statistics (accuracy & average loss).

5) predict_label(model, test_images, index) - prints the top 3 most likely labels for the image at the given index with their probabilities.