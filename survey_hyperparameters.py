"""
This documents contains the dictionary survey_hyperparameters which specifies the
values to perform grid search over when training the mental health model.
Feel free to modify however you like, but be aware that adding too
many configurations will slow training down.
"""
"*** Your Code Here ***"
survey_hyperparameters = {
    'input_dim': [8], # Do not change
    'output_dim': [2], # Do not change
    'hidden_dim': [10, 20, 40, 80, 100],
    'layers': [1, 2],
    'learning_rate': [-0.005, -0.01],
    'epochs': [1, 2],
    'batch_size': [1, 2, 4],
}