"""
This documents contains the configuration dictionary for MNIST digit classification

You set the hyperparameter values when training.
"""
"*** Your Code Here Jerry Li ***"
student_config = {
    'input_dim': 784,  # DO NOT CHANGE
    'output_dim': 10,  # DO NOT CHANGE
    # Change the following
    'hidden_dim': 375,
    'layers': 4,
    'learning_rate': -0.02,
    'epochs': 5,
    'batch_size': 20
}
