import pickle

# define the training data
training_data = {
    'words': ['hello', 'hi', 'goodbye', 'bye'],
    'classes': ['greeting', 'farewell'],
    'train_x': [
        [1, 1, 0, 0],  # hello
        [1, 0, 0, 0],  # hi
        [0, 0, 1, 1],  # goodbye
        [0, 0, 0, 1]   # bye
    ],
    'train_y': [
        [1, 0],  # greeting
        [1, 0],  # greeting
        [0, 1],  # farewell
        [0, 1]   # farewell
    ]
}

# save the training data to disk
with open('models/training_data', 'wb') as file:
    pickle.dump(training_data, file)