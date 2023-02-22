# chat-ia

# In this folder structure:
📂 **chatbot/**:

*  📁 **data**/: 
        This folder contains the data file that the chatbot will use to learn how to respond to user queries. In this case, the data is stored in a JSON file called intents.json. You can create additional data files or modify this file to suit your needs.

* 📁 **models**/:
        This folder contains the trained machine learning model that the chatbot will use to make predictions based on user input. In this case, the model is stored in a file called model.tflearn.

*  📝 **train_chatbot.py**: This is the script that you will use to train the chatbot on your data. It will read in the intents.json file, preprocess the data, and train a machine learning model using a library like TensorFlow or Keras.

*  📝 **chatbot.py**: This is the main script that you will use to run the chatbot. It will load the trained machine learning model from the models/ folder, and use it to respond to user queries.


### Project entry point:

The project entry point will be the `chatbot.py` file, which is the script that runs the chatbot and interacts with the user.

When the user starts the chatbot by running `chatbot.py`, the script will load the pre-trained machine learning model and other necessary data structures, 
and then it will begin listening for user input. The chatbot will process each input using the model and generate an appropriate response. 
The chatbot will continue listening for user input until the user chooses to exit.

The `train_chatbot.py` file is used to train the machine learning model on a dataset of user queries and corresponding responses. 
This file is run separately from `chatbot.py` and is not part of the project's entry point. Instead, it is used to generate the pre-trained model that is loaded by `chatbot.py`.

### Missing installation

run in the console: `pip install keras tensorflow`