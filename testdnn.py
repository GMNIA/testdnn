from keras.models import Sequential
from keras.layers import Dense
import keras as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def unscale_normal(target, mean, std):
    """Unscale normalized target values back to original scale.

    Keyword arguments:
    target -- normalized target values
    mean -- mean of the original data
    std -- standard deviation of the original data
    """
    return target * std + mean


def train_model(layer_structure, num_epochs):
    """Train a neural network model using Keras.

    Keyword arguments:
    layer_structure -- list of integers representing the number of nodes in each layer
    num_epochs -- number of epochs for training
    """
    # Load the dataset
    df = pd.read_csv("experimentaltestresults.csv", delimiter='\t')

    # Drop unwanted columns
    df = df.drop(columns=['DesignationOfTest ', 'R_pl ', 'R_cr ', 'EC3_class', 'NominalYoungModulus '])

    # Extract the target column and remove it from the dataframe
    y_column = df[['ultimate_load_amplification_factor ']]
    df = df.drop(columns=['ultimate_load_amplification_factor '])

    # Convert the dataframe to a NumPy array and append the target column
    dataset = np.column_stack((df.values, y_column.values))

    # Convert categorical data to numerical values
    for i in range(dataset.shape[0]):
        if dataset[i, 0] == "SHS":
            dataset[i, 0] = 0
        elif dataset[i, 0] == "RHS":
            dataset[i, 0] = 1
        dataset[i, 1] = float(dataset[i, 1].replace("S", ""))

    # Convert the entire dataset to numeric (float)
    dataset = dataset.astype(float)

    # Scale the data into the interval [-1, 1]
    mean_y = 0
    std_y = np.std(dataset[:, -1])
    for j in range(dataset.shape[1]):
        dataset[:, j] = dataset[:, j] / np.std(dataset[:, j])

    # Shuffle the dataset
    np.random.shuffle(dataset)

    # Split the dataset into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    train_size = 0.9
    train_X, train_y = X[:int(train_size * X.shape[0])], y[:int(train_size * y.shape[0])]
    test_X, test_y = X[int(train_size * X.shape[0]):], y[int(train_size * y.shape[0]):]

    # Ensure input data types are correct
    train_X = np.asarray(train_X).astype(np.float32)
    train_y = np.asarray(train_y).astype(np.float32)
    test_X = np.asarray(test_X).astype(np.float32)
    test_y = np.asarray(test_y).astype(np.float32)

    # Define the Keras model
    print('Building model...')
    model = Sequential()
    random_normal_init = K.initializers.RandomNormal(mean=0.5, stddev=0.25)
    for idx, nodes in enumerate(layer_structure):
        if idx == 0:
            print('Adding input layer')
            model.add(Dense(nodes, input_dim=train_X.shape[1], activation='softmax',
                            kernel_initializer=random_normal_init, bias_initializer=random_normal_init))
        else:
            print(f'Adding hidden layer {idx}')
            model.add(Dense(nodes, activation='relu', kernel_initializer=random_normal_init,
                            bias_initializer=random_normal_init))
    # Output layer
    model.add(Dense(1, activation='relu', kernel_initializer=random_normal_init, bias_initializer=random_normal_init))
    print(model.summary())

    # Compile the Keras model
    print('Compiling model...')
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the Keras model on the dataset
    print('Fitting model...')
    model.fit(train_X, train_y, epochs=num_epochs, batch_size=100, verbose=1)

    # Make predictions with the model
    print('Making predictions...')
    predictions_test = model.predict(test_X)[:, 0]
    predictions_train = model.predict(train_X)[:, 0]

    # Clear the session after training to free up memory
    K.backend.clear_session()

    # Unscale predictions and true values
    print('Scaling back...')
    y_hat_train = unscale_normal(predictions_train, mean_y, std_y)
    y_hat_test = unscale_normal(predictions_test, mean_y, std_y)
    y_train = unscale_normal(train_y, mean_y, std_y)
    y_test = unscale_normal(test_y, mean_y, std_y)

    # Calculate variance for loss value
    print('Calculating variance for loss...')
    loss_test = np.var(y_test - y_hat_test)
    loss_train = np.var(y_train - y_hat_train)

    return loss_test, loss_train, y_test, y_hat_test


def save_results_to_csv(file_path, header, data):
    """Save the results to a CSV file.

    Keyword arguments:
    file_path -- path to the file where the results will be saved
    header -- list of column names for the CSV file
    data -- list of data to be saved
    """
    data = [[str(item) for item in row] for row in data]
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)
        for line in data:
            writer.writerow(line)


def generate_plot(save_path, x=None, y=None, x_label='Predicted', y_label='Expected'):
    """Generate and save a plot comparing predicted and expected values.

    Keyword arguments:
    save_path -- file path where the plot will be saved
    x -- x-axis values (predicted)
    y -- y-axis values (expected)
    x_label -- label for the x-axis (default 'Predicted')
    y_label -- label for the y-axis (default 'Expected')
    """
    plt.plot(x, y, 'r^', [min(min(x), min(y)), max(max(x), max(y))], [min(min(x), min(y)), max(max(x), max(y))], '-')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(save_path)
    plt.clf()


if __name__ == '__main__':
    # Initialize the list for storing results
    results = []
    node_structure_list = [[nodes] * layers for layers in range(3, 5) for nodes in [5, 10, 15, 25]]
    epoch_list = [10, 20, 50, 200, 500]

    # Iterate over the layer structures and epoch values
    for node_structure in node_structure_list:
        for epochs in epoch_list:
            # Train the model and get the results
            loss_test, loss_train, y_test, y_hat_test = train_model(node_structure, epochs)

            # Save the results to the CSV file
            header = ['nodes', 'layers', 'epochs', 'loss_test', 'loss_train']
            results.append([node_structure[0], len(node_structure), epochs, loss_test, loss_train])
            save_results_to_csv('summaryresults.csv', header, results)

            # Generate and save the plot
            plot_file_name = f'plot_nodes{node_structure[0]}_layers{len(node_structure)}_epochs{epochs}.png'
            plot_file_path = os.path.join('plotresults', plot_file_name)
            generate_plot(plot_file_path, x=y_hat_test, y=y_test)
