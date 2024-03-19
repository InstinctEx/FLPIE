import argparse
import time
import numpy as np

import flwr as fl
import tensorflow as tf
from tensorflow import keras

# Function to generate simulated sensor data (replace this with your actual sensor data source)
def generate_sensor_data():
    while True:
        # Simulate sensor data for CO2 level
        co2_level = np.random.randint(400, 1000)
        yield co2_level
        time.sleep(1)  # Simulate sensor data update rate

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, generator):
        self.generator = generator
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(1,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, params):
        self.model.set_weights(params)

    def fit(self, parameters, config):
        x_train, y_train = [], []
        for _ in range(config['batch_size']):
            # Generate a batch of sensor data
            co2_level = next(self.generator)
            x_train.append(co2_level)
            y_train.append(co2_level * 0.5)  # Simulated target label

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        self.model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
        return self.get_parameters({}), len(x_train), {}

    def evaluate(self, parameters, config):
        x_val, y_val = [], []
        batch_size = config.get('batch_size', 10)
        for _ in range(batch_size):
            # Generate a batch of sensor data for validation
            co2_level = next(self.generator)
            x_val.append(co2_level)
            y_val.append(co2_level * 0.5)  # Simulated target label

        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # Evaluate the model on the validation data
        loss = self.model.evaluate(x_val, y_val, verbose=0)
        # Return the loss and the number of validation examples
        return loss, len(x_val), {"accuracy": loss}  # Modify as needed

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Client")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
    args = parser.parse_args()




    fl.client.start_client(
        server_address=args.server_address, 
        client = FlowerClient(generate_sensor_data()).to_client(),
    )

if __name__ == "__main__":
    main()
