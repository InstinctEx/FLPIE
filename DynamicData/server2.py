import argparse
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
# Define a dummy metric aggregation function
def aggregate_metrics(metrics):
    return 0,0
# Dummy fit config function
def fit_config(round_num):
    return {"batch_size": 10, "epochs": 1}

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
args = parser.parse_args()
# Start Flower server
fl.server.start_server(
    server_address=args.server_address,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=aggregate_metrics),
)
