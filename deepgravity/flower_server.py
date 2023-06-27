from typing import List, Tuple

import flwr as fl
import sys
from flwr.common import Metrics

num_clients = int(sys.argv[1])

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["cpc"] for num_examples, m in metrics]
    stddev = [num_examples * m["cpc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"cpcs": sum(accuracies) / sum(examples), "std-dev":sum(stddev) / sum(examples)}

strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,min_available_clients=num_clients)
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=20), strategy=strategy)  