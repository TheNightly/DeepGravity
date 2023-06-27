import os
import sys
import subprocess

clients = int(sys.argv[1])

for i in range(clients):
    subprocess.Popen(f"cmd /C python flower_client.py --dataset new_york --oa-id-column GEOID --flow-origin-column geoid_o --flow-destination-column geoid_d --flow-flows-column pop_flows --epochs 1 --device gpu --mode train --client-size={clients} --client-idx={i}")