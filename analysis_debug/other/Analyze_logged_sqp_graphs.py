import pickle
import pathlib

# Define path to the log file (make sure the correct path is used)
log_dir = pathlib.Path("./graph_logs")
log_file = log_dir / "graph_solve0000.pkl"  # Adjust if needed

# Load the graph data
with open(log_file, 'rb') as f:
    graph_data = pickle.load(f)

# Access the components of the graph
states = graph_data['states']
nodes = graph_data['nodes']
edges = graph_data['edges']
node_type = graph_data['node_type']
senders = graph_data['senders']
receivers = graph_data['receivers']

print("Loaded graph data:")
# print(f"States shape: {states.shape}")
# print(f"Nodes shape: {nodes.shape}")
# print(f"Edges shape: {edges.shape}")
# print(f"Node types: {node_type}")
# print(f"Senders: {senders}")
# print(f"Receivers: {receivers}")
print(f"{edges}")
