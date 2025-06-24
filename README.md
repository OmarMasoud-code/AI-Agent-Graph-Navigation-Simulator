# AI Agent Graph Navigation Simulator

This project simulates agents navigating through graph structures using various strategies such as random walk, shortest path, breadth-first search (BFS), depth-first search (DFS), and A*. It provides tools to define custom graphs, measure graph metrics, and evaluate agent navigation performance.

## 📌 Features

- Custom graph creation using `Node` and `Edge` classes.
- Automatic shortest-path computation between all node pairs.
- Navigation strategies:
  - Random Walk
  - Shortest Path (Greedy)
  - BFS, DFS, A* (Structure in place)
- Graph metrics (e.g., degree, density).
- Agent memory and simulation logging.

## 📁 Project Structure

- `Node` and `Edge`: Core graph components.
- `Graph`: Stores nodes, edges, and computes shortest paths.
- `Agent`: Moves through the graph using specified navigation modes.
- `NavigationAgent`: Extended logic for managing and recording navigation sessions.
- `GraphMetrics`: Calculates degree and density.
- `World`: Handles simulation logic with random start/target nodes.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- No external libraries required (uses standard Python)

### Example Usage

```python
# Create nodes
nodes = [Node(i) for i in range(7)]

# Create edges and connect nodes
edges = [
    Edge(nodes[0], nodes[1]), Edge(nodes[0], nodes[2]),
    Edge(nodes[1], nodes[3]), Edge(nodes[1], nodes[4]),
    Edge(nodes[2], nodes[5]), Edge(nodes[2], nodes[6])
]

# Manually connect neighbors
nodes[0].neighbors = [nodes[1], nodes[2]]
nodes[1].neighbors = [nodes[0], nodes[3], nodes[4]]
nodes[2].neighbors = [nodes[0], nodes[5], nodes[6]]
# etc...

# Build graph
graph = Graph(nodes, edges)

# Create agent and run simulation
agent = Agent(start=0, target=6, graph=graph)
world = World(graph)
visited_nodes = world.run_simulation(agent)
print("Visited:", visited_nodes)
