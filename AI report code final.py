from typing import List, Tuple
import random

class Node:
    def __init__(self, val: int):
        # Constructor for Node class that takes an integer value as input
        self.val = val
        self.neighbors = []  # List to store neighboring nodes

class Edge:
    def __init__(self, node1: Node, node2: Node):
        # Constructor for Edge class that takes two nodes as input
        self.node1 = node1
        self.node2 = node2

class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        # Constructor for Graph class that takes a list of nodes and a list of edges as input
        self.nodes = nodes
        self.edges = edges
        self.shortest_paths = self._compute_shortest_paths()

    def _compute_shortest_paths(self) -> dict:
        # Function to compute the shortest paths between all pairs of nodes in the graph
        dist = {}
        for node in self.nodes:
            # Initialize distance dictionary for each node
            dist[node.val] = {}
            for other_node in self.nodes:
                # Set distance to infinity for all other nodes
                dist[node.val][other_node.val] = float('inf')
            dist[node.val][node.val] = 0  # Set distance to 0 for current node
            queue = [node]  # Initialize queue with current node
            while queue:
                curr_node = queue.pop(0)  # Remove first node from queue
                for neighbor in curr_node.neighbors:
                    if dist[node.val][neighbor.val] == float('inf'):
                        # Update distance if it has not been set before
                        dist[node.val][neighbor.val] = dist[node.val][curr_node.val] + 1
                        queue.append(neighbor)  # Add neighbor to queue
        return dist

    def get_node(self, val: int) -> Node:
        # Function to get the node with the given value
        for node in self.nodes:
            if node.val == val:
                return node
        return None

class Agent:
    def __init__(self, start: int, target: int, graph: Graph):
        # Constructor for Agent class that takes a starting node, target node, and graph as input
        self.start = start
        self.target = target
        self.graph = graph
        self.visited_nodes = []  # List to store visited nodes
        self.current_node = self.graph.get_node(start)  # Set current node to starting node
        self.shortest_path = self.graph.shortest_paths[start][target]  # Get shortest path from starting node to target node

    def sense_state(self) -> Tuple[int, List[int]]:
        # Function to return current node value and list of visited nodes
        return self.current_node.val, self.visited_nodes

    def random_walk(self):
        # Function to randomly select a neighboring node and move to it
        neighbors = self.current_node.neighbors
        if neighbors:
            next_node = random.choice(neighbors)  # Randomly select a neighboring node
            self.visited_nodes.append(self.current_node.val)  # Add current node to visited nodes list
            self.current_node = next_node  # Move to next node
        else:
            raise ValueError("Current node has no neighbors")  # Raise error if current node has no neighbors

    def shortest_path_walk(self):
        if self.current_node.val != self.target:
            neighbors = self.current_node.neighbors
            shortest_path_length = float('inf')
            next_node = None
            for neighbor in neighbors:
                # Check if a neighbor has a shorter path to the target node
                if self.graph.shortest_paths[neighbor.val][self.target] < shortest_path_length:
                    shortest_path_length = self.graph.shortest_paths[neighbor.val][self.target]
                    next_node = neighbor
            self.visited_nodes.append(self.current_node.val)
            self.current_node = next_node
        else:
            # If the target node has already been reached, raise an error
            raise ValueError("Target node already reached")

class Agent:
    def __init__(self, start, target, graph):
        self.location = start
        self.target = target
        self.path = None
        self.graph = graph
        
    def navigate(self, mode):
        if mode == 'bfs':
            self.path = bfs(self.location, self.target, self.graph)
        elif mode == 'dfs':
            self.path = dfs(self.location, self.target, self.graph)
        elif mode == 'astar':
            self.path = astar(self.location, self.target, self.graph)
        else:
            raise ValueError('Invalid mode')





class GraphMetrics:
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def degree(self, val: int) -> int:
        # Returns the degree (number of neighbors) of a node with a given value
        node = self.graph.get_node(val)
        return len(node.neighbors)
    
    def density(self) -> float:
        # Computes the density of the graph
        n = len(self.graph.nodes)
        m = len(self.graph.edges)
        if n <= 1:
            return 0.0
        return 2 * m / (n * (n - 1))


class NavigationAgent:
    def __init__(self, graph: Graph, shortest_paths: dict):
        self.graph = graph
        self.shortest_paths = shortest_paths
        self.memory = []

    def navigate(self, start_node: Node, target_node: Node, mode: str) -> List[Node]:
        """
        Navigate from start_node to target_node using either a random walk or the shortest path.

        Args:
            start_node (Node): The starting node.
            target_node (Node): The target node.
            mode (str): The mode to use for navigation. Either "random" or "shortest".

        Returns:
            A list of nodes visited during navigation from start_node to target_node.
        """
        visited_nodes = []
        curr_node = start_node
        while curr_node != target_node:
            visited_nodes.append(curr_node)
            if mode == "random":
                next_node = self._random_walk(curr_node)
            elif mode == "shortest":
                next_node = self._shortest_path(curr_node, target_node)
            if next_node is None:
                return None
            curr_node = next_node
        visited_nodes.append(curr_node)
        self.memory.append(visited_nodes)
        return visited_nodes

    def _random_walk(self, curr_node: Node) -> Node:
        """
        Perform a random walk from curr_node to one of its neighbors.

        Args:
            curr_node (Node): The current node.

        Returns:
            The next node to visit, chosen at random from the current node's neighbors.
        """
        neighbors = curr_node.neighbors
        if not neighbors:
            return None
        return random.choice(neighbors)

    def _shortest_path(self, curr_node: Node, target_node: Node) -> Node:
        """
        Determine the next node to visit on the shortest path from curr_node to target_node.

        Args:
            curr_node (Node): The current node.
            target_node (Node): The target node.

        Returns:
            The next node to visit on the shortest path from curr_node to target_node.
        """
        path = self.shortest_paths[curr_node][target_node]
        if not path:
            return None
        return path[1]

    def get_memory(self) -> List[List[Node]]:
        """
        Get the memory of visited nodes during navigation.

        Returns:
            A list of lists, where each inner list represents the visited nodes during a single navigation.
        """
        return self.memory

    def clear_memory(self):
        """
        Clear the memory of visited nodes during navigation.
        """
        self.memory = []

class World:
    def __init__(self, graph: Graph):
        self.graph = graph

    def run_simulation(self, agent: Agent) -> List[int]:
        # Select a random start and target node from the graph
        start_node = random.choice(self.graph.nodes)
        target_node = random.choice(self.graph.nodes)
        while target_node == start_node:
            target_node = random.choice(self.graph.nodes)

        # Navigate from start to target node using the agent
        visited_nodes = agent.navigate(start_node.val, target_node.val, mode="shortest_path")

        # Count all visited nodes and store the result
        num_visited_nodes = len(visited_nodes)
        return visited_nodes
    
#this part of the code is the simulation part where I would create the agent that I coded and simulate it, however the code is giving plenty of errors   
## Create a graph object
#nodes = [Node(i) for i in range(7)]
#edges = [Edge(nodes[0], nodes[1]), Edge(nodes[0], nodes[2]), Edge(nodes[1], nodes[3]),
#         Edge(nodes[1], nodes[4]), Edge(nodes[2], nodes[5]), Edge(nodes[2], nodes[6])]
#graph = Graph(nodes, edges)


## Create an agent and a world object, and run the simulation
#agent = Agent(start=0, target=6, graph=graph)
#world = World(graph)
#visited_nodes = world.run_simulation(agent)
#print(visited_nodes)


## Repeat the simulation multiple times and store the results
#num_simulations = 10
#results = []
#for i in range(num_simulations):
#    visited_nodes = world.run_simulation(agent)
#   results.append(visited_nodes)

# Print the number of visited nodes for each simulation
#for i, visited_nodes in enumerate(results):
#    print(f"Simulation {i+1}: {len(visited_nodes)} nodes visited")


