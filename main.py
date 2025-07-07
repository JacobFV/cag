# architecture
# ai_instance
# env_generator
# env
# (env, initial_state)
# abstract
# concrete
# THIS SYSTEM IS USING GRAPHS!

from typing import Callable, Optional, List, Union
import numpy as np
import networkx as nx  # NEW: Subgraph matching library
from dataclasses import dataclass


# Graph representation types
type NodeId = int
type EdgeType = int
type NodeFeatures = np.ndarray  # Feature vector for each node
type EdgeFeatures = np.ndarray  # Feature vector for each edge

@dataclass
class Graph:
    """Graph representation compatible with GNNs"""
    # Adjacency matrix: [num_nodes, num_nodes, num_edge_types]
    adjacency: np.ndarray
    
    # Node features: [num_nodes, node_feature_dim]
    node_features: np.ndarray
    
    # Edge features: [num_nodes, num_nodes, edge_feature_dim]
    edge_features: np.ndarray
    
    # Optional node and edge masks for variable-sized graphs
    node_mask: Optional[np.ndarray] = None  # [num_nodes]
    edge_mask: Optional[np.ndarray] = None  # [num_nodes, num_nodes]
    
    def to_tensor_dict(self) -> dict:
        """Convert to tensor dictionary for GNN input"""
        return {
            'adjacency': self.adjacency,
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'node_mask': self.node_mask,
            'edge_mask': self.edge_mask
        }

    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert internal graph to a NetworkX MultiDiGraph for algorithms."""
        G = nx.MultiDiGraph()
        num_nodes = self.node_features.shape[0]
        for i in range(num_nodes):
            if self.node_mask is not None and not self.node_mask[i]:
                continue
            G.add_node(i, features=self.node_features[i])
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.edge_mask is not None and not self.edge_mask[i, j]:
                    continue
                # Assume single edge type for now (index 0)
                if self.adjacency[i, j, 0] != 0:
                    G.add_edge(i, j, features=self.edge_features[i, j])
        return G

    @staticmethod
    def from_networkx(G: nx.MultiDiGraph, node_feature_dim: int, edge_feature_dim: int, max_nodes: int):
        """Create Graph from a NetworkX graph."""
        g = create_empty_graph(max_nodes, node_feature_dim, edge_feature_dim)
        for node, data in G.nodes(data=True):
            g.node_mask[node] = True
            g.node_features[node] = data.get('features', np.zeros(node_feature_dim))
        for u, v, data in G.edges(data=True):
            g.edge_mask[u, v] = True
            g.adjacency[u, v, 0] = 1
            g.edge_features[u, v] = data.get('features', np.zeros(edge_feature_dim))
        return g

@dataclass
class GraphPattern:
    """Pattern for graph matching (L-pattern in graph rewriting)"""
    pattern_graph: Graph
    # Optional: constraints on node/edge features for matching
    node_constraints: Optional[dict] = None
    edge_constraints: Optional[dict] = None

@dataclass
class GraphRewriteRule:
    """Graph rewriting rule: L-pattern → R-pattern"""
    # Left-hand side: pattern to match
    lhs: GraphPattern
    
    # Right-hand side: replacement pattern
    rhs: Graph
    
    # Optional: conditions for rule application
    condition: Optional[Callable[[Graph], bool]] = None
    
    # Priority for rule application (higher = applied first)
    priority: float = 0.0
    
    def to_tensor_dict(self) -> dict:
        """Convert rule to tensor representation for GNN"""
        return {
            'lhs_pattern': self.lhs.pattern_graph.to_tensor_dict(),
            'rhs_replacement': self.rhs.to_tensor_dict(),
            'priority': np.array([self.priority])
        }

# Updated type definitions
type State = Graph
type Action = List[GraphOperation]  # Redefine Action type
type Rule = GraphRewriteRule
type Env = list[Rule]  # Environment is a set of rewrite rules
type Traj = list[tuple[State, Action, State]]

class GraphMatcher:
    """Finds subgraph matches for rule application using NetworkX VF2 algorithm"""

    def __init__(self):
        # Define feature comparison functions
        self.node_match = lambda n1, n2: np.array_equal(n1.get('features'), n2.get('features'))
        self.edge_match = lambda e1, e2: np.array_equal(e1.get('features'), e2.get('features'))

    def find_matches(self, graph: Graph, pattern: GraphPattern) -> List[dict]:
        G = graph.to_networkx()
        P = pattern.pattern_graph.to_networkx()
        matcher = nx.algorithms.isomorphism.MultiDiGraphMatcher(G, P,
            node_match=self.node_match,
            edge_match=self.edge_match)
        return list(matcher.subgraph_isomorphisms_iter())

    def apply_rule(self, graph: Graph, rule: GraphRewriteRule, match: dict) -> Graph:
        """Apply rule by replacing matched nodes with RHS graph (simple version)."""
        # Convert to networkx for manipulation
        G = graph.to_networkx()
        # Remove matched nodes
        G.remove_nodes_from(match.keys())
        # Compose RHS graph and relabel to new indices
        rhs_nx = rule.rhs.to_networkx()
        offset = max(G.nodes, default=-1) + 1
        mapping = {n: n + offset for n in rhs_nx.nodes}
        rhs_nx = nx.relabel_nodes(rhs_nx, mapping)
        G.update(rhs_nx)
        # Convert back to Graph
        new_graph = Graph.from_networkx(G, graph.node_features.shape[1], graph.edge_features.shape[2], graph.node_features.shape[0])
        return new_graph

class GraphEnvironment:
    """Environment that applies graph rewrite rules"""
    
    def __init__(self, rules: list[GraphRewriteRule]):
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.matcher = GraphMatcher()
    
    def step(self, state: State, action: Action) -> tuple[State, float, bool]:
        """Apply operations then graph rewrite rules"""
        current_graph = apply_operations(state, action) if action else state
        reward = 0.0
        done = False
        # apply rules as before
        for rule in self.rules:
            if rule.condition and not rule.condition(current_graph):
                continue
            matches = self.matcher.find_matches(current_graph, rule.lhs)
            if matches:
                current_graph = self.matcher.apply_rule(current_graph, rule, matches[0])
                reward += 1.0
                break
        if self._is_terminal(current_graph):
            done = True
        return current_graph, reward, done
    
    def _is_terminal(self, graph: Graph) -> bool:
        """Check if graph represents a terminal state"""
        # TODO: Implement termination logic
        return False

# Helper functions for creating common graph structures
def create_empty_graph(max_nodes: int, node_feature_dim: int, edge_feature_dim: int) -> Graph:
    """Create an empty graph with specified dimensions"""
    return Graph(
        adjacency=np.zeros((max_nodes, max_nodes, 1)),
        node_features=np.zeros((max_nodes, node_feature_dim)),
        edge_features=np.zeros((max_nodes, max_nodes, edge_feature_dim)),
        node_mask=np.zeros(max_nodes, dtype=bool),
        edge_mask=np.zeros((max_nodes, max_nodes), dtype=bool)
    )

def create_rule_from_tensors(lhs_tensors: dict, rhs_tensors: dict) -> GraphRewriteRule:
    """Create a graph rewrite rule from tensor dictionaries (for GNN output)"""
    lhs_graph = Graph(
        adjacency=lhs_tensors['adjacency'],
        node_features=lhs_tensors['node_features'],
        edge_features=lhs_tensors['edge_features'],
        node_mask=lhs_tensors.get('node_mask'),
        edge_mask=lhs_tensors.get('edge_mask')
    )
    
    rhs_graph = Graph(
        adjacency=rhs_tensors['adjacency'],
        node_features=rhs_tensors['node_features'],
        edge_features=rhs_tensors['edge_features'],
        node_mask=rhs_tensors.get('node_mask'),
        edge_mask=rhs_tensors.get('edge_mask')
    )
    
    return GraphRewriteRule(
        lhs=GraphPattern(pattern_graph=lhs_graph),
        rhs=rhs_graph
    )

class AbstractAgent:
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        # TODO: Initialize GNN model for processing graph states
        pass
    
    def select_action(self, state: State) -> Action:
        """Select action based on current graph state"""
        # TODO: Use GNN to process state and select action
        return {'action_type': 'explore', 'target_node': 0}
    
    def process_graph(self, graph: Graph) -> np.ndarray:
        """Process graph through GNN to get representation"""
        # TODO: Implement GNN forward pass
        tensor_dict = graph.to_tensor_dict()
        # Placeholder: return flattened adjacency matrix
        return tensor_dict['adjacency'].flatten()

class ConcreteAgent:
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        # TODO: Initialize concrete GNN model
        pass
    
    def select_action(self, state: State) -> Action:
        """Select action for concrete environment"""
        # TODO: Implement concrete action selection
        return {'action_type': 'apply_rule', 'rule_id': 0}

def observe_only_policy(state: State, *args, **kwargs) -> Action:
    """Policy that only observes without taking meaningful actions"""
    return {'action_type': 'observe'}

def driver(env: GraphEnvironment, agent, steps: int = 100) -> Traj:
    """Run agent in graph environment and collect trajectory"""
    trajectory = []
    
    # Initialize with empty graph or random graph
    current_state = create_empty_graph(
        max_nodes=10, 
        node_feature_dim=64, 
        edge_feature_dim=32
    )
    
    for step in range(steps):
        # Agent selects action
        if hasattr(agent, 'select_action'):
            action = agent.select_action(current_state)
        else:
            action = agent(current_state)
        
        # Environment processes action
        next_state, reward, done = env.step(current_state, action)
        
        # Store transition
        trajectory.append((current_state, action, next_state))
        
        current_state = next_state
        
        if done:
            break
    
    return trajectory

def train(agent, trajectory: Traj):
    """Train agent on trajectory data"""
    # TODO: Implement training loop for GNN
    print(f"Training agent on {len(trajectory)} transitions")
    pass

def evaluate(agent, trajectory: Traj):
    """Evaluate agent performance"""
    # TODO: Implement evaluation metrics
    print(f"Evaluating agent on {len(trajectory)} transitions")
    pass

def concrete_synthetic_data_collection(agent_factory, env_generator, env_run_kwargs):
    """Collect synthetic data from concrete environments"""
    for env_rules in env_generator():
        env = GraphEnvironment(env_rules)
        agent = agent_factory()
        
        trajectory = driver(env, agent, **env_run_kwargs)
        train(agent, trajectory)
        evaluate(agent, trajectory)

config = {
    'off_policy_episodes_per_env': 10,
    'off_policy_steps_per_episode': 100,
    'on_policy_episodes_per_env': 10,
    'on_policy_steps_per_episode': 100,
}

def benchmark(
        abstract_agent_factory: Callable[[], AbstractAgent], 
        concrete_agent_factory: Callable[[], ConcreteAgent], 
        env_generator: Callable[[], list[list[GraphRewriteRule]]] = None):
    """Benchmark agents on graph environments"""
    
    if env_generator is None:
        env_generator = lambda: [create_example_ruleset()]
    
    # First, collect broad sampling of abstract data across all environments
    print("Collecting off-policy data...")
    for env_rules in env_generator():
        env = GraphEnvironment(env_rules)
        for episode in range(config['off_policy_episodes_per_env']):
            off_policy_data = driver(
                env, 
                agent=observe_only_policy, 
                steps=config['off_policy_steps_per_episode']
            )
            print(f"Collected {len(off_policy_data)} off-policy transitions")

    # Next, run on-policy directly on abstract data
    print("Running abstract agents...")
    for env_rules in env_generator():
        env = GraphEnvironment(env_rules)
        agent = abstract_agent_factory()
        data = driver(env, agent, steps=config['on_policy_steps_per_episode'])
        train(agent, data)
        evaluate(agent, data)

    # Next, run on-policy on concrete data
    print("Running concrete agents...")
    concrete_synthetic_data_collection(
        agent_factory=concrete_agent_factory, 
        env_generator=env_generator, 
        env_run_kwargs={'steps': 100}
    )

# Example usage and rule creation
def create_example_ruleset() -> list[GraphRewriteRule]:
    """Create example graph rewrite rules for testing"""
    rules = []
    
    # Example rule: Add a node connected to existing nodes
    lhs_pattern = create_empty_graph(max_nodes=3, node_feature_dim=64, edge_feature_dim=32)
    rhs_replacement = create_empty_graph(max_nodes=4, node_feature_dim=64, edge_feature_dim=32)
    
    # Configure the pattern and replacement (simplified example)
    lhs_pattern.node_mask = np.array([True, True, False])  # 2 nodes in pattern
    rhs_replacement.node_mask = np.array([True, True, True, False])  # 3 nodes in replacement
    
    rule = GraphRewriteRule(
        lhs=GraphPattern(pattern_graph=lhs_pattern),
        rhs=rhs_replacement,
        priority=1.0
    )
    
    rules.append(rule)
    return rules

def example_gnn_integration():
    """Example of how this integrates with GNNs"""
    
    # Create a rule
    rule = create_example_ruleset()[0]
    
    # Convert to tensor format for GNN processing
    rule_tensors = rule.to_tensor_dict()
    
    print("Rule as tensors:")
    for key, value in rule_tensors.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Create a graph state
    state = create_empty_graph(max_nodes=5, node_feature_dim=64, edge_feature_dim=32)
    state_tensors = state.to_tensor_dict()
    
    print("\nState as tensors:")
    for key, value in state_tensors.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")

# ================= Graph Operations =================
class OpType:
    NODE_ADD = 'NODE_ADD'
    NODE_MUL = 'NODE_MUL'
    EDGE_ADD = 'EDGE_ADD'
    EDGE_MUL = 'EDGE_MUL'
    NODE_CREATE = 'NODE_CREATE'
    NODE_REMOVE = 'NODE_REMOVE'
    EDGE_CREATE = 'EDGE_CREATE'
    EDGE_REMOVE = 'EDGE_REMOVE'

@dataclass
class GraphOperation:
    op: str
    params: dict

def apply_operations(graph: Graph, operations: Action) -> Graph:
    """Apply a sequence of operations to a graph and return modified graph."""
    g = graph  # operate directly for now
    for operation in operations:
        op = operation.op
        p = operation.params
        if op == OpType.NODE_ADD:
            nid = p['node_id']
            g.node_features[nid] += p['vec']
        elif op == OpType.NODE_MUL:
            nid = p['node_id']
            g.node_features[nid] *= p['vec']
        elif op == OpType.EDGE_ADD:
            u, v = p['src'], p['dst']
            g.edge_features[u, v] += p['vec']
        elif op == OpType.EDGE_MUL:
            u, v = p['src'], p['dst']
            g.edge_features[u, v] *= p['vec']
        elif op == OpType.NODE_CREATE:
            # find first unused slot
            idxs = np.where(~g.node_mask)[0] if g.node_mask is not None else []
            if len(idxs) == 0:
                continue  # graph full
            nid = idxs[0]
            g.node_mask[nid] = True
            g.node_features[nid] = p['vec']
        elif op == OpType.NODE_REMOVE:
            nid = p['node_id']
            g.node_mask[nid] = False
            g.node_features[nid] = 0
            g.edge_mask[nid, :] = False
            g.edge_mask[:, nid] = False
            g.adjacency[nid, :] = 0
            g.adjacency[:, nid] = 0
        elif op == OpType.EDGE_CREATE:
            u, v = p['src'], p['dst']
            g.edge_mask[u, v] = True
            g.adjacency[u, v, 0] = 1
            g.edge_features[u, v] = p['vec']
        elif op == OpType.EDGE_REMOVE:
            u, v = p['src'], p['dst']
            g.edge_mask[u, v] = False
            g.adjacency[u, v, 0] = 0
            g.edge_features[u, v] = 0
    return g

if __name__ == "__main__":
    # Example usage
    print("Graph-based CAG System Example")
    print("=" * 40)
    
    example_gnn_integration()
    
    print("\nRunning benchmark...")
    benchmark(
        abstract_agent_factory=lambda: AbstractAgent(),
        concrete_agent_factory=lambda: ConcreteAgent(),
        env_generator=lambda: [create_example_ruleset()]
    )
