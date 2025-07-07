from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional
import random
import copy


def main():
    print("Hello from cag!")
    
    # Example: Generate and execute a simple ruleset
    generator = RulesetGenerator()
    ruleset = generator.generate_ruleset(complexity=2)
    
    # Sample initial state
    initial_state = {"grid": [[0, 1], [1, 0]], "position": (0, 0)}
    
    # Execute ruleset
    result = ruleset.execute(initial_state)
    print(f"Initial: {initial_state}")
    print(f"Result: {result}")


# Classes and functions defined first


@dataclass
class Graph:
    nodes: list[str]
    edges: list[tuple[str, str]]


@dataclass 
class Rule(ABC):
    name: str
    
    @abstractmethod
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_applicable(self, state: Dict[str, Any]) -> bool:
        pass


@dataclass
class World:
    state: Graph
    rules: List[Rule]


# Basic atomic rules
class MoveRule(Rule):
    def __init__(self, direction: Tuple[int, int]):
        super().__init__(f"move_{direction}")
        self.direction = direction
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        if "position" in state:
            pos = state["position"]
            new_pos = (pos[0] + self.direction[0], pos[1] + self.direction[1])
            new_state["position"] = new_pos
        return new_state
    
    def is_applicable(self, state: Dict[str, Any]) -> bool:
        return "position" in state


class FlipRule(Rule):
    def __init__(self, coord: Tuple[int, int]):
        super().__init__(f"flip_{coord}")
        self.coord = coord
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        if "grid" in state:
            grid = new_state["grid"]
            x, y = self.coord
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                grid[x][y] = 1 - grid[x][y]
        return new_state
    
    def is_applicable(self, state: Dict[str, Any]) -> bool:
        return "grid" in state


class ConditionalRule(Rule):
    def __init__(self, condition: Callable, then_rule: Rule, else_rule: Optional[Rule] = None):
        super().__init__(f"if_{condition.__name__}")
        self.condition = condition
        self.then_rule = then_rule
        self.else_rule = else_rule
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.condition(state):
            return self.then_rule.apply(state)
        elif self.else_rule:
            return self.else_rule.apply(state)
        return state
    
    def is_applicable(self, state: Dict[str, Any]) -> bool:
        return True


class SequenceRule(Rule):
    def __init__(self, rules: List[Rule]):
        super().__init__(f"seq_{len(rules)}")
        self.rules = rules
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_state = state
        for rule in self.rules:
            if rule.is_applicable(current_state):
                current_state = rule.apply(current_state)
        return current_state
    
    def is_applicable(self, state: Dict[str, Any]) -> bool:
        return len(self.rules) > 0


class Ruleset:
    def __init__(self, rules: List[Rule]):
        self.rules = rules
    
    def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        current_state = initial_state
        for rule in self.rules:
            if rule.is_applicable(current_state):
                current_state = rule.apply(current_state)
        return current_state


class RulesetGenerator:
    def __init__(self):
        self.atomic_rules = [
            lambda: MoveRule((0, 1)),  # right
            lambda: MoveRule((0, -1)), # left  
            lambda: MoveRule((1, 0)),  # down
            lambda: MoveRule((-1, 0)), # up
            lambda: FlipRule((0, 0)),
            lambda: FlipRule((0, 1)),
            lambda: FlipRule((1, 0)),
            lambda: FlipRule((1, 1)),
        ]
        
        self.conditions = [
            lambda state: state.get("position", (0, 0))[0] == 0,  # at_top
            lambda state: state.get("position", (0, 0))[1] == 0,  # at_left
            lambda state: state.get("grid", [[0]])[0][0] == 1,    # cell_is_one
        ]
    
    def generate_atomic_rule(self) -> Rule:
        return random.choice(self.atomic_rules)()
    
    def generate_conditional_rule(self) -> Rule:
        condition = random.choice(self.conditions)
        then_rule = self.generate_atomic_rule()
        else_rule = self.generate_atomic_rule() if random.random() < 0.5 else None
        return ConditionalRule(condition, then_rule, else_rule)
    
    def generate_sequence_rule(self, length: int) -> Rule:
        rules = [self.generate_atomic_rule() for _ in range(length)]
        return SequenceRule(rules)
    
    def generate_ruleset(self, complexity: int = 1) -> Ruleset:
        rules = []
        
        # Add some atomic rules
        for _ in range(complexity):
            rules.append(self.generate_atomic_rule())
        
        # Add conditional rules
        if complexity > 1:
            rules.append(self.generate_conditional_rule())
        
        # Add sequence rules
        if complexity > 2:
            rules.append(self.generate_sequence_rule(2))
        
        return Ruleset(rules)


# Approach 2: Symbolic Programming with S-expressions
class SExpression:
    def __init__(self, op: str, args: List[Any]):
        self.op = op
        self.args = args
    
    def __repr__(self):
        return f"({self.op} {' '.join(map(str, self.args))})"


class SymbolicRuleGenerator:
    def __init__(self):
        self.primitives = ['move', 'flip', 'rotate', 'swap']
        self.combinators = ['seq', 'if', 'while', 'map']
        self.predicates = ['at_edge', 'cell_is', 'position_is']
        
    def generate_expression(self, depth: int = 3) -> SExpression:
        if depth == 0:
            # Base case: primitive operation
            op = random.choice(self.primitives)
            if op == 'move':
                return SExpression(op, [random.choice(['up', 'down', 'left', 'right'])])
            elif op == 'flip':
                return SExpression(op, [random.randint(0, 1), random.randint(0, 1)])
            else:
                return SExpression(op, [])
        else:
            # Recursive case: combinator
            op = random.choice(self.combinators)
            if op == 'seq':
                return SExpression(op, [self.generate_expression(depth-1) for _ in range(2)])
            elif op == 'if':
                pred = random.choice(self.predicates)
                then_expr = self.generate_expression(depth-1)
                else_expr = self.generate_expression(depth-1)
                return SExpression(op, [pred, then_expr, else_expr])
            else:
                return SExpression(op, [self.generate_expression(depth-1)])


# Approach 3: Graph Rewriting Systems
@dataclass
class GraphNode:
    id: str
    label: str
    attributes: Dict[str, Any]


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str


class GraphState:
    def __init__(self, nodes: List[GraphNode], edges: List[GraphEdge]):
        self.nodes = {n.id: n for n in nodes}
        self.edges = edges
    
    def copy(self):
        return GraphState(list(self.nodes.values()), self.edges.copy())


class GraphRewriteRule:
    def __init__(self, name: str, pattern: GraphState, replacement: GraphState):
        self.name = name
        self.pattern = pattern
        self.replacement = replacement
    
    def apply(self, graph: GraphState) -> Optional[GraphState]:
        # Simple pattern matching - in practice you'd want more sophisticated matching
        if self.matches(graph):
            return self.rewrite(graph)
        return None
    
    def matches(self, graph: GraphState) -> bool:
        # Simplified matching logic
        return len(graph.nodes) >= len(self.pattern.nodes)
    
    def rewrite(self, graph: GraphState) -> GraphState:
        # Apply rewrite rule
        new_graph = graph.copy()
        # ... actual rewriting logic would go here
        return new_graph


# Approach 4: Domain-Specific Language (DSL)
class DSLParser:
    def __init__(self):
        self.keywords = ['if', 'then', 'else', 'while', 'do', 'move', 'flip', 'rotate']
        
    def parse_rule(self, rule_text: str) -> Rule:
        # This would be a full parser in practice
        tokens = rule_text.split()
        return self.parse_tokens(tokens)
    
    def parse_tokens(self, tokens: List[str]) -> Rule:
        if not tokens:
            return None
        
        if tokens[0] == 'move':
            direction = tokens[1] if len(tokens) > 1 else 'right'
            direction_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
            return MoveRule(direction_map.get(direction, (0, 1)))
        
        elif tokens[0] == 'flip':
            x, y = int(tokens[1]), int(tokens[2]) if len(tokens) > 2 else 0, 0
            return FlipRule((x, y))
        
        elif tokens[0] == 'if':
            # Parse conditional
            condition = lambda state: True  # Simplified
            then_rule = self.parse_tokens(tokens[2:4])
            else_rule = self.parse_tokens(tokens[4:6]) if len(tokens) > 4 else None
            return ConditionalRule(condition, then_rule, else_rule)
        
        return None


class DSLRuleGenerator:
    def __init__(self):
        self.templates = [
            "move {direction}",
            "flip {x} {y}",
            "if at_edge then move {direction}",
            "if cell_is 1 then flip {x} {y} else move {direction}",
        ]
        
    def generate_rule_text(self) -> str:
        template = random.choice(self.templates)
        return template.format(
            direction=random.choice(['up', 'down', 'left', 'right']),
            x=random.randint(0, 1),
            y=random.randint(0, 1)
        )


# Approach 5: Cellular Automata Rule Generation
class CellularAutomataRule:
    def __init__(self, rule_number: int):
        self.rule_number = rule_number
        self.lookup_table = self.generate_lookup_table(rule_number)
    
    def generate_lookup_table(self, rule_number: int) -> Dict[Tuple[int, int, int], int]:
        """Generate Wolfram rule lookup table"""
        binary = format(rule_number, '08b')
        return {
            (1, 1, 1): int(binary[0]),
            (1, 1, 0): int(binary[1]),
            (1, 0, 1): int(binary[2]),
            (1, 0, 0): int(binary[3]),
            (0, 1, 1): int(binary[4]),
            (0, 1, 0): int(binary[5]),
            (0, 0, 1): int(binary[6]),
            (0, 0, 0): int(binary[7]),
        }
    
    def apply_to_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply CA rule to 2D grid"""
        new_grid = copy.deepcopy(grid)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                neighborhood = self.get_neighborhood(grid, i, j)
                new_grid[i][j] = self.lookup_table.get(neighborhood, 0)
        return new_grid
    
    def get_neighborhood(self, grid: List[List[int]], i: int, j: int) -> Tuple[int, int, int]:
        """Get 3-cell neighborhood (simplified to 1D for this example)"""
        left = grid[i][(j-1) % len(grid[0])]
        center = grid[i][j]
        right = grid[i][(j+1) % len(grid[0])]
        return (left, center, right)


class CABasedRuleGenerator:
    def __init__(self):
        self.interesting_rules = [30, 90, 110, 150, 184]  # Known interesting CA rules
    
    def generate_ca_rule(self) -> CellularAutomataRule:
        rule_number = random.choice(self.interesting_rules)
        return CellularAutomataRule(rule_number)


# Enhanced main function to demonstrate different approaches
def enhanced_main():
    print("=== CAG: Enhanced Rule Generation Approaches ===\n")
    
    # Approach 1: Compositional Grammar (existing)
    print("1. Compositional Grammar Approach:")
    generator = RulesetGenerator()
    ruleset = generator.generate_ruleset(complexity=2)
    initial_state = {"grid": [[0, 1], [1, 0]], "position": (0, 0)}
    result = ruleset.execute(initial_state)
    print(f"   Initial: {initial_state}")
    print(f"   Result: {result}\n")
    
    # Approach 2: Symbolic Programming
    print("2. Symbolic Programming Approach:")
    sym_gen = SymbolicRuleGenerator()
    expr = sym_gen.generate_expression(depth=2)
    print(f"   Generated expression: {expr}\n")
    
    # Approach 3: Graph Rewriting
    print("3. Graph Rewriting Approach:")
    nodes = [GraphNode("n1", "agent", {}), GraphNode("n2", "goal", {})]
    edges = [GraphEdge("n1", "n2", "moves_to")]
    graph = GraphState(nodes, edges)
    print(f"   Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges\n")
    
    # Approach 4: DSL
    print("4. Domain-Specific Language Approach:")
    dsl_gen = DSLRuleGenerator()
    rule_text = dsl_gen.generate_rule_text()
    print(f"   Generated rule: {rule_text}\n")
    
    # Approach 5: Cellular Automata
    print("5. Cellular Automata Approach:")
    ca_gen = CABasedRuleGenerator()
    ca_rule = ca_gen.generate_ca_rule()
    test_grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    ca_result = ca_rule.apply_to_grid(test_grid)
    print(f"   CA Rule {ca_rule.rule_number}")
    print(f"   Before: {test_grid}")
    print(f"   After:  {ca_result}\n")
    
    print("=== Recommendations ===")
    print("Start with Approach 1 (Compositional Grammar) for simplicity")
    print("Add DSL (Approach 4) for human readability")
    print("Consider Graph Rewriting (Approach 3) for complex relational reasoning")
    print("Use Cellular Automata (Approach 5) for emergent behaviors")


# Call enhanced demo
if __name__ == "__main__":
    enhanced_main()