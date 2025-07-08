from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Self, TypedDict, Union
import networkx as nx
import numpy as np

def parse_code_graph(code: str) -> nx.Graph:
    ...

def render_code_graph(code_graph: nx.Graph) -> str:
    ...

'''
Program Graph:
- control pointer node: node that points to the next node to execute
- control nodes: nodes that control rules match up with
- data nodes: other nodes
'''

# Connectivity classes (renamed from *Node)
class BaseProgramConnectivity(TypedDict):
    pass
class BaseStatementProgramConnectivity(BaseProgramConnectivity):
    pass
class ReturnStatementProgramConnectivity(BaseStatementProgramConnectivity):
    retval: 'ExpressionConnectivity'
class AssignmentStatementProgramConnectivity(BaseStatementProgramConnectivity):
    lhs: SymbolConnectivity
    rhs: ExpressionConnectivity
class ConditionalExecutionProgramConnectivity(BaseStatementProgramConnectivity):
    condition: ExpressionConnectivity
    true_branch_entrypoint: 'BaseStatementProgramConnectivity'
    false_branch_entrypoint: 'BaseStatementProgramConnectivity'
class DeclarationProgramConnectivity(BaseStatementProgramConnectivity):
    pass
class FunctionCallProgramConnectivity(BaseStatementProgramConnectivity):
    fn_symbol: SymbolConnectivity
    args: Dict[str, ExpressionConnectivity]
class SymbolConnectivity(BaseProgramConnectivity):
    class Config:
        name: str
        constraints: list[str] # in the future, we will have a value constraint system, starting with type constraints
class StdInConnectivity(SymbolConnectivity):
    class Config:
        name: 'stdin'
class StdOutConnectivity(SymbolConnectivity):
    class Config:
        name: 'stdout'
class StdErrConnectivity(SymbolConnectivity):
    class Config:
        name: 'stderr'
class ExitCodeConnectivity(SymbolConnectivity):
    class Config:
        name: 'exit_code'
class FunctionDefinitionConnectivity(BaseProgramConnectivity):
    entrypoint: 'BaseStatementProgramConnectivity'
    parameters: Dict[str, SymbolConnectivity]
class AtomicDataConnectivity(BaseProgramConnectivity):
    class Config:
        value: Any
class ListDataConnectivity(BaseProgramConnectivity):
    children: list[ExpressionConnectivity]
class DictDataConnectivity(BaseProgramConnectivity):
    children: dict[str, ExpressionConnectivity]
ExpressionConnectivity = Union[FunctionDefinitionConnectivity, AtomicDataConnectivity, ListDataConnectivity, DictDataConnectivity, FunctionCallProgramConnectivity]
class ProcessorConnectivity(BaseProgramConnectivity):
    current_control_pointer: 'BaseStatementProgramConnectivity'

# Node classes (mostly empty, with name and constraints on specific ones)
class BaseProgramNode:
    pass
class BaseStatementProgramNode(BaseProgramNode):
    pass
class ReturnStatementProgramNode(BaseStatementProgramNode):
    pass
class AssignmentStatementProgramNode(BaseStatementProgramNode):
    pass
class ConditionalExecutionProgramNode(BaseStatementProgramNode):
    pass
class DeclarationProgramNode(BaseStatementProgramNode):
    pass
class FunctionCallProgramNode(BaseStatementProgramNode):
    pass
class SymbolNode(BaseProgramNode):
    class Config:
        name: str
        constraints: list[str] # in the future, we will have a value constraint system, starting with type constraints
class StdInNode(SymbolNode):
    pass
class StdOutNode(SymbolNode):
    pass
class StdErrNode(SymbolNode):
    pass
class ExitCodeNode(SymbolNode):
    pass
class FunctionDefinitionNode(BaseProgramNode):
    pass
class AtomicDataNode(BaseProgramNode):
    class Config:
        value: Any
class ListDataNode(BaseProgramNode):
    pass
class DictDataNode(BaseProgramNode):
    pass
ExpressionNode = Union[FunctionDefinitionNode, AtomicDataNode, ListDataNode, DictDataNode, FunctionCallProgramNode]
class ProcessorNode(BaseProgramNode):
    pass

@dataclass
class Program:
    program_graph: nx.Graph
    locals: dict[str, nx.Graph] = field(default_factory=dict)
    
    def read_stdin(self: Self) -> nx.Graph:
        return self.locals['stdin']
    
    def write_stdout(self: Self, value: nx.Graph):
        self.locals['stdout'] = value
    
    def write_stderr(self: Self, value: nx.Graph):
        self.locals['stderr'] = value

    def set_exit_code(self: Self, value: int):
        self.locals['exit_code'] = value

    def step(self: Self) -> Self:
        ...

    def run(self: Self, max_steps: int) -> Self:
        for _ in range(max_steps):
            self.step()
        return self


# an object is a graph. an atomic object is just a 1-node graph
# a program is a graph with control nodes
# control statement nodes are used to modify other nodes
# definition statement nodes create control nodes

# the benchmark measures how efficiently an ai can learn programs
# by measuring the bit distance (error) between estimated program and actual program according to a weighted profile of distance techniques
# penalized by bits of program interaction (experience), and direct program observation (priors)

# intelligence = 1 / (E[error] x experience x priors)

class Env:
    """Base environment class"""
    pass

class PerformanceLearningPrevious:
    """Type for performance learning previous actions"""
    pass

class SingleProgramInstanceEnv(Env):
    program_instance: Program


class SingleProgramClassEnv(Env):
    program_class: Program

    def reset(self) -> Program:
        ...
    
    def step(self) -> Program:
        ...

class ProgramGeneratorEnv(Env):
    '''
    A simple program dataset would just be a non-markovian uniform random distribution of programs
    BUT the ProgramGeneratorEnv can also be used to adversarially generate programs that are on the
    edge of the agent's ability to learn. It could even be ANOTHER agent INSIDE this env that generates
    the programs in blindfolded self-challenge.
    '''

    def reset(self) -> Program:
        ...

    def step(self, action: PerformanceLearningPrevious) -> Program:
        ...



'''
in the future we will also make a generalization of this to have various rings of fixed and stationary characteristics

every environment ring includes:
- hidden state
- observable state
- action
- forward dynamics

program instance: this is a regular environment
- hidden state: general search latents, program graph, locals[, standard graph trajectory]
- observable state: [standard graph trajectory, ]standard graph output
- action: standard graph input
- forward dynamics: run the program. the next state is determined by applying the program rules to the locals which include the standard graph input
- initial input: ??? (i think there should not be an input or output since this is task-agnostic but i feel this should come from the agent at the ring above this agent)
- final output: ??? (i think there should not be an input or output since this is task-agnostic but i feel this should come from the agent at the ring above this agent)

program class:
- hidden state: general search latents, program graph
- observable state: standard graph trajectories
- action: none?
- forward dynamics: run a full program instance data collection episode with the new initial state for that episode determine by the action (but it may just be random sampling of all possible initial states)

general search:
- hidden state: general search latents
- observable state: none?
- action: none?
- forward dynamics: sample a new program class (using an LLM + code graph parser). it could be conditioned to sample programs that promote the greatest possible learning (though this would be tied to the AI so it might corrupt the evaluation metric).

'''