from typing import Generator
import networkx as nx
from openrl.envs import Env

Program = str
Graph = nx.Graph


def code2graph(code: str) -> nx.Graph:
    ...

# an object is a graph. an atomic object is just a 1-node graph
# a program is a graph with control nodes
# control statement nodes are used to modify other nodes
# definition statement nodes create control nodes

# the benchmark measures how efficiently an ai can learn programs
# by measuring the bit distance (error) between estimated program and actual program according to a weighted profile of distance techniques
# penalized by bits of program interaction (experience), and direct program observation (priors)

# intelligence = 1 / (E[error] x experience x priors)

class SingleProgramInstanceEnv(Env):
    program_class: Program
    initial_state: Graph


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

    def step(self) -> Program:
        ...



def eval(ai_factory):

    training_stats = []

    def program_generator() -> Generator[Program, None, None]:
        ...
    
    train(agent=meta_learning_agent, env=program_generator





# in the future we will also make a generalization of this to have various rings of fixed and stationary characteristics