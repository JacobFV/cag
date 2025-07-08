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

    def step(self, action: PerformanceLearningPrevious) -> Program:
        ...



def eval(ai_factory):

    training_stats = []

    def program_generator() -> Generator[Program, None, None]:
        ...
    
    train(agent=meta_learning_agent, env=program_generator





# in the future we will also make a generalization of this to have various rings of fixed and stationary characteristics


'''
every environment ring includes:
- hidden state
- observable state
- action
- forward dynamics

program instance:
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