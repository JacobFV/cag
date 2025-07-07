# architecture
# ai_instance
# env_generator
# env
# (env, initial_state)
# abstract
# concrete
# THIS SYSTEM IS USING GRAPHS!

from typing import Callable


type State = dict
type Action = dict
type Rule = tuple[Trigger, Change]
type Env = list[Rule]
type Traj = list[tuple[State, Action, State]]


class AbstractAgent:
    def __init__(self):
        pass

class ConcreteAgent:
    def __init__(self):
        pass

observe_only_policy = lambda *a, **kw: []

config = {
    'off_policy_episodes_per_env': 10,
    'off_policy_steps_per_episode': 100,
    'on_policy_episodes_per_env': 10,
    'on_policy_steps_per_episode': 100,
}

def benchmark(
        abstract_agent_factory: Callable[[], AbstractAgent], 
        concrete_agent_factory: Callable[[], ConcreteAgent], 
        env_generator: Callable[[], Env] = None):

    # first, just collect a broad sampling of abstract data across all environments
    for env in env_generator():
        for episode in range(config['off_policy_episodes_per_env']):
            off_policy_data = driver(env, agent=observe_only_policy, steps=config['off_policy_steps_per_episode'])

    # next, run on-policy directly on abstract data
    for env in env_generator():
        agent = abstract_agent_factory()
        data = driver(env, agent)
        train(agent, data)
        evaluate(agent, data)

    # next, run on-policy on concrete data
    concrete_synthetic_data_collection(
        agent_factory=concrete_agent_factory, 
        env_generator=env_generator, 
        env_run_kwargs={'max_steps': 100}
    )
