type PROGRAM = str
type STATE = str

def run(ruleset: PROGRAM, initial_state: STATE):
    # Create a local namespace to execute the ruleset code
    local_namespace = {}
    exec(ruleset, globals(), local_namespace)
    
    # Extract the main function from the executed code
    step_fn = local_namespace['main']
    gen_initial_state_fn = local_namespace['gen_initial_state']
    done_fn = local_namespace['done']
    
    for initial_state in gen_initial_state_fn():
        world_state = initial_state
        while not done_fn(world_state):
            world_state = step_fn(world_state)
            yield world_state, False
        yield world_state, True

def gather_data(rulesets_and_initial_states: list[tuple[PROGRAM, list[STATE]]]):
    for ruleset, initial_states in rulesets_and_initial_states:
        for initial_state in initial_states:
            world = run(ruleset, initial_state)
            yield world

def main():
    rulesets_and_initial_states = [
        (open("rulesets/0.py", 'r').read(), [{"positions": [[0, 0, 0]], "velocities": [[0, 0, 0]], "masses": [1], "stiffness": [[0]], "damping": [[0]], "rest_length": [[0]]}])
    ]
    gather_data(rulesets_and_initial_states)