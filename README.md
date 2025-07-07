# CAG: Computational Abstraction & Generalization

1. At CAG devtime, the CAG developer designs a simple, general-purpose graph world system
2. At world gentime, the world creator samples a ruleset and one or more initial states
3. ABSTRACT SYNTHETIC DATA GENERATION: the CAG system runs the world ruleset on the initial state to generate abstract input-output pairs
4. OFF-POLICY ABSTRACT SYNTHETIC DATA COLLECTION: the CAG system samples a fixed policy on the world ruleset on the initial state to generate abstract input-output pairs
5. ON-POLICY ABSTRACT SYNTHETIC DATA COLLECTION: the CAG system samples a learning policy on the world ruleset on the initial state to generate abstract input-output pairs

Train the AI on varying degrees of representational abstraction:

- graphs (direct abstract representations)
- text (mostly a graph, but flattened)
- grids (low abstraction)
- 3d worlds (extremely distant representational-to-abstraction correspondance)

Additional features:

- various corruptions, increasing at lower degrees of abstraction
- dropout, drop-on(?)

I feel like there should be some upper bound on the amount of noise that can be tolerated for a given level of representational abstraction/density. eg, grids can tolerate much more noise within a higher error margin because of the redundancy and the ruleset.

---

- make synthetic environments
- make a unified graph representation space

---

THE conrete rendition of the abstract representiaton does not need to be differentiable

alternatively it COULD be differentiable just to provide some gradients:
- differentiable render engine
- non-differentiable graph restructuring into a euclidean lattice but differentiable values
Although the poolibng operation requires making some non-differentiable selection of nodes for inclusion/exclusion. this might bnot be as problemnatic if we constrain the graph size along the way