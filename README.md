# CAG: Abstract Computational Generalization

CAG helps shift focus from memorization to compositional generalization. 


Prolog or Bend2 for the rules

train an AI to learn these rules when the system is expressed in unstructured modalities


        the dataset is all possible system rulesets!
        x -> true system rules -> y
        x -> ai -> y prediction


for all rules

---

i don't have time to work on this but we need to learn to learn and i think compositional abstraction and generalization is the lowest hanging fruit on this agenda. i think this is going to become even more important with agentic vlms that need to preserve this structure as a first class citizen in their latent representations as opposed to embeddings which 'flatten' the inherent structure. Not that the structure can't be recovered with probes or network decomposition, but latent dynamics are much more efifcient instilled as architectural priors than learned through gradient descent

1. sample from rulesets r \in R,
1.1. sample from initial states x \in X(r),
1.1.1. determine the result state y = r(x)
1.1.2. render the unstructured states x', y' = f_render(x), f_render(y)
1.1.3. let the AI learn from x', y'

min num_examples max accuracy

but actually separate out the collection and training phases

this is kinda like inverse rendering, meta learning, and program synthesis.

you can see how this would be modified to build environments

partly based on @fchollet's talk

this paper seems helpful:

---

x: state

individually(x, fn): applies fn to each elem in x individually

conditionally(cond, true_val, false_val)
