## Summary
The paper trains GPT-2-small sized Language models on a custom generated grade school math dataset to investigate the emergence of reasoning capabilities in models.
By restricting the training data to only math examples witha reduced vocabulary, the authors avoid data contamination, allowing them to verify that observed behaviour can be contributed to OOD generalisation rather than memorisation.

They ask several research questions to understand the model's reasoning behaviour and capabilities:
* How do mistakes happen in model's reasoning process? Can we predict hallucinations in advance?
* Other research found that for memorisation tasks, the number of parameters of a model is a better predictor of capabilities than the depth or width of a model. Does that also hold for reasoning tasks?


### Methods
#### V-probes:
* linear probes added to the output  
  ✚
* low-rank modification of embeddings

➤ estimate the capability added by the training of low-rank adaptation and linear probes to the model's output by using the same process on a randomly initialised transformer and measuring the performance uplift.

If this is small, we assume it's likely that most of the capability comes from the training of the transformer itself, rather than the probes.

#### Custom GSM dataset:
* generated from dependency graphs and hierarchical vocabulary
* train on optimal (shortest) solutions

#### Random order in fact hints
Tasks are structured as a sequence of known facts followed by a question. To avoid biasing the model to a certain thinking pattern through the order of the facts, their order is randomised. This forces the model to decide when each fact is relevant to the current stage of the solution, rather than extracting information from the order.

#### Restriction to mod 23 arithmetic
To avoid simple arithmetic errors, the authors restrict the arithmetic to mod 23. This ensures that the model can easily memorise the addition and multiplication tables, and that any mistakes are likely to be due to reasoning errors rather than arithmetic errors. The reason for this is that they expect deployed models to have tools like calculators available to avoid such errors. Removing this source of error allows them to more cleanly attribute mistakes to reasoning errors.

#### Interpret model's reasoning process
Use V-probes to predict:
* `nece(A)`: if parameter `A` is necessary for computing the answer.
* `dep(A, B)`: if parameter `A` (recursively) depends on parameter `B` given the problem statement.
* `known(A)`: if parameter `A` has already been computed.
* `value(A)`: the value of parameter `A` (a number between 0-22, or 23 if `known(A) = false`).
* `can_next(A)`: if `A` can be computed in the next solution sentence (namely, its predecessors
have all been calculated). Note that `A` might not be necessary to answer the question.
* `nece_next(A)`: if parameter `A` satisfies both `can_next(A)` and `nece(A)`.

