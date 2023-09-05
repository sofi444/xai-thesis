## LLM Responses Directory

filename format: <run-id>_<freetext|parsed>_<generation-model>_<parsing-model>_<data>.jsonl

each file also contains
+ idx from the run
+ uuid from the original dataset to keep track of the processed instances

tmp responses/ contains old or bad responses that might still be useful for showing differences, etc.

#### freetext
the full generation from the model.

#### parsed
parsed generation with external parser, mainly to extract the prediction label.

#### turbo
gpt-3.5-turbo openAI LLM. Via Azure I use the 0301 version as it is the only one deployed (which is going to be deprecated eventually).

#### codellama
LLM used for output parsing.

#### eval
responses after being evaluated against the gold standard (contains the gold label and whether the llm's prediction was correct)


