# Learned rules

`glioma.json` holds the decision rules that SAGEAgent's semantic memory learned in the paper's experiments, one rule set per pipeline of the nested 5×5 cross-validation (keyed `outer_k/inner_m`, as in [`embedding/glioma/splits.json`](../embedding/glioma/splits.json)). A rule is plain text that the LLM reads in its decision prompt, so any chat model can use it.

To evaluate the rules without running experience accumulation (step 3), train the predictors and uncertainty heads, then run:

```bash
python evaluate.py --config configs/glioma.yaml --rules rules/glioma.json --no-episodic
```

[Getting Started](../docs/GETTING_STARTED.md#released-rules) describes the file format and how to export the rules of your own runs.
