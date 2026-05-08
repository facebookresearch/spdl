You are an expert in SPDL data loading pipeline optimization for AI training. Your task is to apply a specific code change to a training/pipeline script.

__KNOWLEDGE__

---

## Change to Apply

**Experiment**: __EXPERIMENT_NAME__
**Description**: __EXPERIMENT_DESCRIPTION__
**Hypothesis**: __EXPERIMENT_HYPOTHESIS__

## Source Code to Modify

The file to modify is: `__PIPELINE_SCRIPT__`

```python
__PIPELINE_CODE__
```

## Instructions

1. Apply the described change to the source code. Follow the description precisely.
2. Preserve all existing functionality that is not explicitly being changed (instrumentation, logging, model setup, training loop structure).
3. **CRITICAL**: You MUST output the complete modified file content inside a single fenced code block with the `python` language tag. Do NOT output a diff, do NOT output partial snippets, do NOT skip the code block. The output MUST contain exactly one block in this format:

```python
# full file content here...
```

4. Make sure the resulting code is correct and will run without errors.
5. If the description mentions SPDL APIs (e.g. `run_pipeline_in_subprocess`, `continuous=True`), use them correctly according to the SPDL knowledge above.
6. **When adding new imports, verify the Buck target exists.** Check the BUCK Dependencies table in the knowledge base for common targets. For modules not listed, look up the target from the module's directory (e.g., `spdl/foo/BUCK` for `import spdl.foo`). Missing Buck deps cause `ModuleNotFoundError` at runtime in the deployed package.
7. Do NOT truncate the output. Output the ENTIRE file, even if it is long.

Output the modified file:

```python
<full file content here>
```
