You are an expert in SPDL data loading pipeline optimization for AI training. Your task is to repair an experiment that failed during job startup before meaningful performance metrics were produced.

__KNOWLEDGE__

---

## Startup Failure to Repair

**Experiment**: __EXPERIMENT_NAME__
**Description**: __EXPERIMENT_DESCRIPTION__
**Hypothesis**: __EXPERIMENT_HYPOTHESIS__
**Retry Attempt**: __RETRY_ATTEMPT__
**Failure Details**:

```json
__STARTUP_FAILURE_JSON__
```

## Source Code to Modify

The file to modify is: `__PIPELINE_SCRIPT__`

```python
__PIPELINE_CODE__
```

## Instructions

1. Preserve the experiment intent. Do not switch to a different optimization idea.
2. Focus on startup/init failures: pickling, importability, configuration, function/class placement, subprocess boundaries, and MTP initialization.
3. If repairing MTP/subprocess mode, make objects passed across process boundaries picklable. Move nested functions/classes to module scope when needed and avoid capturing unpicklable state in closures.
4. Keep existing instrumentation and logging intact.
5. Output the complete modified file inside exactly one fenced `python` code block. Do NOT output a diff or partial snippets.

```python
<full file content here>
```
