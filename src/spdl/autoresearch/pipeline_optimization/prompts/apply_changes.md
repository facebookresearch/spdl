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
2. **CRITICAL — single-file constraint**: You are modifying **only** the file shown above (`__PIPELINE_SCRIPT__`). The engine writes your output back to this exact file path. If the experiment description references a different file (e.g., `utils/pipeline.py`, `pipeline.py`, or any other module), you MUST translate the described change into a modification of the given file. Options:
   - If the target function (e.g., `build_pipeline()`) is imported from another module, move or inline the relevant code into this file and apply the change there.
   - If the change can be applied by wrapping or monkey-patching the imported function within this file, do that.
   - If the change truly cannot be expressed as a modification to this file, output the file UNCHANGED and add a comment `# AUTORESEARCH: change requires modifying a different file` at the top.
   **NEVER output the content of a different file** — the engine will overwrite `__PIPELINE_SCRIPT__` with your output, destroying all existing functions (main, parse_args, init_logging, train, etc.) and causing ImportError crashes.
3. **Preserve ALL existing top-level symbols**: The output file MUST contain every function, class, and module-level variable that exists in the input. Other modules import symbols from this file (e.g., `init_logging`, `main`, `parse_args`, `train`). Dropping any of them causes `ImportError` at runtime.

   **VERIFICATION CHECKLIST** (do this before outputting):
   - List every `def` and `class` from the input file. Confirm each one appears in your output.
   - If your output is missing `train()`, `main()`, `parse_args()`, or `init_logging()`, you are almost certainly outputting the content of a different file (e.g., `utils/pipeline.py`) instead of modifying the pipeline script. STOP and re-read the input file.
   - If your output no longer imports `torch.distributed`, `DDP`, or `torchvision`, you are outputting the wrong file.
   - The training loop, model creation, optimizer setup, and DDP wrapping must remain intact.
4. Preserve all existing functionality that is not explicitly being changed (instrumentation, logging, model setup, training loop structure).
5. **CRITICAL**: You MUST output the complete modified file content inside a single fenced code block with the `python` language tag. Do NOT output a diff, do NOT output partial snippets, do NOT skip the code block. The output MUST contain exactly one block in this format:

```python
# full file content here...
```

6. Make sure the resulting code is correct and will run without errors.
7. If the description mentions SPDL APIs (e.g. `run_pipeline_in_subprocess`, `continuous=True`), use them correctly according to the SPDL knowledge above.
   - **Error handling in pipeline stage functions**: When writing `try/except` blocks in pipeline stage callables (decode functions, fetch functions, etc.), ALWAYS log the exception before returning `None`. Do NOT use bare `except RuntimeError: return None` or `except Exception: return None` — these silently swallow errors like shape mismatches, type errors, and API misuse, making it impossible to diagnose why a pipeline produces zero output. Instead: `except RuntimeError as e: _LG.warning("stage failed: %s", e); return None`. This ensures failures are visible in pipeline stats and job logs.
8. **When adding new imports, verify the Buck target exists.** Check the BUCK Dependencies table in the knowledge base for common targets. For modules not listed, look up the target from the module's directory (e.g., `spdl/foo/BUCK` for `import spdl.foo`). Missing Buck deps cause `ModuleNotFoundError` at runtime in the deployed package.
9. Do NOT truncate the output. Output the ENTIRE file, even if it is long.
10. **Prefer modifying the existing pipeline factory function in-place.** If the codebase has a `build_pipeline()` (or similarly named) function that constructs the pipeline, modify it directly rather than creating a new separate function. This keeps the diff minimal and avoids code duplication. Only create a new function if the original must be preserved for a separate code path.
11. **MTP (subprocess) robustness**: When applying MTP changes, carefully inspect the existing pipeline code to understand its return types and structure:
   - `run_pipeline_in_subprocess()` requires a **`PipelineConfig`** as input (obtained via `builder.get_config()`). It does NOT accept a built `Pipeline` object or an iterator/iterable.
   - If the existing code has a function that calls `.build()` and/or `.get_iterator()` (returning a `Pipeline` or iterator), you MUST refactor it to return an unbuilt `PipelineBuilder` or `PipelineConfig` instead. Do NOT pass the result of `.build()` to `run_pipeline_in_subprocess`.
   - **Check type annotations and return statements** to determine what the existing function returns (`PipelineBuilder`, `Pipeline`, `Iterator`, `Iterable`, etc.) and adapt accordingly.
   - **Prefer module-level functions over bound methods as pipe callables in the backend pipeline.** While bound methods like `dataset.__getitem__` are picklable in Python 3.5+, module-level functions with `functools.partial` are the established SPDL convention. They make the serialization boundary explicit, produce clearer error messages when pickling fails, and can be extended with lazy initialization or thread-local storage without modifying the original class:
     ```python
     # AVOID: bound method — works but opaque
     backend.pipe(dataset.__getitem__, concurrency=8)

     # PREFERRED: module-level function + partial
     def _fetch(index: int, *, dataset):
         return dataset[index]
     backend.pipe(partial(_fetch, dataset=dataset), concurrency=8)
     ```
   - **Separate CPU and GPU stages**: The backend pipeline (running in the subprocess) must contain ONLY CPU-bound stages (fetch, decode, aggregate, collate). GPU stages like `transfer_tensor` require a CUDA context and MUST be in the frontend pipeline (main process). If the existing code includes GPU stages in the pipeline builder, exclude them from the backend config and add them to the frontend pipeline.
   - The frontend pipeline takes the subprocess iterable as its source and applies GPU transfer: `PipelineBuilder().add_source(subprocess_iterable, continuous=True).pipe(transfer_tensor).add_sink(buffer_size=3).build(num_threads=1)`.
   - **`Pipeline` is iterable, not iterator**: When iterating a `Pipeline` object, always use `pipeline.get_iterator(timeout=<seconds>)` to get an iterator with a timeout. Do NOT use `iter(pipeline)` or `for batch in pipeline` directly — these lack timeout handling.
12. **GPU video decode (NVDEC) robustness**: When replacing CPU video decode with GPU decode:
   - Replace `decode_packets(packets, ...)` + `convert_frames(frames)` + `transfer_buffer(buffer, ...)` with a single `decode_packets_nvdec(packets, device_config=cuda_cfg, pix_fmt="rgb")`. The result is a `CUDABuffer` already on GPU.
   - Remove the `transfer_buffer` or `transfer_tensor` stage from the pipeline — data is already on GPU after NVDEC decode.
   - Create `CUDAConfig` with PyTorch's caching allocator: `spdl.io.cuda_config(device_index=..., allocator=(torch.cuda.caching_allocator_alloc, torch.cuda.caching_allocator_delete))`.
   - The `device_index` must match the current GPU rank. Use `torch.cuda.current_device()` or the rank/local_rank variable from the training setup.
   - Set decode concurrency based on GPU hardware — H100/B100 have 7 NVDEC instances per GPU (concurrency ~7; higher for sparse decoding patterns). Older GPUs (A100, V100) have 3–5 slots.
   - If the existing code applies FFmpeg filters for resize/crop, replace them with `scale_width`/`scale_height`/`crop_*` parameters on `decode_packets_nvdec` (dimensions must be even numbers).
   - `decode_packets_nvdec` auto-applies bitstream filtering — remove any manual `apply_bsf()` calls.
   - **Use a dedicated `ThreadPoolExecutor` for the NVDEC decode stage.** Do NOT rely on the pipeline's default shared thread pool. Create a separate `ThreadPoolExecutor(max_workers=C)` and pass it as `executor=` to the `.pipe()` call for the decode stage. This prevents NVDEC decode threads from competing with fetch/other CPU threads for the same pool, which is critical because NVDEC decode needs its own CUDA contexts per thread. Example: `.pipe(decode_fn, concurrency=7, executor=ThreadPoolExecutor(7))`.
   - **Compatible with MTP (subprocess mode)** — `VideoPackets` are picklable, so MTP and GPU decode can be combined. Run CPU-bound stages (fetch, demux) in the subprocess backend, then decode with NVDEC in the frontend (main process). The NVDEC decode stage must use a dedicated `ThreadPoolExecutor` in the frontend. See Option A (MTP + GPU decode) in the knowledge base for the recommended pattern.
13. **TorchTNT scripts**: If the code uses TorchTNT (`torchtnt.framework.fit`, `train`, `AutoUnit`), the SPDL `Pipeline` is passed directly to TorchTNT as the `train_dataloader` (Pipeline is iterable — no wrapper class is required unless instrumentation was added). When applying changes:
   - **Pipeline construction changes** (concurrency, MTP, batch size): Modify the function that builds the `PipelineBuilder`, same as non-TorchTNT code. The `Pipeline` abstracts MTP vs pure multithreading, so the code passing Pipeline to TorchTNT does not change.
   - **Do NOT modify TorchTNT internals** (`fit()`, `train()`, `AutoUnit.train_step`). Only modify the pipeline construction.
   - The pipeline is built once and iterated many times. `auto_stop()` is obsolete — do not call it, and do not rebuild per epoch.

Output the modified file:

```python
<full file content here>
```
