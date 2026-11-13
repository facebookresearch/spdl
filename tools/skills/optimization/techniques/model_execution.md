# Model execution after resolving the input path

These five existing entries retain historical model-side experiments and their input-adjacent evidence. First resolve input starvation and trainer interference; benchmark an already-fed training path with unchanged distributed topology, global batch and data. This area is not a catalog of new model optimizations. The historical tested compiler configuration and alternative convolution backends failed numeric checks; they remain rejected variants.

[Catalog index](../technique_catalog.md)

The [cached model benchmark](../cached_model_benchmark.md) explains how to measure these candidates once traces show model compute or layout work on the critical path.

**`device_resident_matching`: remove per-sample host round trips in the loss**
- Trigger: the training step calls a CPU solver per sample (for example linear-sum assignment) on GPU-produced matrices, causing blocking D2H copies and many small kernels.
- Technique: use a batched GPU solver and keep cost construction, assignment, gathers and reductions on the device. Prepare dense metadata during collation.
- Guards: equal-cost ties allow different valid assignments. Test the objective value, losses and gradients, not tie choices.

**`reduced_precision_compute`: evaluate lower-precision compute without changing optimizer state**
- Trigger: on an already-fed pipeline, profiles show floating-point model computation is the exposed cost rather than acquisition or batch transfer.
- Technique: compare compatible BF16 model operations with the FP32 baseline while retaining FP32 master parameters/optimizer state where the training recipe requires them. Convert only supported operations and preserve loss-sensitive reductions; do not cast the entire training state blindly. The historical case used packed BF16 computation in eager execution and later BF16 feature computation.
- Guards: BF16 has lower precision; speed depends on hardware, operator shapes and conversion overhead. Validate output/loss/gradient behavior and final model quality against the original baseline. This is a model-specific experiment, not an SPDL loader utility.

**`norm_layout_materialization`: avoid redundant copies around normalization**
- Trigger: profiles show layout changes and contiguous copies around normalization taking time on an already-fed training path.
- Technique: retain a compatible tensor layout through adjacent operations, or use normalization that accepts the existing strides. A permutation can be a view, while making it contiguous may copy the whole tensor. Avoid the copy only when the next operation can consume that layout.
- Guards: preserve normalization axes, epsilon, dtype and reduction behavior. Check outputs and gradients and measure the full block: a strided input can slow a later kernel.

**`fused_copy_widen`: combine a required copy with dtype conversion**
- Trigger: the same input is first copied into a destination/staging buffer and then read again to widen its dtype, creating two passes over large data.
- Technique: where the destination API supports conversion, convert during the required copy instead of constructing an intermediate widened allocation. For example, copy compact integer values directly into the model's wider destination dtype in one conversion-capable operation. The copy and conversion still happen; the target is the intermediate allocation and extra memory pass.
- Guards: preserve cast rules, value range, destination layout, pinning and ownership. Moving conversion before IPC/H2D increases payload size. The standalone change was previously neutral, so compare the complete copy/conversion path instead of assuming fusion helps.

**`selective_kernel_replacement`: specialize only a measured slow operator case**
- Trigger: an operator is slow for a particular shape or stride pattern after input delivery is no longer limiting training.
- Technique: compare an alternative implementation for that supported case and retain the original fallback elsewhere. Include dispatch, conversions and synchronization in the comparison, not just the replacement kernel.
- Guards: check forward values, gradients, corner cases and unsupported-input fallback. Previously tested compiler and convolution alternatives failed numerical checks; an isolated fast operator is not enough to justify switching the whole backend.
