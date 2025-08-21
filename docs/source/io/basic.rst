Loading media as array
======================

To simply load audio/video/image data from a file or in-memory buffer, you can use the following functions.

- :py:func:`spdl.io.load_audio`
- :py:func:`spdl.io.load_video`
- :py:func:`spdl.io.load_image`

They return an object of :py:class:`spdl.io.CPUBuffer` or :py:class:`spdl.io.CUDABuffer`.
The buffer object contains the decoded data as a contiguous memory.
It implements `the array interface protocol <https://numpy.org/doc/stable/reference/arrays.interface.html>`_ or `the CUDA array interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_,
which allows to convert the buffer object into commonly used array class without
making a copy.

SPDL has the following functions to cast the buffer to framework-specific class.

- :py:func:`spdl.io.to_numpy` †
- :py:func:`spdl.io.to_torch`
- :py:func:`spdl.io.to_jax` †
- :py:func:`spdl.io.to_numba`

† :py:func:`~spdl.io.to_numpy` and  :py:func:`~spdl.io.to_jax` support only CPU array.

.. admonition:: Example - Loading audio from bytes
   :class: note

   .. code-block::

      data: bytes = download_from_remote_storage("my_audio.mp3")
      buffer: CPUBuffer = spdl.io.load_audio(data)

      # Cast to NumPy NDArray
      array = spdl.io.to_numpy(buffer)  # shape: (time, channel)

.. admonition:: Example - Loading image from file
   :class: note

   .. code-block::

      buffer: CPUBuffer = spdl.io.load_image("my_image.jpg")

      # Cast to PyTorch tensor
      tensor = spdl.io.to_torch(buffer)  # shape: (height, width, channel=RGB)

.. admonition:: Example - Loading video from remote
   :class: note

   .. code-block::

      buffer: CPUBuffer = spdl.io.load_video("https://example.com/my_video.mp4")

      # Cast to PyTorch tensor
      tensor = spdl.io.to_torch(buffer)  # shape: (time, height, width, channel=RGB)


By default, image/video are converted to interleaved RGB format (i.e. ``NHWC`` where ``C=3``),
and audio is converted to 32-bit floating point sample of interleaved channels. (i.e. channel-last and ``dtype=float32``)

To change the output format, you can customize the conversion behavior by providing
a custom ``filter_desc`` value.

You can use :py:func:`spdl.io.get_audio_filter_desc` and
:py:func:`spdl.io.get_video_filter_desc` (for image and video) to construct
a filter description.

.. admonition:: Example - Customizing audio output format
   :class: note

   The following code snippet shows hot to decode audio into
   16k Hz, monaural, 16-bit signed integer with planar format (i.e. channel first).
   It also fix the duration to 5 seconds (80,000 samples) by silencing the residual
   or padding the silence at the end.

   .. code-block::

      buffer: CPUBuffer = spdl.io.load_audio(
          "my_audio.wav",
          filter_desc=spdl.io.get_audio_filter_desc(
              sample_rate=16_000,
              num_channels=1,
              sample_fmt="s16p",  # signed 16-bit, planar format
              num_frames=80_000,  # 5 seconds
          )
      )
      array = spdl.io.to_numpy(buffer)
      array.shape  # (1, 8000)
      array.dtype  # int16

.. admonition:: Example - Customizing the image output format
   :class: note

    
   .. code-block::

      buffer: CPUBuffer = spdl.io.load_video(
          "my_video.wav",
          filter_desc=spdl.io.get_video_filter_desc(
              frame_rate=30,      # Change frame rate by duplicating or culling frames.
              scale_width=256,    # Rescale the image to 256, 256, with bicubic
              scale_height=256,
              scale_algo='bicubic',
              scale_mode='pad',   # pad the image if the aspect ratio of the
                                  # original resolution and
                                  # rescaling do not match.
              crop_width=128,     # Perform copping
              crop_height=128,
              num_frames=10,      # Stop decoding after 10 frames.
                                  # If it's shorter than 10 frames,
                                  # insert extra frames.
              pad_mode="black"    # Specify the extra frames to be black.
          )
      )
      array = spdl.io.to_numpy(buffer)
      array.shape  # (10, 128, 128, 3)
      array.dtype  # uint8
