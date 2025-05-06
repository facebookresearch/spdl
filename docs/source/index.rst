.. SPDL documentation master file, created by
   sphinx-quickstart on Fri Jun 14 19:39:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPDL (Scalable and Performant Data Loading)
===========================================

Publications

- `Introducing SPDL: Faster AI model training with thread-based data loading <https://ai.meta.com/blog/spdl-faster-ai-model-training-with-thread-based-data-loading-reality-labs/>`_ (Meta Engineering Blog)
- `Scalable and Performant Data Loading <https://arxiv.org/abs/2504.20067>`_ (arXiv)

Please use the following BibTex for citing our project if you find it useful.

.. code-block:: text

   @misc{hira2025scalableperformantdataloading,
      title={Scalable and Performant Data Loading}, 
      author={Moto Hira and Christian Puhrsch and Valentin Andrei and Roman Malinovskyy and Gael Le Lan and Abhinandan Krishnan and Joseph Cummings and Miguel Martin and Gokul Gunasekaran and Yuta Inoue and Alex J Turner and Raghuraman Krishnamoorthi},
      year={2025},
      eprint={2504.20067},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.20067}, 
   }
  
.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   getting_started/index
   performance_analysis/index
   migration/index
   best_practice
   examples
   fb/examples
   faq

.. toctree::
   :maxdepth: 2
   :caption: API References

   API Reference <api>
   API Reference (Meta) <fb/api>
   API Reference (C++) <cpp>
   API Index <genindex>

.. toctree::
   :maxdepth: 2
   :caption: Development Notes

   notes/index
