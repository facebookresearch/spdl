.. SPDL documentation master file, created by
   sphinx-quickstart on Fri Jun 14 19:39:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPDL (Scalable and Performant Data Loading)
===========================================

Publications
------------

- `Optimizing Data Loading for Efficient AI Model Training <https://github.com/facebookresearch/spdl/releases/download/release-assets/Hira_optimizing_data_loading_for_efficient_ai_model_traininig_@scale_2025.pdf>`_ (`@Scale: AI & DATA <https://atscaleconference.com/scale-data-ai-infra/?tab=1&item=36#agenda-item-36>`_, 2025-06-25)
- `Scalable and Performant Data Loading <https://arxiv.org/abs/2504.20067>`_ (arXiv, 2025-04-23)
- `Introducing SPDL: Faster AI model training with thread-based data loading <https://ai.meta.com/blog/spdl-faster-ai-model-training-with-thread-based-data-loading-reality-labs/>`_ (Meta Engineering Blog, 2024-11-22)

Citation
--------

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
   optimization_guide/index
   case_studies/index
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
