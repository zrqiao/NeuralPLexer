============
NeuralPLexer
============

Official implementation of NeuralPLexer, a deep generative model to jointly predict protein-ligand complex 3D structures and beyond.

.. image:: docs/demo2_122023.gif
  :align: center
  :width: 600

Reference
---------

    Qiao Z, Nie W, Vahdat A, Miller III TF, Anandkumar A. State-specific protein-ligand complex structure prediction with a multi-scale deep generative model. *Nature Machine Intelligence*, 2024. https://doi.org/10.1038/s42256-024-00792-z.

Pretrained model checkpoints described in the published manuscript, downstream evaluation datasets, and predicted structures are available at the following Zenodo repository for **non-commercial usage** under the CC BY-NC-SA 4.0 license: https://doi.org/10.5281/zenodo.10373581.

Installation
------------

A GPU machine with CUDA>=10.2 support is required to run the model. 

We recommend setting up the `libmambda solver for conda <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`_ before installation.
For a Linux environment, run the following commands to install the package:

.. code-block:: bash

    make environment
    make install


Model inference for new protein-ligand pairs
--------------------------------------------

Example usage for the base model with a template structure in pdb format:

.. code-block:: bash

    neuralplexer-inference --task=batched_structure_sampling \
                           --input-receptor input.pdb \
                           --input-ligand <ligand>.sdf \
                           --use-template  --input-template <template>.pdb \
                           --out-path <output_path> \
                           --model-checkpoint <data_dir>/models/complex_structure_prediction.ckpt \
                           --n-samples 16 \
                           --chunk-size 4 \
                           --num-steps=40 \
                           --cuda \
                           --sampler=langevin_simulated_annealing


NeuralPLexer CLI supports the prediction of biological complexes without ligands, with a single ligand, with multiple ligands (e.g. substrate-cofactor systems), 
and/or with receptors of single or multiple protein chains. Common input options are:

- :code:`input-receptor` and :code:`input-ligand` are the input protein and ligand structures;
    - :code:`input-receptor` can be either a PDB file or protein sequences. In case the input is a multi-chain protein in the primary sequence format, the chains should be separated by a :code:`|` sign; in case the input is a PDB file, no coordinate information from the file is used for generation unless the file itself is separately provided as a template structure via :code:`input-template`.
    - :code:`input-ligand` can be either sdf files or SMILES strings. In case the input is a multi-ligand complex, the ligands should be separated by a :code:`|` sign;
- :code:`use-template` and :code:`input-template` are the options to use a template structure for the input protein;
- :code:`out-path` is the output directory to store the predicted structures;
- :code:`model-checkpoint` is the path to the trained model checkpoint;
- :code:`n-samples` is the number of conformations to generate in total;
- :code:`chunk-size` is the number of conformation to generate in parallel;
- :code:`num-steps` is the number of steps for the diffusion part of the sampling process;
- :code:`separate-pdb` determines whether to output the predicted protein structures into dedicated PDB files;
- :code:`rank-outputs-by-confidence` determines whether to rank-order the predicted ligand (and potentially protein) output files, where outputs are ranked using the predicted ligand confidence if available and using the predicted protein confidence otherwise;


Expected outputs under :code:`<output_path>`:


- :code:`prot_all.pdb` and :code:`lig_all.sdf` contains the output geometries of all `n_samples` predicted conformations of the biological assembly;
    - `prot_0.pdb`, `prot_1.pdb`, ... stores the individual frames of the predicted protein conformations;
    - `lig_0.sdf`, `lig_1.sdf`, ... stores the individual frames of the predicted ligand conformations.

In :code:`benchmark_tiny.sh` we also provided minimal example commands for running complex generation over many distinct input
sets using data provided in in the Zenodo repo, analogous to the process used
to obtain the benchmarking results but with reduced number of samples, denoising steps, and template choices.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

