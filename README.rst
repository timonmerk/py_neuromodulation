py_neuromodulation
==================

This py_neuromodulation fork and rns_generator branch contains the code that was used to generate features of RNS data for the Brain-Modulation-Lab.
The computation script is examples/example_rns_stream_cluster.py
or, equivalently example_example_rns_stream_TURBO.py

Data was computed on the BIH-HPC Cluster https://bihealth.github.io/bih-cluster/. Slurm was used to call the run_1.sh script to allocate several jobs computing features for patients in parallel.

The feature definitions can be found in example_rns_stream_cluster.init_stream_all()

Features for all channels were computed, without rereferencing (since all channels are already bipolarly rereferenced).
Data was not normalized, since the .dat file recordings are not continuous. 

The following features were computed:
 - FFT, in frequency bands theta, alpha, low beta, high beta, low gamma and broadband
 - linelength
 - temporal sharpwave features
 - burst features, in low and high beta, low gamma
 - FOOOF, aperiodic components from BML Fooof fork (see Bush et al. 2023, https://doi.org/10.1101/2023.02.08.527719, code: https://github.com/Brain-Modulation-Lab/fooof/tree/lorentzian) 
 - Coherence, between channel pairs: (ch1, ch2), (ch1, ch3), (ch3, ch4) in high beta and low gamma frequency bands. In this way within and across side connectivity was calculated.

For further details see the documentation: https://neuromodulation.github.io/py_neuromodulation/
And the py_neuromodulation/settings.json file for parametrization details.

.. image:: https://app.travis-ci.com/neuromodulation/py_neuromodulation.svg?branch=main
    :target: https://app.travis-ci.com/neuromodulation/py_neuromodulation

------------------------------------------------

The py_neuromodulation toolbox allows for real time capable processing of multimodal electrophysiological data. The primary use is movement prediction for `adaptive deep brain stimulation <https://pubmed.ncbi.nlm.nih.gov/30607748/>`_.

Find the documentation here https://neuromodulation.github.io/py_neuromodulation/ for example usage and parametrization.

Setup
=====

For running this toolbox first create a new virtual conda environment and activate it:

.. code-block::

    conda create -n pynm-test python=3.10
    conda activate pynm-test

Then install the packages listed in the `pyproject.toml`.

.. code-block::

    pip install .[dev]
    pytest -v .


Optionally the ipython kernel can be specified to installed for the pynm-test conda environment:

.. code-block::

    ipython kernel install --user --name=pynm-test

Then py_neuromodulation can be imported via:

.. code-block::

    import py_neuromodulation as py_nm

The main modules include running real time enabled feature preprocessing based on `iEEG BIDS <https://www.nature.com/articles/s41597-019-0105-7>`_ data.

Different features can be enabled/disabled and parametrized in the `https://github.com/neuromodulation/py_neuromodulation/blob/main/pyneuromodulation/nm_settings.json>`_.

The current implementation mainly focuses band power and `sharpwave <https://www.sciencedirect.com/science/article/abs/pii/S1364661316302182>`_ feature estimation.

An example folder with a mock subject and derivate `feature <https://github.com/neuromodulation/py_neuromodulation/tree/main/examples/data>`_ set was estimated.

To run feature estimation given the example BIDS data run in root directory.

.. code-block::

    python examples/example_BIDS.py


This will write a feature_arr.csv and different sidecar files in the 'examples/data/derivatives' folder.

For further documentation view `ParametrizationDefinition <ParametrizationDefinition.html#>`_ for description of necessary parametrization files.
`FeatureEstimationDemo <FeatureEstimationDemo.html#>`_ walks through an example feature estimation and explains sharpwave estimation.
