.. Python Anesthesia Simulator documentation master file, created by
   sphinx-quickstart on Sat Apr  1 13:30:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Python Anesthesia Simulator's documentation!
=======================================================

The Python Anesthesia Simulator (PAS) models the effect of drugs on physiological variables during total 
intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark 
for the design of multidrug controllers. The available drugs are Propofol, Remifentanil, and Norepinephrine, 
the outputs are the Bispectral index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO), and Tolerance 
of Laryngoscopy (TOL). PAS includes different well-known models along with their uncertainties to simulate 
inter-patient variability. Blood loss can also be simulated to assess the controller's performance in a 
shock scenario. Finally, PAS includes standard disturbance profiles and metric computation to facilitate 
the evaluation of the controller's performances. This document provides the statement of need and the 
documentation to use the package. Some examples are also available in the package folder to better 
understand how to use it.


.. toctree::
   :maxdepth: 5
   :caption: Contents:

   intro
   src.python_anesthesia_simulator



