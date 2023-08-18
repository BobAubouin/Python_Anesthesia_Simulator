[![status](https://joss.theoj.org/papers/61d34ad9ef855a128509b4279e2c9325/status.svg)](https://joss.theoj.org/papers/61d34ad9ef855a128509b4279e2c9325)
[![Documentation Status](https://readthedocs.org/projects/python-anesthesia-simulator/badge/?version=latest)](https://python-anesthesia-simulator.readthedocs.io/en/latest/?badge=latest)
<img src ="https://img.shields.io/github/last-commit/BobAubouin/Python_Anesthesia_Simulator" alt="GitHub last commit"> 
# Python_Anesthesia_Simulator
The Python Anesthesia Simulator (PAS) models the effect of drugs on physiological variables during total intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrug controllers. The available drugs are Propofol, Remifentanil, and Norepinephrine, the outputs are the Bispectral index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO), and Tolerance of Laryngoscopy (TOL). PAS includes different well-known models along with their uncertainties to simulate inter-patient variability. Blood loss can also be simulated to assess the controller's performance in a shock scenario. Finally, PAS includes standard disturbance profiles and metrics computation to facilitate the evaluation of the controller's performances.

- **Documentation and examples:** https://python-anesthesia-simulator.readthedocs.io
- **Associated paper:** https://joss.theoj.org/papers/10.21105/joss.05480
## Installation
Use pip to install the package:
```python
    pip install git+https://github.com/BobAubouin/Python_Anesthesia_Simulator.git
```
The package can be imported in your python script with:
```python
    import python_anesthesia_simulator as pas
```
## Citation

To cite PAS in your work, cite this paper:
```
Aubouin-Pairault et al., (2023). PAS: a Python Anesthesia Simulator for drug control. Journal of Open Source Software, 8(88), 5480, https://doi.org/10.21105/joss.05480
```


## Guidelines
- To report a bug or request a feature please open an issue.
- To contribute, you can fork the repository and ask for a pull request.

If you want to contact me for any reason, I'm available by [mail](bob.aubouin-pairault@gipsa-lab.fr).

## Structure

    .
    ├─── src
    |   ├─── python_anesthesia_simulator           # Simulator library + metrics function
    |
    ├── paper              # markdown paper for JOSS submition
    |
    ├── tests              # files for testing the package
    |
    ├── docs               # files for generating the docs
    | 
    ├── LICENSE
    ├── pyproject.toml      # packaging file
    ├── requirements.txt
    ├── README.md
    └── .gitignore          

## License

_GNU General Public License 3.0_

## Project status
Published in the journal of Open Source Software!

## Author
Bob Aubouin--Paitault
