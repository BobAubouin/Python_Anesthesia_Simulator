# Python_Anesthesia_Simulator
The Python Anesthesia Simulator (PAS) models the effect of drug on physiological variables during total intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrugs controller. The available drugs are Propofol, Remifentanil, Epinephrine and Norepinephrine and the ouputs are the Bispectral index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO) and Tolerance of Laryngoscopy (TOL). PAS includes differents well known models along with their uncertainties in order to simulate interpatient variability. Blood loss can also be simulated to assess the controller performances on a schock scenario. Finally PAS includes standard disturbance profils and metrics computation to facilitate the evaluation of controllers performances.

## Structure 

    .
    ├─── src
    |   ├─── python_anesthesia_simulator           # Simulator library + metrics function
    |
    ├── example            # example of controller test pipeline with the library 
    ├── ...
    ├── paper              # markdown paper for JOSS submition
    ├── ...
    ├── LICENSE
    ├── pyproject.toml      # packaging file
    ├── requirements.txt
    ├── README.md
    └── .gitignore          


## Documentation
Available soon in the _Documentation.pdf_ file.

## License

_GNU General Public License 3.0_

## Project status
Preparing for JOSS submition

## Author
Bob Aubouin--Paitault
