# Python_Anesthesia_Simulator

This repository includes the Python Anesthesia Simulator (PAS), used to perform multi-drugs anesthesia simulation in order to evaluate the performance of a given controller. The models used are mostly a translation of the Matlab [open-source simulator](https://fr.mathworks.com/matlabcentral/fileexchange/85208-open-source-patient-simulator) proposed in  by Ionescu _et al._. Different drugs along with different well-know models are available. The output are the Bispectral Index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO), Neuro-Muscular-Blockade (NMB) and Ramsay agitation Sedation Scale (RASS). The Simulator also includes common disturbance profiles, measurement noise and an option to process Monte-Carlo simulation with intra-patient variability. Finally, the computation of the most commonly performance index used in the control community is also handle.

## Structure 

    .
    ├─── src
    |   ├─── PAS           # Simulator library + metrics function
    |   ├─── Control       # Control + Estimation library
    |
    ├── example            # example of controller test pipeline with the library 
    |   ├── ...
    ├── our_idea 	    # our control idea with EKF and MPC
    |   ├── ...
    ├── LICENSE
    ├── requirements.txt
    └── README.md


## Documentation
Available soon in the _Documentation.pdf_ file.

## License

_GNU General Public License 3.0_

## Project status
In dev, PK model ok, PD-BIS model ok, example on PID ok, working on MISO control

## Author
Bob Aubouin--Paitault
