---
title: 'PAS: A Python Anesthesia Simulator for the control community'
tags:
  - Python
  - Anesthesia
  - Drug Control
  - Drug effect modelling
  - Test pipeline
authors:
  - name: Bob Aubouin--Pairault
    orcid: 0000-0003-0029-438X
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    email: bob.aubouin-pairault@gipsa-lab.fr
  - name: Mirko Fiacchini
	orcid: 0000-0002-3601-0302
	affiliation: 1
	email: mirko.fiacchini@gipsa-lab.fr
  - name: Thao Dang
	orcid: 0000-0002-3637-1415
    affiliation: 2
    email: thao.dang@univ-grenoble-alpes.fr
affiliations:
 - name: Univ. Grenoble Alpes, CNRS, Grenoble INP, GIPSA-lab, 38000 Grenoble, France
   index: 1
 - name: Univ. Grenoble Alpes, CNRS, Grenoble INP, VERIMAG, 38000 Grenoble, France
   index: 2
date: 22 February 2022
bibliography: bibli.bib
---

# Summary

The Python Anesthesia Simulator (PAS) models the effect of drug on physiological variables during total intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrugs controller. The available drugs are Propofol, Remifentanil, Epinephrine and Norepinephrine and the ouputs are the Bispectral index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO) and Tolerance of Laryngoscopy (TOL). PAS includes differents well know models along with their uncertainties in order to simulate interpatient variability. It also includes standard disturbance profil and metrics computation to facilitate the controllers performances. Finally blood loss can also be simulated to assess the controller performances on a schock scenario.

# Statement of need

Closing the loop for drug dosage during general anesthesia is a challenging task which keeps the attention of the control community for more than two decades. In fact, the high need for reliability coupled with a highly uncertain system makes the design of a controller an arduous work. Thus, numerous closed-loop control strategies have been proposed, reviewed in @ilyasReviewModernControl2017, @copotAutomatedDrugDelivery2020, @singhArtificialIntelligenceAnesthesia2022 and @ghitaClosedLoopControlAnesthesia2020 for instance. \\

  

Because it is a long term process to clinically test a control method for drug injection, a lot of papers only rely on simulations to demonstrate the advantages of their proposition. However, two meta-studies agree on the fact that automated control during anesthesia brings better stability to the physiological signals during the procedure @brogiClinicalPerformanceSafety2017@, @puriMulticenterEvaluationClosedLoop2016@. Since the methods were first tested on simulation and obtain then good performances in clinical situation it could be conclude that simulation are realistic enough to be used as a first test.\\

  

In addition of being way faster than clinical test, simulation also allow the researchers to used the same framework to compare their methods. Up to now, many different controllers have been proposed and tested on their own framework and it is often hard to compare them properly. This is made worse by the fact that the code of the algorithms is almost never open-source which makes hard the reproducibility of the results (to the authors knowledge @comanAnesthesiaGUIDEMATLABTool2022 is the only open-source controller for anesthesia). Ionescu *et al.* recently address this issue by providing an open-source anesthesia simulator to the community in  @ionescuOpenSourcePatient2021a. If this this a first step to a FAIR [¹] science, it can be improved. In fact, the measurement noise and disturbance scenarios are still not specified. Moreover, while the simulator could incorporate them, the level of uncertainty to take into account in the patient model is also not fixed. Finally, the metrics used to compare the performances must be explicit and used by all to compare our controls methods.\\

[¹]: Findable, Accessible, Interoperable, Reusable
  

  

With the Python Anesthesia Simulator we proposed a full pipeline to test control methods for drug dosage in anesthesia. By using *Pyhton*, an open-source language, we hope that everyone will be able to use our simulator. In addition to all the model available to link Drugs to their effect some functionalities are added:

- More population models are available to describe the pharmacokinetics of Propofol and Remifentanil, especially the ones proposed by Eleveld and co-authors in @eleveldPharmacokineticPharmacodynamicModel2018}and @eleveldAllometricModelRemifentanil2017;
- The uncertainties associated with Propofol and Remifentanil PK-PD models are available to model inter-patient variability;
- As this simulator also include hemodynamics, the cardiac output can be used to actualize the PK models as proposed in @bienertInfluenceCardiacOutput2020;
- A blood loss scenario is to test the controllers in extrem conditions;
- Standard additive disturbance profiles used in the literature are available;
-  Metrics computation is directly available in the package.


# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References