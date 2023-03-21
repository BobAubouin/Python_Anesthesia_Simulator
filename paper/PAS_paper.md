---
title: 'PAS: a Python Anesthesia Simulator for drug control'
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

The Python Anesthesia Simulator (PAS) models the effect of drugs on physiological variables during total intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrug controllers. The available drugs are Propofol, Remifentanil, and Norepinephrine, the outputs are the Bispectral index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO), and Tolerance of Laryngoscopy (TOL). PAS includes different well-known models along with their uncertainties to simulate inter-patient variability. Blood loss can also be simulated to assess the controller performance in a shock scenario. Finally, PAS includes standard disturbance profiles and metrics computation to facilitate the evaluation of controllers performances. The statement of need of this package is first discussed, then some model informations are provided to the reader. The future developments of the package are discussed at the end.

# Statement of need

Closing the loop for drug dosage during general anesthesia is a challenging task which keeps the attention of the control community for more than two decades. In fact, the high need for reliability coupled with a highly uncertain system makes the design of a controller an arduous work. Thus, numerous closed-loop control strategies have been proposed and reviewed in @ilyasReviewModernControl2017, @copotAutomatedDrugDelivery2020, @singhArtificialIntelligenceAnesthesia2022, and @ghitaClosedLoopControlAnesthesia2020 for instance. 


Because it is a long-term process to clinically test a control method for drug injection, a lot of papers only rely on simulations to demonstrate the advantages of their proposition. However, two meta-studies agree on the fact that automated control during anesthesia brings better stability to the physiological signals during the procedure @brogiClinicalPerformanceSafety2017, @puriMulticenterEvaluationClosedLoop2016. Since the methods were first tested on simulation and obtained then good performances in clinical situations it could be concluded that simulations are realistic enough to be used as a first test.


In addition to being way faster than clinical tests, simulations also allow researchers to use the same framework to compare their methods. Up to now, many different controllers have been proposed and tested on their own framework and it is often hard to compare them properly. This is made worse by the fact that the code of the algorithms is rarely open-source which makes hard the reproducibility of the results (to the author's knowledge @comanAnesthesiaGUIDEMATLABTool2022 is the only open-source controller for anesthesia). Ionescu *et al.* recently address this issue by providing an open-source anesthesia simulator to the community in @ionescuOpenSourcePatient2021a. If this is a first step to a FAIR (Findable, Accessible, Interoperable, Reusable) science, it can be improved. In fact, the measurement noise and disturbance scenarios are still not specified. Moreover, while the simulator could incorporate them, the level of uncertainty to take into account in the patient model is also not fixed. Finally, the metrics used to compare the performances must be explicit and used by all to compare our control methods.
  

With the Python Anesthesia Simulator, a full pipeline is proposed to test control methods for drug dosage in anesthesia. By using *Pyhton*, an open-source language, we hope that everyone will be able to use our simulator. If the control community has been historically using MATLAB, the use of the control package [@fullerPythonControlSystems2021] in PAS helps to transition, with Matlab compatibility functions. Moreover, from an interdisciplinary point of view, Python is already widely used by the data science and machine learning community. On the anesthesia topic, the database *VitalDB* [@leeVitalRecorderFree2018] has a Python API and could be used together with PAS to propose data-based approaches.

In addition to all the models available to link drugs to their effect some useful functionalities are available in PAS:

- The uncertainties associated with Pharmacokinetic (PK) and Pharmacodynamic (PD) models are available to model inter-patient variability;

- Initialization of the patient simulator at the maintenance phase (equilibrium point) is possible.

- As this simulator also includes hemodynamics, the cardiac output can be used to actualize the PK models as proposed in @bienertInfluenceCardiacOutput2020;

- A blood loss scenario is available to test the controllers in extreme conditions;

- Standard additive disturbance profiles used in the literature are available;

-  Control metric computation is directly available in the package.

# Model information

In PAS all drugs effect are described by the well know Pharmacokinetic-Pharmacodynamic (PK-PD) model. PK models are used to describe the distribution of drugs in the body and PD models describe the effect of the drug on a specific output. Our simulator includes the most famous models along with their uncertainties with a log-normal distribution for all the parameters. Uncertainties can be activated in PK and PD parts separately.

## Pharmacokinetic

The standard way to model Pharmacokinetic of drugs is to used compartments model. Both Propofol and Remifentanil have been studied in many clinical trials and 3-comparments model is considered as the standard way to model those drugs parmacokinetics and also the way it is implemented in PAS.  Population The different population model available are listed beloow:

- For Propofol: @schniderInfluenceAgePropofol1999, @marshPharmacokineticModelDriven1991, Marsh model with modified time constant for the effect site compartment [@struysComparisonPlasmaCompartment2000],  @schuttlerPopulationPharmacokineticsPropofol2000 and  @eleveldPharmacokineticPharmacodynamicModel2018.

- For Remifentanil: @mintoInfluenceAgeGender1997 and @eleveldAllometricModelRemifentanil2017.


For Norepinephrine clinical trial are rarer and usually only one comparment is used to model the distribution of those drugs in blood. In PAS the model from @beloeilNorepinephrineKineticsDynamics2005 is programmed.

Several studies have shown the influence of Cardiac Output (CO) on the pharmacokinetic of Propofol [@uptonCardiacOutputDeterminant1999; @kuritaInfluenceCardiacOutput2002; @adachiDeterminantsPropofolInduction2001]. In @bienertInfluenceCardiacOutput2020 the authors proposed the assumption that the clearance rate of Propofol and Fentanil could be proportionnal to CO resulting in non-constant clearance rate. In the simulator the same assumption is made for the Propofol and extended to Remifentanil, Epinephrine and Norepinephrine clearance rates PK. It can be activated or desactivated to simulate the interaction between CO and the hypnotic system.

Blood loss is known to change the distribution of drug in the body [@johnsonInfluenceHemorrhagicShock2001; @kuritaInfluenceHemorrhagicShock2009a; @johnsonInfluenceHemorrhagicShock2003]. In fact, the reduce volume of blood will impact the PK system of the drugs. Thus, during blood loss simulation the blood volume is updated in all the PK model to represent the remaining fraction of remaining blood volume. As blood loss also strongly impact the hemodynamic system and a decrease in blood volume often leads to a decrease of, the clearance rates of the PK system will also decrease.

## Pharmacodynamic

Pharmacodynamics model describe the link between drug concentration and the observed effect on physiological variable. In PAS, the considered variable are the Bispectral Index (BIS) to characterize the hypnotic system, Mean Arterial Presure (MAP) and Cardiac Output (CO) for the hemodynamic system and Tolerance of Laryngoscopy (TOL) as an analgesia indicator. Unlike @ionescuOpenSourcePatient2021a which used Richmond Agitation Sedation Scale (RASS) to assess analgesia, TOL was choosen due to the availability of more clinical sutides on this index. Since this output does not affect the system another index could be programmed to replace it.

The standard way to model pharmacodynamics is to consider a delay between a rise in blood concentration and the appearance of the physiological effect by adding an *effect site* compartments. Then a hill cruve is used to model the relation between the effect site drug concentration and the dedicated effect.

For BIS and TOL, PAS include 3D hill curves to model the synergical effect of Propofol and Remifentanil with the value from @bouillonPharmacodynamicInteractionPropofol2004. 

For MAP and CO, the interaction between drugs have not been studied yet. Thus the effect of each drug is added to obtain the overall treatment effect:

- For Propofol the value from where used for MAP. For CO, we chosed to not consider the effect of propofol sinced they are negligible [@dewitEffectPropofolHaemodynamics2016]. However the interaction programmed in the simulator and the value can be changed to model this interaction.

- For Remifentanil 

- For Norepinephrine the value from @beloeilNorepinephrineKineticsDynamics2005 are used for MAP interaction in accordance to the PK model. For the impact on Cardiac Output we extrapolate value from @hamzaouiEarlyAdministrationNorepinephrine2010.


# Conclusion and Future development
As shown in the available examples included in PAS, many functions are implemented to help further research on drug control during anesthesia. This package provides a full pipeline to design and test multidrug controllers on a wide variety of scenarios. In the future many improvements can be imagined to develop PAS:

- Other drugs or models could be added;

- Neuromuscular blockade system could be implemented as this is an important component of anesthesia paradigm;

- A more physiological model for the hemodynamic system could be proposed to better link CO and MAP and eventually add Heart Rate as an output variable;

- The Respiratory system could be included to include new variables;

- New shock scenarios could be implemented to test our controller in a diverse environment.

Input from the community is welcome, in particular, to implement a physiological model which is not the author's specialty. We also hope that the code of controllers tested on PAS will be released in an open-source manner.


# Acknowledgements

This work has been partially supported by the LabEx PERSYVAL-Lab (ANR-11-LABX-0025-01) funded by the *French program Investissement d’avenir*. The authors also wants to thanks Mathias Réus for his help on practical Python programming and packaging.

# References
