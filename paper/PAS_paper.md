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

The Python Anesthesia Simulator (PAS) models the effect of drug on physiological variables during total intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrugs controller. The available drugs are Propofol, Remifentanil, Epinephrine and Norepinephrine and the ouputs are the Bispectral index (BIS), Mean Arterial Pressure (MAP), Cardiac Output (CO) and Tolerance of Laryngoscopy (TOL). PAS includes differents well known models along with their uncertainties in order to simulate interpatient variability. Blood loss can also be simulated to assess the controller performances on a schock scenario. Finally PAS includes standard disturbance profils and metrics computation to facilitate the evaluation of controllers performances.

# Statement of need

Closing the loop for drug dosage during general anesthesia is a challenging task which keeps the attention of the control community for more than two decades. In fact, the high need for reliability coupled with a highly uncertain system makes the design of a controller an arduous work. Thus, numerous closed-loop control strategies have been proposed, reviewed in @ilyasReviewModernControl2017, @copotAutomatedDrugDelivery2020, @singhArtificialIntelligenceAnesthesia2022 and @ghitaClosedLoopControlAnesthesia2020 for instance. 


Because it is a long term process to clinically test a control method for drug injection, a lot of papers only rely on simulations to demonstrate the advantages of their proposition. However, two meta-studies agree on the fact that automated control during anesthesia brings better stability to the physiological signals during the procedure @brogiClinicalPerformanceSafety2017, @puriMulticenterEvaluationClosedLoop2016. Since the methods were first tested on simulation and obtain then good performances in clinical situation it could be conclude that simulation are realistic enough to be used as a first test.


In addition of being way faster than clinical test, simulation also allow the researchers to used the same framework to compare their methods. Up to now, many different controllers have been proposed and tested on their own framework and it is often hard to compare them properly. This is made worse by the fact that the code of the algorithms is almost never open-source which makes hard the reproducibility of the results (to the authors knowledge @comanAnesthesiaGUIDEMATLABTool2022 is the only open-source controller for anesthesia). Ionescu *et al.* recently address this issue by providing an open-source anesthesia simulator to the community in  @ionescuOpenSourcePatient2021a. If this this a first step to a FAIR (Findable, Accessible, Interoperable, Reusable) science, it can be improved. In fact, the measurement noise and disturbance scenarios are still not specified. Moreover, while the simulator could incorporate them, the level of uncertainty to take into account in the patient model is also not fixed. Finally, the metrics used to compare the performances must be explicit and used by all to compare our controls methods.
  

With the Python Anesthesia Simulator we proposed a full pipeline to test control methods for drug dosage in anesthesia. By using *Pyhton*, an open-source language, we hope that everyone will be able to use our simulator. If the control community have been historically using MATLAB we believed that the use of python is not a big step. Moreover, the use of the control package [@fullerPythonControlSystems2021] make the transition even easier with Matlab compatibility functions. 

In addition to all the model available to link drugs to their effect some usefull functionalities are available in PAS:

- The uncertainties associated with Pharmacokinetic (PK) and Pharmacodynamic (PD) models are available to model inter-patient variability;

- Initialization of the patient simulator at maintenance phase (equilirbium point) is possible.

- As this simulator also include hemodynamics, the cardiac output can be used to actualize the PK models as proposed in @bienertInfluenceCardiacOutput2020;

- A blood loss scenario is available to test the controllers in extrem conditions;

- Standard additive disturbance profiles used in the literature are available;

-  Control metric computation is directly available in the package.

# Model information

In PAS all drugs effect are described by the well know Pharmacokinetic-Pharmacodynamic (PK-PD) model. PK model are used to describe the distribution of drug in the body and PD model describe the effect of the drug on a specific output. Our simulator includes the most famous models allong with their uncertainties with a log-normal distribution for all the parameters. Uncertainties can be activated in PK and PD part separatly.

## Pharmacokinetic

The standard way to model Pharmacokinetic of drugs is to used compartments model. Both Propofol and Remifentanil have been studied in many clinical trials and 3-comparments model is considered as the standard way to model those drugs parmacokinetics and also the way it is implemented in PAS.  Population The different population model available are listed beloow:

- For Propofol: Schnider @schniderInfluenceAgePropofol1999, @marshPharmacokineticModelDriven1991, Marsh model with modified time constant for the effect site compartment @struysComparisonPlasmaCompartment2000,  @schuttlerPopulationPharmacokineticsPropofol2000 and  @eleveldPharmacokineticPharmacodynamicModel2018.

- For Remifentanil: @mintoInfluenceAgeGender1997 and @eleveldAllometricModelRemifentanil2017.


For Norepinephrine and Epinephrine clinical trial are rarer and usually only one comparment is used to model the distribution of those drugs in blood. In PAS the model from @abboudPharmacokineticsEpinephrinePatients2009 for Epinephrine and @beloeilNorepinephrineKineticsDynamics2005 for Norepinephrine.

Several studies have shown the influence of Cardiac Output (CO) on the pharmacokinetic of Propofol [@uptonCardiacOutputDeterminant1999; @kuritaInfluenceCardiacOutput2002; @adachiDeterminantsPropofolInduction2001]. In @bienertInfluenceCardiacOutput2020 the authors proposed the assumption that the clearance rate of Propofol and Fentanil could be proportionnal to CO resulting in non-constant clearance rate. In the simulator the same assumption is made for the Propofol and extended to Remifentanil, Epinephrine and Norepinephrine clearance rates PK. It can be activated or desactivated to simulate the interaction between CO and the hypnotic system.

Blood loss is known to change the distribution of drug in the body [@johnsonInfluenceHemorrhagicShock2001; @kuritaInfluenceHemorrhagicShock2009a; @johnsonInfluenceHemorrhagicShock2003]. In fact, the reduce volume of blood will impact the PK system of the drugs. Thus, during blood loss simulation the blood volume is updated in all the PK model to represent the remaining fraction of remaining blood volume. As blood loss also strongly impact the hemodynamic system and a decrease in blood volume often leads to a decrease of, the clearance rates of the PK system will also decrease.

## Pharmacodynamic

Pharmacodynamics model describe the link between drug concentration and the observed effect on physiological variable. In PAS, the considered variable are the Bispectral Index (BIS) to characterize the hypnotic system, Mean Arterial Presure (MAP) and Cardiac Output (CO) for the hemodynamic system and Tolerance of Laryngoscopy (TOL) as an analgesia indicator. Unlike @ionescuOpenSourcePatient2021a which used Richmond Agitation Sedation Scale (RASS) to assess analgesia, TOL was choosen due to the availability of more clinical sutides on this index. Since this output does not affect the system another index could be programmed to replace it.

The standard way to model pharmacodynamics is to consider a delay between a rise in blood concentration and the appearance of the physiological effect by adding an *effect site* compartments. Then a hill cruve is used to model the relation between the effect site drug concentration and the dedicated effect.

For BIS and TOL, PAS include 3D hill curves to model the synergical effect of Propofol and Remifentanil with the value from @bouillonPharmacodynamicInteractionPropofol2004. 

For MAP and CO, the interaction between drugs have not been studied yet. Thus the effect of each drug is added to obtain the overall treatment effect:

- For Propofol the value from

- For Remifentanil

- For Epinephrine

- For Norepinephrine


# Future development
Many impovement can be imagine to develop PAS:

- Other drugs or model could be added.

- A more physiological model for hemodynamic system could be proposed to better link CO and MAP and eventually add Heart Rate as an output variable.

- The Respiratory system could be included to include new variables.

- New schock scenarios could be implemented to test our controller on diverse environment.

Input from the community is welcome, in particular to implement physiological models which is not the authors speciality. We also hope that the code of the controller tested on PAS will be released in an open-source manner.

## Conclusion
As shown in the available examples included in PAS, many functions are implemented to help furtur resarch on drug control during anesthesia. This simulator provide a full pipeline to design and test multidrugs controller on a wide variety of scenarios.

# Acknowledgements

This work has been partially supported by the LabEx PERSYVAL-Lab (ANR-11-LABX-0025-01) funded by the *French program Investissement d’avenir*. The authors also wants to thanks Mathias Réus for his help on practical Python programming and packaging.

# References
