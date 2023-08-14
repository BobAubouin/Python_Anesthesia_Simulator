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

The Python Anesthesia Simulator (PAS) models the effect of drugs on physiological variables during total intravenous anesthesia. It is particularly dedicated to the control community, to be used as a benchmark for the design of multidrug controllers. The available drugs are propofol, remifentanil, and norepinephrine, the outputs are the bispectral index (BIS), mean arterial pressure (MAP), cardiac output (CO), and tolerance of laryngoscopy (TOL). PAS includes different well-known models along with their uncertainties to simulate inter-patient variability. Blood loss can also be simulated to assess the controller's performance in a shock scenario. Finally, PAS includes standard disturbance profiles and metric computation to facilitate the evaluation of the controller's performance. The statement of need of this package is first discussed; then some pieces of information about the programmed models are provided to the reader. The future developments of the package are discussed at the end.

# Statement of need

Closing the loop for drug dosage during general anesthesia is a challenging task that has attracted the attention of the control community for more than two decades. In fact, the high need for reliability coupled with a highly uncertain system makes the design of a controller arduous work. Numerous closed-loop control strategies have been proposed and reviewed in @ilyasReviewModernControl2017, @copotAutomatedDrugDelivery2020, @singhArtificialIntelligenceAnesthesia2022, and @ghitaClosedLoopControlAnesthesia2020 for instance. 


Since it is a long process to clinically test a control method for drug injection, many papers only rely on simulations to demonstrate the advantages of their suggestions. However, two meta-studies @brogiClinicalPerformanceSafety2017, @puriMulticenterEvaluationClosedLoop2016 agree on the fact that automated control during anesthesia brings better stability to the physiological signals during the procedure. Since the methods that were first tested on simulation led to good performances in clinical situations, it could be concluded that simulations are realistic enough to be used as a first test.


In addition to being significantly faster than clinical tests, simulations also allow researchers to use the same framework to compare their methods. Up to now, many controllers have been proposed and tested on their own frameworks, and it is often hard to compare them properly. This is aggravated by the fact that the code of the algorithms is rarely open-source which makes hard the reproducibility of the results (to the author's knowledge, @comanAnesthesiaGUIDEMATLABTool2022 is the only open-source controller for anesthesia). Ionescu *et al.* recently address this issue by providing an open-source anesthesia simulator in @ionescuOpenSourcePatient2021a. If this is the first step to a FAIR (Findable, Accessible, Interoperable, Reusable) science, it can be improved. In fact, the measurement noise and disturbance scenarios are not yet specified. Moreover, while the simulator could incorporate them, the level of uncertainty to take into account in the patient model is also not fixed. Finally, the metrics used to compare the performance must be explicit and used by all users to compare the proposed control methods.
  

With the Python Anesthesia Simulator, a full pipeline is proposed to test control methods for drug dosage in anesthesia. By using Python, an open-source language, we hope that everyone will be able to use our simulator. If the control community has historically used Matlab, the use of the control package [@fullerPythonControlSystems2021] in PAS facilitates the transition, with Matlab compatibility functions. Moreover, from an interdisciplinary point of view, Python is already widely used by the data science and machine learning community. On the anesthesia topic, the database VitalDB [@leeVitalRecorderFree2018] has a Python API and could be used together with PAS to develop data-based approaches, as already done in @aubouinpairaultDatabasedPharmacodynamicModeling2023.

In addition to all the models available to link drugs to their effects, some useful functionalities are available in PAS:

- The uncertainties associated with Pharmacokinetic (PK) and Pharmacodynamic (PD) models are available to model inter-patient variability;

- Initialization of the patient simulator in the maintenance phase (around an equilibrium point) is possible.

- As this simulator also includes hemodynamics, the cardiac output can be used to actualize the PK models as proposed in @bienertInfluenceCardiacOutput2020;

- A blood loss scenario is available to test the controllers in extreme conditions;

- Standard additive disturbance profiles used in the literature are available;

- Control metric computation is directly available in the package.

# Model information

In PAS all drug effects are described by the well-know Pharmacokinetic-Pharmacodynamic (PK-PD) models. PK models are used to describe the distribution of drugs in the body, and PD models describe the effect of the drug on a specific output. Our simulator includes the most commonly employed models along with their uncertainties following a log-normal distribution for all the parameters. Uncertainties can be activated in the PK and PD parts separately.

## Pharmacokinetics

The standard way to model the Pharmacokinetics of drugs is to use a compartment model. Both propofol and remifentanil have been studied in many clinical trials and the 3-compartment model is considered as the standard way to model those drugs' pharmacokinetics, and it is also implemented in PAS.  The different population models available are listed below:

- For propofol: @schniderInfluenceAgePropofol1999, @marshPharmacokineticModelDriven1991, @struysComparisonPlasmaCompartment2000 (Marsh model with the modified time constant for the effect site compartment), @schuttlerPopulationPharmacokineticsPropofol2000 and @eleveldPharmacokineticPharmacodynamicModel2018.

- For remifentanil: @mintoInfluenceAgeGender1997 and @eleveldAllometricModelRemifentanil2017.


Norepinephrine clinical trials are rarer and usually, only one compartment is used to model the distribution of this drug in blood. In PAS the model from @beloeilNorepinephrineKineticsDynamics2005 is programmed.

Several studies have shown the influence of cardiac output (CO) on the pharmacokinetics of propofol [@uptonCardiacOutputDeterminant1999; @kuritaInfluenceCardiacOutput2002; @adachiDeterminantsPropofolInduction2001]. In @bienertInfluenceCardiacOutput2020 the authors proposed the assumption that the clearance rate of propofol and fentanil could be proportional to CO resulting in a non-constant clearance rate. In the simulator, the same assumption is made for propofol and extended to remifentanil and remifentanil clearance rates PK. It can be activated or deactivated to simulate the interaction between CO and the PK systems.

Blood loss is known to change the distribution of drugs in the body [@johnsonInfluenceHemorrhagicShock2001; @kuritaInfluenceHemorrhagicShock2009a; @johnsonInfluenceHemorrhagicShock2003]. In fact, the reduced volume of blood will affect the PK system of the drugs. Thus, during a blood loss simulation the blood volume is updated in all the PK models to represent the remaining fraction of the remaining blood volume. As blood loss also strongly impacts the hemodynamic system leading to a decrease in the CO, the clearance rates of the PK system will also decrease.

## Pharmacodynamics

Pharmacodynamics models describe the link between drug concentrations and the observed effect on physiological variables. In PAS, the considered variables are bispectral index (BIS) to characterize the hypnotic system, mean arterial pressure (MAP) and cardiac output (CO) for the hemodynamic system, and tolerance of laryngoscopy (TOL) as an analgesia indicator. Unlike @ionescuOpenSourcePatient2021a which uses Richmond agitation sedation scale (RASS) to assess analgesia, TOL is chosen due to the availability of more clinical studies on this index. Since this output does not affect the system, another index could be programmed to replace it.

The standard way to model pharmacodynamics is to consider a delay between a rise in blood concentration and the appearance of the physiological effect by adding an effect site compartments. A Hill curve was then used to model the relation between the effect site drug concentration and the dedicated effect.

For BIS and TOL, PAS includes respectively a surface-response model and a hierarchical model to represent the synergic effect of propofol and remifentanil with values from @bouillonPharmacodynamicInteractionPropofol2004. 

For MAP and CO, the interaction between drugs has not been studied yet. Thus, the effect of each drug is added to obtain the overall treatment effect:

- For propofol values from @jeleazcovPharmacodynamicResponseModelling2015 are used for MAP. For CO, experimental values from @fairfieldHaemodynamicEffectsPropofol1991.

- For remifentanil only studies in infants were found, the parameters from @standingPharmacokineticPharmacodynamicModeling2010b are used for the MAP effect, and the experimental results from @chanavazHaemodynamicEffectsRemifentanil2005 for the CO effect.

- For remifentanil values from @beloeilNorepinephrineKineticsDynamics2005 are used for MAP interaction in accordance with the PK model. For the impact on CO, we extrapolate values from @monnetNorepinephrineIncreasesCardiac2011.

Note that for the effect of all drugs on CO, there is no study proposing a Hill curve. Thus, the Hill curve parameters are computed to match experimental results.

## Blood loss
In addition to the effect of blood loss in the PK systems, the crude assumption that MAP and CO are proportional to the blood volume was made in the simulator. The transient behavior of bleeding and transfusion does not verify this assumption, however the steady-state experimental values do agree with it [@rinehartEvaluationNovelClosedloop2011]. A more complex hemodynamic model should be integrated to obtain better results. The simulator also takes into account the fact that the BIS pharmacodynamics depends on bleeding [@kuritaInfluenceHemorrhagicShock2009a] leading to a deeper hypnosis state. 


# Conclusion and Future development
As shown in the available examples included in PAS, many functions are implemented to support further research on drug control during anesthesia. This package provides a full pipeline for designing and testing multidrug controllers in a wide variety of scenarios. In the future many improvements can be imagined to develop PAS:

- Other drugs or models could be added;

- Neuromuscular blockade system could be implemented as this is an important component of the anesthesia paradigm;

- A more physiological model for the hemodynamic system could be proposed to better link CO and MAP and eventually add Heart Rate as an output variable;

- The respiratory system could be considered to include new variables;

- New shock scenarios could be implemented to test our controller in a diverse environment.

Feedback and input from the community are welcome, in particular, to implement a physiological model which is not our domain of expertise. We also hope that the code of controllers tested on PAS will be released in an open-source manner.


# Acknowledgements

This work has been partially supported by the LabEx PERSYVAL-Lab (ANR-11-LABX-0025-01) funded by the *French program Investissement d’avenir* and the *French-Japanese ANR-JST CyphAI* project. The authors also thank Mathias Réus for his help with practical Python programming and packaging.

# References
