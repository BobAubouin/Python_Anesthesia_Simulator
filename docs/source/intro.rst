Introduction
============


Installation
------------

Set your terminal path to the PAS package and install it using pip:

.. code-block:: console

   $ pip install .

Statement of need
-----------------

Closing the loop for drug dosage during general anesthesia is a challenging task which has drawn the attention 
of the control community for more than two decades. In fact, the high need for reliability coupled with a highly 
uncertain system makes the design of a controller an arduous work. Numerous closed-loop control strategies have 
been proposed and reviewed in [Ilyas2017]_, [Copot2020]_, [Singh2022]_, and [Ghita2020]_ for instance. 


Since it is a long process to clinically test a control method for drug injection, many papers only rely on 
simulations to demonstrate the advantages of their propositions. However, two meta-studies 
[Brogi2017]_, [Puri2016]_ agree on the fact that automated 
control during anesthesia brings better stability to the physiological signals during the procedure. Since the 
methods that were first tested on simulation led to good performances in clinical situations, it could be concluded 
that simulations are realistic enough to be used as a first test.


In addition to being significantly faster than clinical tests, simulations also allow researchers to use the same 
framework to compare their methods. Up to now, many different controllers have been proposed and tested on their own 
frameworks and it is often hard to compare them properly. This is aggravated by the fact that the code of the algorithms 
is rarely open-source which makes hard the reproducibility of the results (to the author's knowledge, 
[Coman2022]_ is the only open-source controller for anesthesia). Ionescu *et al.* 
recently address this issue by providing an open-source anesthesia simulator in [Ionescu2021]_. 
If this is a first step to a FAIR (Findable, Accessible, Interoperable, Reusable) science, it can be improved. 
In fact, the measurement noise and disturbance scenarios are still not specified. Moreover, while the simulator 
could incorporate them, the level of uncertainty to take into account in the patient model is also not fixed. 
Finally, the metrics used to compare the performances must be explicit and used by all users to compare the 
proposed control methods.


With the Python Anesthesia Simulator, a full pipeline is proposed to test control methods for drug dosage in anesthesia. 
By using Python, an open-source language, we hope that everyone will be able to use our simulator. If the control 
community has been historically using Matlab, the use of the control package [Fuller2021]_ in 
PAS facilitate the transition, with Matlab compatibility functions. Moreover, from an interdisciplinary point of 
view, Python is already widely used by the data science and machine learning community. On the anesthesia topic, 
the database VitalDB [Lee2018]_ has a Python API and could be used together with PAS to develop 
data-based approaches, as already done in [Aubouin2023]_.

In addition to all the models available to link drugs to their effects, some useful functionalities are available in PAS:

- The uncertainties associated with Pharmacokinetic (PK) and Pharmacodynamic (PD) models are available to model inter-patient variability;

- Initialization of the patient simulator at the maintenance phase (around an equilibrium point) is possible.

- As this simulator also includes hemodynamics, the cardiac output can be used to actualize the PK models as proposed in [Bienert2020]_;

- A blood loss scenario is available to test the controllers in extreme conditions;

- Standard additive disturbance profiles used in the literature are available;

- Control metric computation is directly available in the package.

.. [Ilyas2017] Ilyas, M., Butt, M. F. U., Bilal, M., Mahmood, K., Khaqan, A., & Ali Riaz, R. (2017). A Review
   of Modern Control Strategies for Clinical Evaluation of Propofol Anesthesia Administration
   Employing Hypnosis Level Regulation. BioMed Research International, 2017, e7432310.
   https://doi.org/10.1155/2017/7432310

.. [Copot2020] Copot, D. (2020). Automated drug delivery in anesthesia. Academic Press. https://doi.org/10.1016/c2017-0-03401-8

.. [Singh2022] Singh, M., & Nath, G. (2022). Artificial intelligence and anesthesia: A narrative review. Saudi
   Journal of Anaesthesia, 16(1), 86–93. https://doi.org/10.4103/sja.sja_669_21

.. [Ghita2020] Ghita, M., Neckebroek, M., Muresan, C., & Copot, D. (2020). Closed-Loop Control of
   Anesthesia: Survey on Actual Trends, Challenges and Perspectives. IEEE Access, 8,
   206264–206279. https://doi.org/10.1109/ACCESS.2020.3037725

.. [Brogi2017] Brogi, E., Cyr, S., Kazan, R., Giunta, F., & Hemmerling, T. M. (2017). Clinical Performance
   and Safety of Closed-Loop Systems: A Systematic Review and Meta-analysis of Randomized
   Controlled Trials. Anesthesia & Analgesia, 124(2), 446–455. https://doi.org/10.1213/ANE.0000000000001372

.. [Puri2016] Puri, G. D., Mathew, P. J., Biswas, I., Dutta, A., Sood, J., Gombar, S., Palta, S., Tsering,
   M., Gautam, P. L., Jayant, A., Arora, I., Bajaj, V., Punia, T. S., & Singh, G. (2016).
   A Multicenter Evaluation of a Closed-Loop Anesthesia Delivery System: A Randomized
   Controlled Trial. Anesthesia & Analgesia, 122(1), 106–114. https://doi.org/10.1213/ANE.0000000000000769

.. [Coman2022] Coman, S., & Iosif, D. (2022). AnesthesiaGUIDE: A MATLAB tool to control the anesthesia.
   SN Applied Sciences, 4(1), 3. https://doi.org/10.1007/s42452-021-04885-x

.. [Ionescu2021] Ionescu, C. M., Neckebroek, M., Ghita, M., & Copot, D. (2021). An Open Source Patient
   Simulator for Design and Evaluation of Computer Based Multiple Drug Dosing Control for
   Anesthetic and Hemodynamic Variables. IEEE Access, 9, 8680–8694. https://doi.org/10.1109/ACCESS.2021.3049880

.. [Fuller2021] Fuller, S., Greiner, B., Moore, J., Murray, R., van Paassen, R., & Yorke, R. (2021). The
   Python Control Systems Library (python-control). 2021 60th IEEE Conference on Decision
   and Control (CDC), 4875–4881. https://doi.org/10.1109/CDC45484.2021.9683368

.. [Lee2018] Lee, H.-C., & Jung, C.-W. (2018). Vital Recorder—a free research tool for automatic recording
   of high-resolution time-synchronised physiological data from multiple anaesthesia devices.
   Scientific Reports, 8(1), 1527. https://doi.org/10.1038/s41598-018-20062-4

.. [Aubouin2023] Aubouin--Pairault, B., Fiacchini, M., Dang, T. (2023 June). Data-based Pharmacodynamic Modeling 
   for  BIS and Mean Artrial Pressure Prediction during General Anesthesia. ECC 2023 21st European
   Control Conference.

.. [Bienert2020] Bienert, A., Sobczyński, P., Młodawska, K., Hartmann-Sobczyńska, R., Grześkowiak, E., &
   Wiczling, P. (2020). The influence of cardiac output on propofol and fentanyl pharmacoki-
   netics and pharmacodynamics in patients undergoing abdominal aortic surgery. Journal
   of Pharmacokinetics and Pharmacodynamics, 47 (6), 583–596. https://doi.org/10.1007/s10928-020-09712-1




