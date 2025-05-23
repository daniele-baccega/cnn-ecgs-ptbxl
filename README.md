## INTRODUCTION
The 12-lead electrocardiogram (ECG) is a fundamental instrument for diagnosing cardiac abnormalities. Traditionally, 12-lead ECG are analyzed by trained medical professionals, however recent advances in Artificial Intelligence (AI) and, in particular, deep neural networks [1] have enabled methods to accurately analyze ECGs [2, 3, 4].
 

Such methods proved capable of recognizing specific patterns and abnormalities in ECG waveforms invisible to the human eye, e.g. detecting cardiac contractile dysfunction from an “apparently” normal ECG [5] or the presence of an underlying atrial fibrillation from a sinus rhythm ECG [6]. Altogether, these findings highlight the potential of an AI-based ECG analysis, with significant implications for early detection and management of different cardiac abnormalities.

This work [7] aims at assessing whether an AI is capable of identifying in a single lead cardiac abnormalities that are typically diagnosed from standard 12-lead ECGs.
The potential outcomes of this point are significant: if an AI would be able to detect cardiac abnormalities in single-lead ECGs, that would be a strong incentive towards integrating diagnostic AIs into wearable devices.
The perspective of single-lead ECGs diagnoses on wearable devices would be game-changer, as it would allow for frequent, accessible and economic screening for large masses of population for both cardiovascular and non-cardiovascular diseases. 

## REQUIREMENTS
### Docker
You need to have docker installed on your machine, for more info see this document: https://docs.docker.com/engine/installation/.

Ensure your user has the rights to run docker (without the use of sudo). To create the docker group and add your user:

Create the docker group.
```
  $ sudo groupadd docker
 ```
 
Add your user to the docker group.
```
  $ sudo usermod -aG docker $USER
```

Log out and log back in so that your group membership is re-evaluated.

### Setting up kaggle token
1. Go to the kaggle website.
2. Click on _Your profile_ button on the top right and then select _Account_.
3. Scroll down to the _API_ section and click on the _Create New Token_ button.
4. It will initiate the download of a file call kaggle.json. Save the file at a known location on your machine.
5. Then move the kaggle.json to ~/.kaggle location, if ~/.kaggle does’t exist you can create a directory in home with the same name.

## HOW TO REPRODUCE
To reproduce the results presented in the paper run:
```
./reproduce.sh
```
We tested the docker on the following GPUs: NVIDIA GeForce 1080, NVIDIA GeForce 1080ti, NVIDIA P106-100

## REFERENCES
[1] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

[2] Siontis, K. C., Noseworthy, P. A., Attia, Z. I., and Friedman, P. A. (2021). Artificial intelligence-enhanced electrocardiography in cardiovascular disease management. Nature Reviews Cardiology 18, 465–478

[3] Hong, S., Zhou, Y., Shang, J., Xiao, C., and Sun, J. (2020). Opportunities and challenges of deep learning methods for electrocardiogram data: A systematic review. Computers in Biology and Medicine 122, 103801. doi:https://doi.org/10.1016/j.compbiomed.2020.103801

[4] Huang, Y.-C., Hsu, Y.-C., Liu, Z.-Y., Lin, C.-H., Tsai, R., Chen, J.-S., et al. (2023). Artificial intelligence-enabled electrocardiographic screening for left ventricular systolic dysfunction and mortality risk prediction. Frontiers in Cardiovascular Medicine 10. doi:10.3389/fcvm.2023.1070641

[5] Attia, Z. I., Kapa, S., Lopez-Jimenez, F., McKie, P. M., Ladewig, D. J., Satam, G., et al. (2019a). Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram. Nature medicine 25, 70–74

[6] Attia, Z. I., Noseworthy, P. A., Lopez-Jimenez, F., Asirvatham, S. J., Deshmukh, A. J., Gersh, B. J., et al. (2019b). An artificial intelligence-enabled ecg algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction. The Lancet 394, 861–867

[7] Saglietto A, Baccega D, Esposito R, Anselmino M, Dusi V, Fiandrotti A and De Ferrari GM (2024) Convolutional neural network (CNN)-enabled electrocardiogram (ECG) analysis: a comparison between standard twelve-lead and single-lead setups. Front. Cardiovasc. Med. 11:1327179. doi: 10.3389/fcvm.2024.1327179

## COPYRIGHT AND LICENSE
Copyright _Andrea Saglietto, Daniele Baccega, Roberto Esposito, Matteo Anselmino, Veronica Dusi, Attilio Fiandrotti, Gaetano Maria De Ferrari_

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.
