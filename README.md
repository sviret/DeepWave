# DeepWave

## MFlight

## GWKeras (Tensorflow implementation of the DGWS repo, without the D2L dependancy)

Info available on this page: (https://sviret.web.cern.ch/Welcome.php?n=Virgo.MML)

## DGWS (obsolete)
  
### Installation
    
Le projet utilise l'incubateur pour le Machine Learning [MXNET](https://mxnet.apache.org/versions/1.9.1/) à travers le paquet [d2l](https://d2l.ai/).
Il faut tout d'abord suivre l'installation sur le site [DIVE INTO DEEP LEARNING](https://d2l.ai/chapter_installation/index.html).
Ensuite il suffit de cloner le projet dans votre *home* par *ssh* ou *https*, le repertoire **deep-gw-search** est alors créé.
Il faut rajouter le répertoire et le sous répertoire *DGWS* dans le `PYTHONPATH`. Dans votre *.bashrc* ajouter la ligne suivante `export PYTHONPATH=$PYTHONPATH:$HOME/deep-gw-search:$HOME/deep-gw-search/DGWS`. Redémarrez alors votre terminal ou faites `source ./.bashrc` puis activez l'environnement *d2l* avec la commande `conda activate d2l`.  
Pour finir, de nouvelles fonctionnalités utilisent [PyCBC](https://github.com/gwastro/pycbc), il faut lancer la commande `pip install pycbc` avec l'environnement *d2l* activé.
  
### Organisation
  
* **params**: Dossier contenant des fichiers de paramètres *.csv* pour la génération des DataSets et pour les paramètres d'entraînements
* **generators**: Dossier contenant les DataSets générés avec `gendata.py` enregistré sous format *pickle*
* **results**: Dossier contentant les fichiers resultats *pickle* enregistré à la fin d'un entraînement avec `trainCNN.py`
* **prints**: Dossier contenant différentes courbes générés à partir de fichier resultats avec `useResults.py`
* **tests**: Dossier contenant des scripts de tests du code source
* `gendata.py`: Script et Module permettant de générer des templates et du bruit ainsi que des DataSets de training et de test 
* `trainCNN.py`: Script et Module permettant d'entrainer le réseau de neurone selon plusieurs paramètres et sauvegardant les résultats
* `useResults.py`: Script et Module permettant de stocker les résultats dans une classe et de manipuler les resultats pour afficher des courbes
  
### Scripts & Modules 
  
Les trois fichiers *.py* peuvent s'utiliser classiquement en tant que module dans un script python défini par l'utilisateur ou 
bien en ligne de commande en tant que script de la manière suivante: `python3 nom_du_script.py ARGUMENTS [OPTIONS..]`.
Tous les scripts sont accompagnés d'un aide avec l'option `-h` ou `--help`.
  
#### gendata.py
  
Le script requiert une commande au choix de trois:
* `noise`: pour générer un bruit et l'afficher ainsi que sa TF et la PSD associée. Les options disponibles sont:
`-fe FE` permettant de définir la fréquence d'échantillonnage en Hz,
`--time/-t TIME` permettant de définir le temps total du template en s,
`--kindPSD/-kp {flat,analytic}` deux options pour le bruit utilisant une PSD plate (bruit blanc) ou analytique (bruit coloré).

* `template`: pour générer un template et l'afficher ainsi que sa TF. Les options disponibles en plus sont:
`-m1 M1` et `-m2 M2` les masses des deux objets en Masse Solaire,
`--kindTemplate/-kt {EM,EOB}` deux options pour le type de génération du signal, *EOB* contenant le ringdown.
Ici, l'option `--kindPSD/-kp` sert au *whitening* du signal.

* `set`: pour générer un DataSet qui sera enregistré sou format pickles dans le dossier **params**. Les options disponibles sont:
`--set/-s {train,test}` deux options de base pour générer un DataSet de training ou de test par défaut,
`--paramfile/-pf PARAMFILE` avec *PARAMFILE* le chemin relatif ou répertoire courant ou absolu vers un fichier de paramètres *.csv* dont la forme est présentée plus bas.
  
#### trainCNN.py
  
Le script requiert deux fichiers pickle de DataSets `TrainGenerator` et `TestGenerator` dans cet ordre qui peuvent être donnés via leur chemin relatif ou absolu. 
Il enregistrera des fichiers résultats dans le dossier **results**.
Les options disponibles sont:
`--SNRTest/-St SNRTEST` permettant de choisir le SNR de test qui sera utilisé pour calculer les performances durant l'entraînement,
`--verbose/-v` choix d'affichage de l'évolution de la sensibilité et de la valeur de la fonction de perte pendant l'entraînement,
`--number/-nb NUMBER` choix du nombre d'entraîments avec le même DataSet afin d'avoir plus de statistiques(attention pas encore paralléliser),
`--paramfile/-pf PARAMFILE` avec *PARAMFILE* le chemin relatif ou répertoire courant ou absolu vers un fichier de paramètres *.csv* dont la forme est présentée plus bas.
  
#### useResults.py
  
Le script requiert un `nom_etude` pour sauvergarder les courbes et au minimum un fichier pickle resultats `Resultats...`, plusieurs peuvent être fourni.
Il enregistrera des courbes de résultats dans le dossier **prints** dans un sous-dossier avec le nom spécifié.
Les options disponibles sont:
`-FAR` qui permet de choisir le taux de fausse alarme souhaité pour calculer les différentes efficacités.
`--display/-d` qui permet directement d'afficher les courbes dans des fênetres graphiques.
  
### Fichiers de paramètres

#### Fichier pour la génération de DataSet
Le fichier *.csv* de paramètres pour la génération de DataSet doit comporter 8 lignes organisées de la manière suivante:

| | | |
| :----: | :---: | :---: |
| Ttot | 1.0 |   |
| fe | 2048.0 |   |
| kindPSD | flat |   |
| mint | 10.0 | 50.0  |
| tcint | 0.75 | 0.95  |
| NbB | 5 |   |
| kindTemplate | EM | |
| kindBank | linear | |

La première colonne doit être toujours celle-ci. Les floats doivent être ecrit avec des points et non des virgules.
**kindBank** peut prendre deux valeur *linear* ou *optimal*.
  
#### Fichier pour l'entraînement
Le fichier *.csv* de paramètres pour la génération l'entraînement doit comporter 5 lignes organisées de la manière suivante:

| | | | | | |
| :----: | :---: | :---: | :---: | :---: | :---: |
| batch_size | 250 | | | | |
| lr | 0.003 | | | | |
| kindTraining | Sca | | | | |
| tabEpochs | 4 | 8 | 20 | 40 | 100 |
| tabSNR | 36 | 24 | 16 | 12 | 8 |

La première colonne doit être toujours celle-ci. Les floats doivent être ecrit avec des points et non des virgules. *tabEpochs* représente le tableau de changement de SNR et tabSNR la valeur des différents SNR entre chaque époque. Pour donner des intervalles de SNR, il suffit de remplacer l'abbréviation *Sca* de **kindTraining** par *Int* et de remplir le tableau **tabSNR** sous cette forme: | tabSNR | *intervalle 1 borne basse* | *intervalle 1 borne haute* | *intervalle 2 borne basse* | *intervalle 2 borne haute* | ... |.  

