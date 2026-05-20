# Octo Fine-Tuning sur ALOHA

## Environnement

Le projet a été réalisé avec :

* Python 3.10.11
* Le fichier `requirements.txt` contient les dépendances qui ont personnellement fonctionné pour ce projet.

Installation des dépendances qui marche pour moi:

```bash
pip install -r requirements.txt
```

---

## Dépôts nécessaires

Cloner les deux repositories suivants :

### Octo

[Octo GitHub](https://github.com/octo-models/octo)

### ACT

[ACT GitHub](https://github.com/tonyzhaozh/act)

---

## Dataset

Dataset utilisé :

[ALOHA Dataset Drive](https://drive.google.com/drive/folders/1sc-E4QYW7A0o23m1u2VWNGVq5smAsfCo)

Créer un dossier :

```txt
aloha_dataset
```

Puis placer tous les fichiers `.hdf5` dedans.

---

## Conversion du dataset vers le format RLDS

Le fichier :

```txt
aloha_rlds_dataset.py
```

sert à convertir les fichiers `.hdf5` au format RLDS compatible avec Octo.

Le fichier doit être placé dans :

```txt
aloha_rlds_dataset/
```

Lancer ensuite :

```bash
python -m aloha_rlds_dataset.aloha_rlds_dataset
```

Le dataset converti sera généré dans :

```txt
C:\Users\<USER>\tensorflow_datasets
```

---

## Fine-Tuning du modèle

Le fine-tuning est réalisé avec :

```txt
02_finetune_new_observation_action.py
```

Commande utilisée :

```bash
python examples/02_finetune_new_observation_action.py ^
--pretrained_path=hf://rail-berkeley/octo-small-1.5 ^
--data_dir="C:\Users\nkeungueu\tensorflow_datasets" ^
--save_dir="C:\Users\nkeungueu\Desktop\atelier_ia\octo\checkpoints\aloha_finetuned"
```

Le modèle pré-entraîné utilisé est :

[Octo Small 1.5 HuggingFace](https://huggingface.co/rail-berkeley/octo-small-1.5?utm_source=chatgpt.com)

---

## Simulation 3D

Le fichier :

```txt
run_octo_aloha_3d.py
```

permet :

* de charger le modèle fine-tuné ;
* de lancer la simulation MuJoCo ;
* d’afficher le robot ALOHA en 3D ;
* et d’enregistrer une vidéo de la simulation.

---

## Résultats

Le fichier :

```txt
resultats.txt
```

contient les résultats texte ainsi que les logs affichés pendant l’entraînement (vidéo résultante : simulatiom_octo.mp4).

Les résultats obtenus restent relativement mauvais, principalement à cause du très faible nombre d’epochs utilisé pour le fine-tuning (10 steps uniquement).

Même avec ce faible nombre d’epochs, l’entraînement a déjà pris environ 1h20 sur CPU sous Windows.

Le robot effectue donc des mouvements instables et le cube peut parfois sembler se téléporter ou réagir de manière incohérente dans la simulation.

## Modèle fine-tuné

Le modèle fine-tuné généré pendant ce projet est disponible ici :

[Télécharger le modèle fine-tuné Octo ALOHA](https://drive.google.com/drive/folders/1DZYL2Qi8Bl2qhzVMqtQwAQhoIr4KjmYv)

Le dossier contient les checkpoints générés après le fine-tuning du modèle Octo sur le dataset ALOHA.

