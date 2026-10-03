# Détection de feu et de fumée par Online Knowledge Distillation (Deep Mutual Learning)
 
Classification d'images de **feu** et de **fumée** avec deux réseaux légers entraînés **en même temps**
et qui s'enseignent mutuellement : c'est le principe du **Deep Mutual Learning (DML)**,
une forme d'*Online Knowledge Distillation*.
 
| Modèle | Backbone | Paramètres de tête |
|--------|----------|--------------------|
| **Model A** | MobileNetV2 (pré-entraîné ImageNet) | 1280 → 512 → 256 → classes |
| **Model B** | EfficientNet-B0 (pré-entraîné ImageNet) | 1280 → 640 → 320 → classes |
 
## Jeu de données
 
**Smoke-Fire-Detection-YOLO** par *sayedgamal99* (Kaggle) :
[https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)
 
Jeu d'images de feu et de fumée annoté au **format YOLO** (un fichier `.txt` par image, une ligne par objet :
`classe x_centre y_centre largeur hauteur`). Les licences et conditions d'utilisation sont indiquées sur la page Kaggle du dataset.
 
Pour ce projet, chaque image est ramenée à **une seule étiquette** (classe majoritaire dans son fichier de labels)
afin de faire de la classification d'image.
 
| Split | Images |
|-------|--------|
| Train | 15 068 |
| Validation | 3 229 |
| Test | 3 230 |
 
Structure attendue (celle du dataset Kaggle) :
 
```
smoke-fire-detection-yolo/
└── data/
    ├── train/   (images + labels .txt)
    ├── val/
    └── test/
```
 
### Récupérer le dataset
 
**Sur Kaggle** : dans le notebook, *Add Input* puis rechercher `smoke-fire-detection-yolo`.
Le chemin utilisé est `/kaggle/input/datasets/sayedgamal99/smoke-fire-detection-yolo`.
 
**En local**, avec `kagglehub` :
 
```python
import kagglehub
path = kagglehub.dataset_download("sayedgamal99/smoke-fire-detection-yolo")
print(path)   # à copier dans DATASET_ROOT du notebook
```
 
ou avec l'API Kaggle (après avoir configuré `kaggle.json`) :
 
```bash
kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo -p data/ --unzip
```
 
> Le dataset n'est **pas** inclus dans ce dépôt (trop volumineux) : merci de le télécharger depuis Kaggle
> et de respecter la licence de son auteur.
 
## Méthode
 
1. **Transfer learning** : deux backbones ImageNet avec une tête de classification personnalisée.
2. **Deep Mutual Learning** : à chaque batch, les deux modèles prédisent, puis chacun apprend
   - de la vérité terrain (Cross-Entropy avec label smoothing 0.1),
   - de l'autre modèle (divergence KL sur les sorties adoucies par une température).
   `loss = (1 − λ) · CE + λ · KL`, avec **T = 4.0** et **λ = 0.7**.
3. **Optimisation** : AdamW (lr 3e-4, weight decay 1e-4), CosineAnnealingWarmRestarts,
   mixed precision (AMP), gradient clipping à 1.0, 40 époques, batch 64.
4. **Inférence** : prédiction par image, fenêtre glissante + NMS pour localiser grossièrement les zones,
   verdict final par **ensemble** (moyenne des probabilités de A et B).
5. **Évaluation** : matrices de confusion, F1 par classe, Kappa de Cohen, MCC, courbes d'apprentissage.
## Résultats (jeu de test, 3 230 images)
 
| Modèle | Accuracy | Macro F1 | Kappa | MCC |
|--------|----------|----------|-------|-----|
| Model A (MobileNetV2) | 94.67 % | 0.9121 | 0.8242 | 0.8245 |
| Model B (EfficientNet-B0) | 94.71 % | 0.9133 | 0.8267 | 0.8273 |
| **Ensemble (A + B) / 2** | **94.77 %** | **0.9138** | **0.8276** | **0.8280** |
 
F1 par classe (ensemble) : Fire **0.968**, Smoke **0.860**.
 
## Structure du dépôt
 
```
smoke-fire-dml/
├── notebooks/
│   └── smoke_fire_detection.ipynb
├── models/
│   ├── README.md
│   ├── modelA.pkl        # à ajouter (voir ci-dessous)
│   └── modelB.pkl
├── requirements.txt
├── .gitignore
└── README.md
```
 
## Exécution
 
**Sur Kaggle (recommandé)** : importer le notebook, ajouter le dataset
`sayedgamal99/smoke-fire-detection-yolo`, activer le GPU, puis *Run All*.
L'entraînement dure environ 75 minutes sur un GPU T4 (≈ 112 s par époque).
 
**En local** :
 
```bash
git clone https://github.com/<ton-username>/smoke-fire-dml.git
cd smoke-fire-dml
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pip install kagglehub          # pour télécharger le dataset
jupyter notebook notebooks/smoke_fire_detection.ipynb
```
 
Adapter `DATASET_ROOT` dans le notebook vers le dossier local du dataset.
 
## Limites et pistes d'amélioration
 
- Le dataset est **déséquilibré** (classe majoritaire à ~82 % du test) : la classe Smoke est plus difficile (F1 ≈ 0.86).
- Le projet fait de la **classification d'image** ; la localisation (boîtes) vient d'une fenêtre glissante
  et reste approximative. Un vrai détecteur (YOLO) donnerait de meilleures boîtes.
- Pistes : pondération des classes, augmentation de données ciblée, ajout d'un 3e modèle, export ONNX pour le temps réel.
## Remerciements
 
- Dataset : [sayedgamal99/smoke-fire-detection-yolo](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) sur Kaggle.
- Deep Mutual Learning : Zhang et al., *Deep Mutual Learning*, CVPR 2018.
## Auteure
 
Rkia, ENSIAS, Université Mohammed V de Rabat
