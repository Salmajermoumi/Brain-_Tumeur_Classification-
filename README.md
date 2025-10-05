Brain Tumor Detection using Deep Learning

Une application web interactive développée avec Streamlit et TensorFlow, permettant de détecter les tumeurs cérébrales à partir d’images IRM (MRI).
Ce projet vise à assister les médecins dans le diagnostic précoce grâce à un modèle de Deep Learning entraîné sur des données médicales.

🚀 Fonctionnalités principales

🧩 Prédiction automatique à partir d’images IRM (MRI)

📊 Affichage du niveau de confiance du modèle

💬 Recommandations médicales selon le résultat

🎨 Interface utilisateur moderne et claire (CSS personnalisé)

🧠 Modèle TensorFlow intégré (BrainTumor10Epoch.h5)

🧠 Technologies utilisées
Catégorie	Outils & Technologies
Langage principal	Python
Framework Web	Streamlit
Machine Learning / Deep Learning	TensorFlow, Keras
Traitement d’images	Pillow (PIL), NumPy
Modèle utilisé	CNN (Convolutional Neural Network)
Autres outils	GitHub, Visual Studio Code
⚙️ Installation et exécution
1️⃣ Cloner le projet
git clone https://github.com/Salmajermoumi/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

2️⃣ Créer et activer un environnement virtuel
python -m venv jermoumi_venv
source jermoumi_venv/Scripts/activate    # (Windows)

3️⃣ Installer les dépendances
pip install -r requirements.txt

4️⃣ Lancer l’application
streamlit run app.py

L’application s’ouvrira automatiquement dans ton navigateur :
➡️ http://localhost:8501

🧩 Structure du projet
Brain-Tumor-Classification/
│
├── app.py                        # Application principale Streamlit
├── BrainTumor10Epoch.h5          # Modèle entraîné TensorFlow
├── requirements.txt              # Librairies nécessaires
├── images/
│   ├── app_preview.png           # Capture d’écran (à ajouter)
│   └── brain_sample.jpg          # Exemple d’image IRM
└── README.md                     # Documentation du projet

