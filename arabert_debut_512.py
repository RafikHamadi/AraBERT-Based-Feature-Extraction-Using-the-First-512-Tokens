# =============================================================================
# EMBEDDINGS AraBERT - Stratégie : 512 PREMIERS TOKENS
# =============================================================================
# POURQUOI AraBERT : modèle BERT pré-entraîné spécifiquement sur l'arabe
# POURQUOI 512 premiers : limite native de BERT, début du texte = introduction/style
# =============================================================================

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

# -----------------------------------------------------------------------------
# PARAMÈTRES MODIFIABLES LIBREMENT
# -----------------------------------------------------------------------------
MODEL_NAME = "aubmindlab/bert-base-arabertv2"  # AraBERT v2
DATA_DIR = "./Corpus"          # POURQUOI : dossier racine contenant un sous-dossier par auteur
OUTPUT_CSV = "embeddings_arabert_debut.csv"
DIM_SIZE = 768                 # POURQUOI : AraBERT produit des vecteurs de 768 dim nativement
                               # tu peux mettre 700, 2000, etc. (voir bloc REDIMENSIONNEMENT)
MAX_TOKENS = 512               # POURQUOI : limite BERT, on prend les 512 premiers

# -----------------------------------------------------------------------------
# CHARGEMENT DU MODÈLE
# -----------------------------------------------------------------------------
print(f"Chargement de {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # POURQUOI : mode évaluation (pas d'entraînement, désactive dropout)

# Préprocesseur AraBERT (normalisation arabe : alif, ya, etc.)
arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device : {device}")

# -----------------------------------------------------------------------------
# FONCTION D'EMBEDDING - 512 PREMIERS TOKENS
# -----------------------------------------------------------------------------
def get_embedding_debut(texte):
    """
    POURQUOI : on tokenize le texte, on garde les 512 premiers tokens,
    on passe dans AraBERT, on récupère [CLS] ou la moyenne des tokens.
    """
    # Préprocessing AraBERT (très important : normalise l'arabe)
    texte_clean = arabert_prep.preprocess(texte)
    
    # Tokenisation avec troncature aux 512 premiers
    inputs = tokenizer(
        texte_clean,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length"  # POURQUOI : pad à 512 pour cohérence
    )
    nb_tokens = int(inputs["attention_mask"].sum().item())
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # POURQUOI mean pooling : moyenne des tokens (mieux que [CLS] pour la représentation globale)
    last_hidden = outputs.last_hidden_state  # [1, 512, 768]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    embedding = embedding.squeeze(0).cpu().numpy()  # [768]

    return embedding, nb_tokens

# -----------------------------------------------------------------------------
# REDIMENSIONNEMENT LIBRE (700, 2000, etc.)
# -----------------------------------------------------------------------------
def redimensionner(vec, taille_cible):
    """
    POURQUOI : permet de changer librement la dimension finale.
    - Si taille_cible < 768 : on tronque (ou PCA serait mieux mais simple ici)
    - Si taille_cible > 768 : on pad avec des zéros (ou on duplique)
    """
    dim_actuelle = len(vec)
    if taille_cible == dim_actuelle:
        return vec
    elif taille_cible < dim_actuelle:
        return vec[:taille_cible]  # troncature simple
    else:
        # Padding avec zéros
        result = np.zeros(taille_cible)
        result[:dim_actuelle] = vec
        return result

# -----------------------------------------------------------------------------
# PARCOURS DU CORPUS
# -----------------------------------------------------------------------------
data = []

for locuteur in sorted(os.listdir(DATA_DIR)):
    dossier_locuteur = os.path.join(DATA_DIR, locuteur)
    if not os.path.isdir(dossier_locuteur):
        continue

    for fichier in sorted(os.listdir(dossier_locuteur)):
        if not fichier.endswith(".txt"):
            continue
        chemin = os.path.join(dossier_locuteur, fichier)

        with open(chemin, "r", encoding="utf-8") as f:
            texte = f.read().strip()

        if not texte:
            continue

        print(f"Traitement : {locuteur}/{fichier}")
        embedding, nb_tokens = get_embedding_debut(texte)
        embedding = redimensionner(embedding, DIM_SIZE)

        ligne = {
            "locuteur": locuteur,
            "fichier": fichier,
            "nb_tokens": nb_tokens,
            "texte_apercu": texte[:80].replace("\n", " ")
        }
        for i, val in enumerate(embedding):
            ligne[f"dim_{i}"] = val

        data.append(ligne)

# -----------------------------------------------------------------------------
# SAUVEGARDE
# -----------------------------------------------------------------------------
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Sauvegardé : {OUTPUT_CSV}")
print(f"Shape : {df.shape}")
print(df.head())
