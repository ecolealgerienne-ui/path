# 📝 CODE CORRIGÉ — train_hovernet_family.py (2025-12-24)

**Fichier:** `scripts/training/train_hovernet_family.py`

---

## ✂️ MODIFICATION 1: Ajouter argument --lambda_magnitude

**Trouver les arguments de loss (lignes 346-353) et AJOUTER:**

```python
    # Options de loss weighting
    parser.add_argument('--lambda_np', type=float, default=1.0,
                       help='Poids loss NP (segmentation)')
    parser.add_argument('--lambda_hv', type=float, default=2.0,
                       help='Poids loss HV (séparation instances)')
    parser.add_argument('--lambda_nt', type=float, default=1.0,
                       help='Poids loss NT (classification)')
    # ↓↓↓ NOUVELLE LIGNE À AJOUTER ↓↓↓
    parser.add_argument('--lambda_magnitude', type=float, default=5.0,
                       help='Poids magnitude loss (Expert: 5.0 pour forcer gradients forts)')
    # ↑↑↑ FIN NOUVELLE LIGNE ↑↑↑
    parser.add_argument('--adaptive_loss', action='store_true',
                       help='Utiliser Uncertainty Weighting (poids appris)')
```

**Résultat attendu après modification:**
```python
    parser.add_argument('--lambda_np', type=float, default=1.0,
                       help='Poids loss NP (segmentation)')
    parser.add_argument('--lambda_hv', type=float, default=2.0,
                       help='Poids loss HV (séparation instances)')
    parser.add_argument('--lambda_nt', type=float, default=1.0,
                       help='Poids loss NT (classification)')
    parser.add_argument('--lambda_magnitude', type=float, default=5.0,
                       help='Poids magnitude loss (Expert: 5.0 pour forcer gradients forts)')
    parser.add_argument('--adaptive_loss', action='store_true',
                       help='Utiliser Uncertainty Weighting (poids appris)')
```

---

## ✂️ MODIFICATION 2: Passer lambda_magnitude à HoVerNetLoss

**Trouver la création du criterion (lignes 408-413) et MODIFIER:**

**AVANT:**
```python
    # Loss et optimizer
    criterion = HoVerNetLoss(
        lambda_np=args.lambda_np,
        lambda_hv=args.lambda_hv,
        lambda_nt=args.lambda_nt,
        adaptive=args.adaptive_loss,
    )
```

**APRÈS:**
```python
    # Loss et optimizer
    criterion = HoVerNetLoss(
        lambda_np=args.lambda_np,
        lambda_hv=args.lambda_hv,
        lambda_nt=args.lambda_nt,
        lambda_magnitude=args.lambda_magnitude,  # ← NOUVELLE LIGNE
        adaptive=args.adaptive_loss,
    )
```

---

## ✂️ MODIFICATION 3: Afficher lambda_magnitude dans les logs

**Trouver l'affichage de la configuration loss (lignes 415-420) et MODIFIER:**

**AVANT:**
```python
    # Afficher configuration loss
    if args.adaptive_loss:
        print(f"  Loss: Uncertainty Weighting (poids appris)")
        criterion.to(device)  # Les paramètres log_var sont sur le device
    else:
        print(f"  Loss: Poids fixes (NP={args.lambda_np}, HV={args.lambda_hv}, NT={args.lambda_nt})")
```

**APRÈS:**
```python
    # Afficher configuration loss
    if args.adaptive_loss:
        print(f"  Loss: Uncertainty Weighting (poids appris)")
        print(f"        Magnitude weight: {args.lambda_magnitude} (fixed)")  # ← NOUVELLE LIGNE
        criterion.to(device)  # Les paramètres log_var sont sur le device
    else:
        print(f"  Loss: Poids fixes (NP={args.lambda_np}, HV={args.lambda_hv}, NT={args.lambda_nt}, Magnitude={args.lambda_magnitude})")
        #                                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ← AJOUT
```

---

## 📝 Checklist Application

### Étape 1: Backup (IMPORTANT)

```bash
# Sauvegarder l'ancienne version
cp scripts/training/train_hovernet_family.py scripts/training/train_hovernet_family.py.backup_before_expert_fix
```

### Étape 2: Appliquer les 3 Modifications

- [ ] **Modification 1:** Ajouter argument `--lambda_magnitude` (après ligne 351)
- [ ] **Modification 2:** Passer `lambda_magnitude=args.lambda_magnitude` à HoVerNetLoss (ligne 412)
- [ ] **Modification 3:** Afficher lambda_magnitude dans les logs (lignes 415-420)

### Étape 3: Vérifier Syntaxe

```bash
python scripts/training/train_hovernet_family.py --help | grep lambda_magnitude
# Attendu: --lambda_magnitude LAMBDA_MAGNITUDE
```

### Étape 4: Test Dry-Run (Sans GPU)

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 1 \
    --lambda_hv 3.0 \
    --lambda_magnitude 5.0 \
    --help | grep "Poids magnitude"
# Attendu: affichage aide avec description
```

---

## 🚀 Commande Re-training Complète

**Après application de TOUS les fixes (hovernet_decoder.py + train_hovernet_family.py):**

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_hv 3.0 \
    --lambda_magnitude 5.0 \
    --batch_size 8 \
    --lr 1e-4
```

**Sortie attendue dans les logs:**
```
🔧 Initialisation du décodeur HoVer-Net...
  Paramètres: 12,345,678 (12.3M)
  Loss: Poids fixes (NP=1.0, HV=3.0, NT=1.0, Magnitude=5.0)
                                                   ^^^^^^^^^^^
                                                   NOUVEAU
```

---

## 🔬 Monitoring pendant Training

**Après 5 epochs, vérifier magnitude:**

```bash
# Vous verrez dans les logs train_losses:
# Epoch 5/50
# Train - Loss: 2.8456
#         hv_l1: 0.0215
#         hv_gradient: 0.0108
#         hv_magnitude: 0.3521  ← DOIT AUGMENTER (>0.25 après 5 epochs)
#                       ^^^^^^
#                       ATTENDU: >0.25 (indicateur succès)
```

**Si magnitude <0.10 après 5 epochs:**
- ❌ Le fix n'a pas fonctionné
- Vérifier que `magnitude_loss()` est bien celle corrigée (epsilon dans racine)
- Vérifier que `lambda_magnitude=5.0` est bien passé

**Si magnitude >0.25 après 5 epochs:**
- ✅ Le fix fonctionne! Continuer le training
- Attendu à epoch 50: magnitude >0.50

---

## 📊 Résumé des Changements

| Fichier | Lignes modifiées | Description |
|---------|------------------|-------------|
| `hovernet_decoder.py` | 302-361 | Fonction `magnitude_loss()` corrigée |
| `hovernet_decoder.py` | ~240 | Paramètre `lambda_magnitude` ajouté à `__init__` |
| `hovernet_decoder.py` | ~416 | Utilisation `self.lambda_magnitude` dans calcul |
| `train_hovernet_family.py` | ~351 | Argument CLI `--lambda_magnitude` |
| `train_hovernet_family.py` | ~412 | Passage paramètre à `HoVerNetLoss()` |
| `train_hovernet_family.py` | ~420 | Affichage logs |

**Total:** 2 fichiers modifiés, 6 sections touchées

---

## ❓ Troubleshooting

### Erreur: "HoVerNetLoss() got an unexpected keyword argument 'lambda_magnitude'"

**Cause:** `hovernet_decoder.py` n'a pas été modifié correctement.

**Solution:** Vérifier que `__init__` accepte `lambda_magnitude` paramètre.

```python
# Vérifier dans hovernet_decoder.py ligne ~240
def __init__(self, lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0, lambda_magnitude=5.0, adaptive=False):
    #                                                           ^^^^^^^^^^^^^^^^^^^^ DOIT être présent
```

---

### Erreur: "unrecognized arguments: --lambda_magnitude"

**Cause:** `train_hovernet_family.py` n'a pas été modifié correctement.

**Solution:** Vérifier que l'argument CLI est bien ajouté ligne ~351.

```python
# Vérifier dans train_hovernet_family.py ligne ~351
parser.add_argument('--lambda_magnitude', type=float, default=5.0,
                   help='Poids magnitude loss (Expert: 5.0 pour forcer gradients forts)')
```

---

### Magnitude reste à 0.02 après 5 epochs

**Cause:** Bug #2 (masking) pas corrigé correctement dans `magnitude_loss()`.

**Solution:** Vérifier que la fonction utilise bien:
```python
loss = (mag_true - mag_pred)**2  # Erreur manuelle
weighted_loss = loss * mask.squeeze(1)  # Masque AVANT réduction
return weighted_loss.sum() / (mask.sum() + 1e-6)  # Normalisation cellules seulement
```

**PAS:**
```python
mag_loss_sum = F.mse_loss(mag_pred_masked, mag_target_masked, reduction='sum')  # ← BUGUÉ
```

---

**STATUT:** ✅ Code de modification prêt

**NEXT STEP:** Appliquer les fixes dans les 2 fichiers, puis lancer re-training
