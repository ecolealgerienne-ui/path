# ✅ VALIDATION PHASE 2 — Hybrid Architecture

## Fichiers Créés

1. **Architecture**: `src/models/hovernet_decoder_hybrid.py` (~300 lignes)
2. **Tests unitaires**: `scripts/validation/test_hybrid_architecture.py` (~350 lignes)

**Composants implémentés**:
- ✅ HoVerNetDecoderHybrid class
- ✅ Bottleneck RGB (1536 → 256, Conv2d 1×1)
- ✅ Bottleneck H (256 → 256, Linear projection)
- ✅ **Fusion additive** (rgb_map + h_map)
- ✅ Shared decoder (2 Conv layers + Dropout)
- ✅ Upsampling (16×16 → 224×224)
- ✅ 3 Branches (NP, HV tanh, NT)
- ✅ HybridDecoderOutput dataclass

---

## 🔧 Commande de Validation

```bash
# Activer environnement cellvit
conda activate cellvit

# Lancer tests unitaires
python scripts/validation/test_hybrid_architecture.py

# Si succès, sortie attendue:
# 🎉 ALL TESTS PASSED! Architecture is ready for training.
```

---

## ✅ Critères de Validation (5 Tests)

### Test 1: Forward Pass
```
✅ ATTENDU:
- NP output: (B, 2, 224, 224)
- HV output: (B, 2, 224, 224), range [-1, 1]
- NT output: (B, n_classes, 224, 224)
```

### Test 2: Gradient Flow
```
✅ ATTENDU:
- RGB gradients ≠ None
- H gradients ≠ None
- Gradient norms > 1e-6
- Gradient ratio (max/min) < 100 (balance)
```

**⚠️ ALERTE** si ratio > 100: Déséquilibre gradients → Ajuster LR ou poids loss

### Test 3: Fusion Additive
```
✅ ATTENDU:
- RGB-only vs Both: différence > 1e-4
- H-only vs Both: différence > 1e-4
- Relative change > 1%
```

Prouve que fusion est **additive** (pas concatenation) et que **les 2 branches contribuent**.

### Test 4: Output Activations
```
✅ ATTENDU:
- HV range: [-1, 1] (Tanh applied)
- NP après sigmoid: [0, 1]
- NT après softmax: sum=1.0
- to_numpy() method fonctionne
```

### Test 5: Parameter Count
```
✅ ATTENDU:
- Params trainable: [100k, 100M]
- Optimal: ~20-30M params
```

---

## 📊 Sortie Attendue (Tests Réussis)

```
🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬
HOVERNET DECODER HYBRID — UNIT TESTS
🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬

================================================================================
TEST 1: FORWARD PASS
================================================================================
Input shapes:
  patch_tokens: torch.Size([2, 256, 1536])
  h_features: torch.Size([2, 256])

Output shapes:
  np_out: torch.Size([2, 2, 224, 224]) ✅
  hv_out: torch.Size([2, 2, 224, 224]) ✅
  nt_out: torch.Size([2, 5, 224, 224]) ✅

HV range: [-0.XXX, 0.XXX] ✅

✅ TEST 1 PASSED: Forward pass OK

================================================================================
TEST 2: GRADIENT FLOW
================================================================================

Gradient norms:
  RGB (patch_tokens): X.XXXX ✅
  H (h_features): X.XXXX ✅

Gradient ratio (max/min): X.XX
  ✅ Gradient balance OK

✅ TEST 2 PASSED: Gradient flow OK

================================================================================
TEST 3: FUSION ADDITIVE
================================================================================

Mean absolute differences:
  RGB-only vs Both: X.XXXX
  H-only vs Both: X.XXXX

✅ Both branches contribute to output

Relative change when adding H-channel: XX.XX%

✅ TEST 3 PASSED: Additive fusion OK

================================================================================
TEST 4: OUTPUT ACTIVATIONS
================================================================================

HV output range:
  Min: -X.XXXX
  Max: X.XXXX
  ✅ HV range OK (Tanh applied)

NP after sigmoid:
  Range: [X.XXXX, X.XXXX]
  ✅ NP range OK (Sigmoid applied)

NT after softmax:
  Sum over classes: 1.0000
  ✅ NT softmax OK

✅ TEST 4 PASSED: Output activations OK

================================================================================
TEST 5: PARAMETER COUNT
================================================================================

Parameter count:
  Trainable: XX,XXX,XXX
  Total: XX,XXX,XXX
  ✅ Parameter count reasonable
  ✅ Model size optimal (XX.XXM params)

✅ TEST 5 PASSED: Parameter count OK

================================================================================
TEST SUMMARY
================================================================================
✅ PASS   — Forward Pass
✅ PASS   — Gradient Flow
✅ PASS   — Fusion Additive
✅ PASS   — Output Activations
✅ PASS   — Parameter Count

Total: 5/5 tests passed

🎉 ALL TESTS PASSED! Architecture is ready for training.
```

---

## 🔍 Diagnostic en Cas d'Échec

### Problème 1: "HV range not in [-1, 1]"

**Cause**: Tanh non appliqué dans hv_head.

**Solution**:
```python
# Vérifier ligne 112 dans hovernet_decoder_hybrid.py:
self.hv_head = nn.Sequential(
    ...
    nn.Tanh()  # DOIT être présent
)
```

### Problème 2: "RGB gradients = None" ou "H gradients = None"

**Cause**: Problème dans fusion ou branches.

**Diagnostic**:
```bash
# Activer mode debug dans test:
# Ajouter dans test_gradient_flow():
print(f"RGB bottleneck weight grad: {model.bottleneck_rgb.weight.grad}")
print(f"H bottleneck weight grad: {model.bottleneck_h.weight.grad}")

# Si None: Vérifier que fusion additive utilise bien rgb_map + h_map
```

### Problème 3: "Gradient ratio > 100 (imbalance)"

**Cause**: Une branche domine l'autre.

**Solution**:
```python
# Dans training, utiliser LR séparés (Mitigation Risque 2):
optimizer = torch.optim.AdamW([
    {'params': model.bottleneck_rgb.parameters(), 'lr': 1e-4},
    {'params': model.bottleneck_h.parameters(), 'lr': 5e-5},  # LR plus faible
])
```

### Problème 4: "Both branches contribute to output: FAIL"

**Cause**: Fusion incorrecte (concatenation au lieu d'addition).

**Solution**:
```python
# Vérifier ligne 191 dans hovernet_decoder_hybrid.py:
fused = rgb_map + h_map  # DOIT être '+' (pas torch.cat)
```

---

## ✅ Checklist de Validation

- [ ] Test 1: Forward pass OK ✅
- [ ] Test 2: Gradient flow OK ✅
- [ ] Test 3: Fusion additive OK ✅
- [ ] Test 4: Output activations OK ✅
- [ ] Test 5: Parameter count OK ✅
- [ ] **5/5 tests passés**

---

## 🎯 Prochaine Étape si Validation OK

**Phase 3**: Créer `scripts/training/train_hovernet_family_v13_hybrid.py`

**Composants**:
1. `HybridDataset` class (charge RGB features + H features)
2. Training loop avec loss hybride
3. Validation loop
4. Checkpoint saving

**Temps estimé**: 2-3h

**Commande**:
```bash
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --epochs 30 \
    --batch_size 16 \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0 \
    --lambda_h_recon 0.1
```

---

**Date**: 2025-12-26
**Phase**: 2 - Hybrid Architecture
**Statut**: ⏳ En attente validation tests unitaires
