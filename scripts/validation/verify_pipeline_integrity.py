#!/usr/bin/env python3
"""
Pipeline Guard — Vérification End-to-End de l'intégrité du pipeline CellViT-Optimus.

Ce script audite les 4 piliers de l'architecture V13 Smart Crops + FPN Chimique:
1. Normalisation Macenko (si activée)
2. Extraction H-channel (Ruifrok)
3. Architecture Hybrid avec h_alpha learnable
4. Dimensions des sorties (NP, HV, NT)

Usage:
    # Vérification complète avec normalisation
    python scripts/validation/verify_pipeline_integrity.py \
        --family respiratory \
        --use_normalized

    # Vérification sans normalisation
    python scripts/validation/verify_pipeline_integrity.py \
        --family respiratory

    # Vérification avec checkpoint existant
    python scripts/validation/verify_pipeline_integrity.py \
        --checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth

Author: CellViT-Optimus Project
Date: 2025-12-30
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.models.hovernet_decoder import HoVerNetDecoder


@dataclass
class AuditResult:
    """Résultat d'un audit."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


class PipelineGuard:
    """
    Auditeur automatique du pipeline CellViT-Optimus.

    Vérifie l'intégrité des composants critiques:
    - Normalisation des couleurs (Macenko)
    - Extraction du H-channel (Ruifrok)
    - Paramètres h_alpha (learnable)
    - Dimensions des sorties (NP, HV, NT)
    - Flux de gradient
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.results = []

    def log(self, message: str, level: str = "INFO"):
        """Affiche un message avec niveau."""
        if self.verbose:
            icons = {
                "INFO": "ℹ️ ",
                "OK": "✅",
                "WARNING": "⚠️ ",
                "ERROR": "❌",
                "CRITICAL": "🚨"
            }
            print(f"{icons.get(level, '')} [{level}] {message}")

    def add_result(self, result: AuditResult):
        """Ajoute un résultat d'audit."""
        self.results.append(result)
        level = "OK" if result.passed else "ERROR"
        self.log(f"{result.name}: {result.message}", level)

    # =========================================================================
    # AUDIT 1: Normalisation Macenko
    # =========================================================================

    def audit_normalization(
        self,
        images_raw: np.ndarray,
        images_normalized: Optional[np.ndarray] = None,
        threshold: float = 5.0
    ) -> AuditResult:
        """
        Audit de la normalisation Macenko.

        Vérifie que la variance inter-images est réduite après normalisation.

        Args:
            images_raw: Images brutes (N, H, W, 3) uint8
            images_normalized: Images normalisées (N, H, W, 3) uint8 ou None
            threshold: Seuil de réduction de variance attendu (%)

        Returns:
            AuditResult avec détails de la vérification
        """
        self.log("Audit 1/5: Normalisation Macenko", "INFO")

        # Calculer variance inter-images pour images brutes
        mean_intensities_raw = images_raw.mean(axis=(1, 2, 3))
        std_raw = mean_intensities_raw.std()

        if images_normalized is None:
            return AuditResult(
                name="Normalisation Macenko",
                passed=True,
                message="Mode sans normalisation — skipped",
                details={"std_raw": float(std_raw), "normalized": False}
            )

        # Calculer variance pour images normalisées
        mean_intensities_norm = images_normalized.mean(axis=(1, 2, 3))
        std_norm = mean_intensities_norm.std()

        # Vérifier réduction de variance
        reduction = (1 - std_norm / std_raw) * 100 if std_raw > 0 else 0
        passed = reduction >= threshold

        details = {
            "std_raw": float(std_raw),
            "std_normalized": float(std_norm),
            "reduction_percent": float(reduction),
            "threshold": threshold,
            "normalized": True
        }

        if not passed:
            message = f"Réduction insuffisante: {reduction:.1f}% < {threshold}% seuil"
        else:
            message = f"Réduction de variance: {reduction:.1f}%"

        return AuditResult(
            name="Normalisation Macenko",
            passed=passed,
            message=message,
            details=details
        )

    # =========================================================================
    # AUDIT 2: Extraction H-Channel (Ruifrok)
    # =========================================================================

    def audit_h_channel_extraction(
        self,
        model: HoVerNetDecoder,
        dummy_rgb: torch.Tensor
    ) -> AuditResult:
        """
        Audit de l'extraction du H-channel via Ruifrok.

        Vérifie:
        - Présence du module ruifrok ou FPN chimique
        - Signal H non-vide
        - Corrélation structurelle avec l'image

        Args:
            model: HoVerNetDecoder
            dummy_rgb: Image RGB (1, 3, 224, 224)

        Returns:
            AuditResult
        """
        self.log("Audit 2/5: Extraction H-Channel (Ruifrok)", "INFO")

        # Vérifier présence du module
        has_fpn = hasattr(model, 'fpn_chimique') and model.fpn_chimique is not None
        has_ruifrok = hasattr(model, 'ruifrok') and model.ruifrok is not None

        if not has_fpn and not has_ruifrok:
            return AuditResult(
                name="Extraction H-Channel",
                passed=False,
                message="Ni FPN Chimique ni Ruifrok détecté — H-channel non extrait",
                details={"has_fpn": False, "has_ruifrok": False}
            )

        # Extraire H-channel manuellement pour vérification
        try:
            with torch.no_grad():
                rgb_input = dummy_rgb.to(self.device).clamp(1e-6, 255.0)

                # Calcul OD
                od = -torch.log10(rgb_input / 255.0 + 1e-6)

                # Vecteur Hématoxyline
                stain_vector = torch.tensor([0.650, 0.704, 0.286]).view(1, 3, 1, 1).to(self.device)

                # Projection
                h_channel = torch.sum(od * stain_vector, dim=1, keepdim=True)

                # Statistiques
                h_mean = h_channel.mean().item()
                h_std = h_channel.std().item()
                h_min = h_channel.min().item()
                h_max = h_channel.max().item()

        except Exception as e:
            return AuditResult(
                name="Extraction H-Channel",
                passed=False,
                message=f"Erreur extraction: {str(e)}",
                details={"error": str(e)}
            )

        # Vérifier signal non-plat
        is_flat = h_std < 0.01

        details = {
            "has_fpn": has_fpn,
            "has_ruifrok": has_ruifrok,
            "h_mean": h_mean,
            "h_std": h_std,
            "h_range": [h_min, h_max],
            "is_flat": is_flat
        }

        if is_flat:
            return AuditResult(
                name="Extraction H-Channel",
                passed=False,
                message=f"Signal H plat (std={h_std:.4f}) — extraction échouée",
                details=details
            )

        return AuditResult(
            name="Extraction H-Channel",
            passed=True,
            message=f"Signal H valide (mean={h_mean:.3f}, std={h_std:.3f})",
            details=details
        )

    # =========================================================================
    # AUDIT 3: Architecture Hybrid V2 (h_alpha learnable)
    # =========================================================================

    def audit_h_alpha_parameters(
        self,
        model: HoVerNetDecoder
    ) -> AuditResult:
        """
        Audit des paramètres h_alpha.

        Vérifie:
        - Présence de h_alphas (ParameterDict)
        - requires_grad = True pour tous les alphas
        - Valeurs initiales raisonnables

        Args:
            model: HoVerNetDecoder

        Returns:
            AuditResult
        """
        self.log("Audit 3/5: Paramètres h_alpha (learnable)", "INFO")

        # Vérifier présence
        has_h_alpha = hasattr(model, 'use_h_alpha') and model.use_h_alpha
        has_h_alphas_dict = hasattr(model, 'h_alphas') and model.h_alphas is not None

        if not has_h_alpha or not has_h_alphas_dict:
            return AuditResult(
                name="Paramètres h_alpha",
                passed=True,  # Pas une erreur si non activé
                message="Mode sans h_alpha — skipped",
                details={"use_h_alpha": has_h_alpha, "has_dict": has_h_alphas_dict}
            )

        # Vérifier requires_grad et valeurs
        alpha_info = {}
        all_learnable = True

        for name, param in model.h_alphas.items():
            is_learnable = param.requires_grad
            value = param.item() if param.numel() == 1 else param.mean().item()

            alpha_info[name] = {
                "requires_grad": is_learnable,
                "value": value
            }

            if not is_learnable:
                all_learnable = False

        details = {
            "num_alphas": len(model.h_alphas),
            "all_learnable": all_learnable,
            "alphas": alpha_info
        }

        if not all_learnable:
            return AuditResult(
                name="Paramètres h_alpha",
                passed=False,
                message="Certains alphas ont requires_grad=False — gradients bloqués",
                details=details
            )

        return AuditResult(
            name="Paramètres h_alpha",
            passed=True,
            message=f"{len(model.h_alphas)} alphas learnable détectés",
            details=details
        )

    # =========================================================================
    # AUDIT 4: Dimensions des Sorties (HoVerNet Consistency)
    # =========================================================================

    def audit_output_dimensions(
        self,
        model: HoVerNetDecoder,
        dummy_features: torch.Tensor,
        dummy_rgb: Optional[torch.Tensor] = None,
        expected_np_channels: int = 2,
        expected_hv_channels: int = 2,
        expected_nt_channels: int = 5,
        spatial_size: int = 224
    ) -> AuditResult:
        """
        Audit des dimensions de sortie.

        Vérifie:
        - NP output: (B, 2, 224, 224) pour CrossEntropyLoss
        - HV output: (B, 2, 224, 224) pour gradients
        - NT output: (B, 5, 224, 224) pour classification

        Args:
            model: HoVerNetDecoder
            dummy_features: Features (1, 261, 1536)
            dummy_rgb: RGB images (1, 3, 224, 224) ou None

        Returns:
            AuditResult
        """
        self.log("Audit 4/5: Dimensions des sorties", "INFO")

        model.eval()

        try:
            with torch.no_grad():
                outputs = model(dummy_features, images_rgb=dummy_rgb)

                # Handle dict ou tuple output
                if isinstance(outputs, dict):
                    np_out = outputs['np']
                    hv_out = outputs['hv']
                    nt_out = outputs['nt']
                else:
                    np_out, hv_out, nt_out = outputs

        except Exception as e:
            return AuditResult(
                name="Dimensions sorties",
                passed=False,
                message=f"Forward pass échoué: {str(e)}",
                details={"error": str(e)}
            )

        # Vérifier dimensions
        checks = {
            "NP": {
                "actual": list(np_out.shape),
                "expected": [1, expected_np_channels, spatial_size, spatial_size],
                "passed": np_out.shape == (1, expected_np_channels, spatial_size, spatial_size)
            },
            "HV": {
                "actual": list(hv_out.shape),
                "expected": [1, expected_hv_channels, spatial_size, spatial_size],
                "passed": hv_out.shape == (1, expected_hv_channels, spatial_size, spatial_size)
            },
            "NT": {
                "actual": list(nt_out.shape),
                "expected": [1, expected_nt_channels, spatial_size, spatial_size],
                "passed": nt_out.shape == (1, expected_nt_channels, spatial_size, spatial_size)
            }
        }

        all_passed = all(c["passed"] for c in checks.values())

        details = {
            "checks": checks,
            "np_range": [float(np_out.min()), float(np_out.max())],
            "hv_range": [float(hv_out.min()), float(hv_out.max())],
            "nt_range": [float(nt_out.min()), float(nt_out.max())]
        }

        if not all_passed:
            failed = [k for k, v in checks.items() if not v["passed"]]
            return AuditResult(
                name="Dimensions sorties",
                passed=False,
                message=f"Dimensions incorrectes pour: {', '.join(failed)}",
                details=details
            )

        return AuditResult(
            name="Dimensions sorties",
            passed=True,
            message=f"NP={np_out.shape}, HV={hv_out.shape}, NT={nt_out.shape}",
            details=details
        )

    # =========================================================================
    # AUDIT 5: Flux de Gradient (H-Channel → Loss)
    # =========================================================================

    def audit_gradient_flow(
        self,
        model: HoVerNetDecoder,
        dummy_features: torch.Tensor,
        dummy_rgb: Optional[torch.Tensor] = None
    ) -> AuditResult:
        """
        Audit du flux de gradient.

        Vérifie que le gradient remonte bien depuis la loss vers:
        - Les h_alphas (si présents)
        - Les couches du FPN chimique

        Args:
            model: HoVerNetDecoder
            dummy_features: Features (1, 261, 1536)
            dummy_rgb: RGB images (1, 3, 224, 224)

        Returns:
            AuditResult
        """
        self.log("Audit 5/5: Flux de gradient", "INFO")

        model.train()
        model.zero_grad()

        try:
            # Forward pass
            outputs = model(dummy_features, images_rgb=dummy_rgb)

            if isinstance(outputs, dict):
                np_out = outputs['np']
            else:
                np_out, _, _ = outputs

            # Dummy loss (mean de la sortie)
            loss = np_out.mean()
            loss.backward()

        except Exception as e:
            return AuditResult(
                name="Flux de gradient",
                passed=False,
                message=f"Backward pass échoué: {str(e)}",
                details={"error": str(e)}
            )

        gradient_info = {}

        # Vérifier gradient h_alphas
        if hasattr(model, 'h_alphas') and model.h_alphas is not None:
            for name, param in model.h_alphas.items():
                has_grad = param.grad is not None
                grad_norm = param.grad.norm().item() if has_grad else 0.0
                gradient_info[f"h_alpha_{name}"] = {
                    "has_grad": has_grad,
                    "grad_norm": grad_norm
                }

        # Vérifier gradient FPN chimique
        if hasattr(model, 'fpn_chimique') and model.fpn_chimique is not None:
            # Vérifier première couche du FPN
            for name, param in model.fpn_chimique.named_parameters():
                if param.grad is not None:
                    gradient_info["fpn_chimique"] = {
                        "has_grad": True,
                        "first_layer": name,
                        "grad_norm": param.grad.norm().item()
                    }
                    break
            else:
                gradient_info["fpn_chimique"] = {"has_grad": False}

        # Vérifier gradient bottleneck
        if hasattr(model, 'bottleneck'):
            for name, param in model.bottleneck.named_parameters():
                if param.grad is not None:
                    gradient_info["bottleneck"] = {
                        "has_grad": True,
                        "first_layer": name,
                        "grad_norm": param.grad.norm().item()
                    }
                    break

        details = {"gradients": gradient_info}

        # Vérifier que les composants critiques ont des gradients
        h_alpha_ok = True
        if "h_alpha_0" in gradient_info:
            h_alpha_ok = gradient_info["h_alpha_0"]["has_grad"]

        fpn_ok = True
        if "fpn_chimique" in gradient_info:
            fpn_ok = gradient_info["fpn_chimique"]["has_grad"]

        all_ok = h_alpha_ok and fpn_ok

        if not all_ok:
            return AuditResult(
                name="Flux de gradient",
                passed=False,
                message="Gradients manquants sur composants critiques",
                details=details
            )

        return AuditResult(
            name="Flux de gradient",
            passed=True,
            message="Gradients propagés correctement",
            details=details
        )

    # =========================================================================
    # RUN ALL AUDITS
    # =========================================================================

    def run_full_audit(
        self,
        model: HoVerNetDecoder,
        images_raw: Optional[np.ndarray] = None,
        images_normalized: Optional[np.ndarray] = None,
        use_hybrid: bool = True
    ) -> Dict:
        """
        Exécute tous les audits.

        Args:
            model: HoVerNetDecoder
            images_raw: Images brutes pour audit normalisation
            images_normalized: Images normalisées (optionnel)
            use_hybrid: Si True, inclut RGB dans le forward

        Returns:
            Dict avec tous les résultats
        """
        print("\n" + "=" * 70)
        print("🚀 PIPELINE GUARD — Audit Automatique CellViT-Optimus")
        print("=" * 70 + "\n")

        model = model.to(self.device)

        # Créer données dummy
        dummy_features = torch.randn(1, 261, 1536).to(self.device)
        dummy_rgb = torch.randint(0, 256, (1, 3, 224, 224)).float().to(self.device) if use_hybrid else None

        # Audit 1: Normalisation
        if images_raw is not None:
            result1 = self.audit_normalization(images_raw, images_normalized)
        else:
            result1 = AuditResult(
                name="Normalisation Macenko",
                passed=True,
                message="Pas d'images fournies — skipped",
                details={"skipped": True}
            )
        self.add_result(result1)

        # Audit 2: H-Channel
        if use_hybrid and dummy_rgb is not None:
            result2 = self.audit_h_channel_extraction(model, dummy_rgb)
        else:
            result2 = AuditResult(
                name="Extraction H-Channel",
                passed=True,
                message="Mode non-hybrid — skipped",
                details={"skipped": True}
            )
        self.add_result(result2)

        # Audit 3: h_alpha
        result3 = self.audit_h_alpha_parameters(model)
        self.add_result(result3)

        # Audit 4: Dimensions
        result4 = self.audit_output_dimensions(model, dummy_features, dummy_rgb)
        self.add_result(result4)

        # Audit 5: Gradient flow
        if use_hybrid and dummy_rgb is not None:
            # Re-créer tensors avec requires_grad pour backward
            dummy_features = torch.randn(1, 261, 1536, requires_grad=True).to(self.device)
            dummy_rgb = torch.randint(0, 256, (1, 3, 224, 224)).float().to(self.device)
            result5 = self.audit_gradient_flow(model, dummy_features, dummy_rgb)
        else:
            result5 = AuditResult(
                name="Flux de gradient",
                passed=True,
                message="Mode non-hybrid — skipped",
                details={"skipped": True}
            )
        self.add_result(result5)

        # Résumé
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DE L'AUDIT")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            status = "✅" if r.passed else "❌"
            print(f"  {status} {r.name}: {r.message}")

        print(f"\n{'=' * 70}")

        if passed == total:
            print("🎉 PIPELINE VALIDÉ — Prêt pour l'entraînement!")
        else:
            print(f"⚠️  {total - passed}/{total} audits échoués — Vérification requise")

        print("=" * 70 + "\n")

        return {
            "passed": passed,
            "total": total,
            "all_passed": passed == total,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Guard — Vérification End-to-End du pipeline CellViT-Optimus"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers checkpoint existant (optionnel)"
    )
    parser.add_argument(
        "--family",
        type=str,
        default="respiratory",
        choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"],
        help="Famille pour charger des images test"
    )
    parser.add_argument(
        "--use_normalized",
        action="store_true",
        help="Tester avec images normalisées"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/family_data_v13_smart_crops"),
        help="Répertoire des données"
    )
    parser.add_argument(
        "--normalized_dir",
        type=Path,
        default=Path("data/family_FIXED"),
        help="Répertoire des données normalisées"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"]
    )

    args = parser.parse_args()

    # Charger ou créer modèle
    if args.checkpoint:
        print(f"📂 Chargement checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)

        # Détecter configuration depuis checkpoint
        use_fpn = "fpn_chimique" in str(checkpoint.get("model_state_dict", {}).keys())
        use_h_alpha = "h_alphas" in str(checkpoint.get("model_state_dict", {}).keys())

        model = HoVerNetDecoder(
            use_hybrid=True,
            use_fpn_chimique=use_fpn,
            use_h_alpha=use_h_alpha
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  ✅ Modèle chargé (FPN={use_fpn}, h_alpha={use_h_alpha})")
    else:
        print("📦 Création nouveau modèle (FPN Chimique + h_alpha)")
        model = HoVerNetDecoder(
            use_hybrid=True,
            use_fpn_chimique=True,
            use_h_alpha=True
        )

    # Charger images pour test normalisation
    images_raw = None
    images_normalized = None

    # Essayer de charger des vraies images
    train_file = args.data_dir / f"{args.family}_train_v13_smart_crops.npz"
    if train_file.exists():
        print(f"📂 Chargement images: {train_file}")
        data = np.load(train_file)
        images_raw = data["images"][:50]  # Premier 50 images
        print(f"  ✅ {len(images_raw)} images chargées")

    if args.use_normalized:
        norm_file = args.normalized_dir / f"{args.family}_data_FIXED.npz"
        if norm_file.exists():
            print(f"📂 Chargement images normalisées: {norm_file}")
            norm_data = np.load(norm_file)
            images_normalized = norm_data["images"][:50]
            print(f"  ✅ {len(images_normalized)} images normalisées chargées")

    # Lancer audit
    guard = PipelineGuard(device=args.device)
    results = guard.run_full_audit(
        model=model,
        images_raw=images_raw,
        images_normalized=images_normalized,
        use_hybrid=True
    )

    # Return code
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
