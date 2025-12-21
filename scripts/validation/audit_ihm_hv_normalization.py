#!/usr/bin/env python3
"""
Audit de l'IHM pour vérification normalisation HV [-1, 1].

Vérifie:
1. Activation tanh() dans le décodeur HoVer-Net
2. Pas de scaling * 127 ou / 127 dans l'inférence
3. Visualisations avec vmin/vmax corrects
4. Seuils watershed adaptés

Usage:
    python scripts/validation/audit_ihm_hv_normalization.py
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


class IHMAuditor:
    """Auditeur pour vérifier la cohérence HV normalization dans l'IHM."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        self.successes = []

    def check_file(self, filepath: Path, checks: List[Tuple[str, str, str]]):
        """
        Vérifie un fichier avec des patterns spécifiques.

        Args:
            filepath: Chemin du fichier
            checks: Liste de (pattern, expected, issue_message)
        """
        if not filepath.exists():
            self.warnings.append(f"❓ Fichier introuvable: {filepath}")
            return

        content = filepath.read_text()

        for pattern, expected, issue_msg in checks:
            matches = re.findall(pattern, content, re.MULTILINE)

            if expected == "PRESENT":
                if matches:
                    self.successes.append(f"✅ {filepath.name}: {issue_msg} trouvé")
                else:
                    self.issues.append(f"❌ {filepath.name}: {issue_msg} manquant")
            elif expected == "ABSENT":
                if matches:
                    self.issues.append(f"❌ {filepath.name}: {issue_msg} détecté: {matches[:3]}")
                else:
                    self.successes.append(f"✅ {filepath.name}: {issue_msg} absent (OK)")
            elif expected == "CHECK":
                if matches:
                    self.warnings.append(f"⚠️  {filepath.name}: {issue_msg} → {matches[:3]}")

    def audit_decoder(self):
        """Vérifie le décodeur HoVer-Net."""
        print("\n" + "="*70)
        print("1. AUDIT DÉCODEUR HOVERNET")
        print("="*70)

        decoder_path = self.project_root / "src/models/hovernet_decoder.py"

        checks = [
            # Vérifier activation tanh pour HV
            (r"nn\.Tanh\(\)", "PRESENT", "Activation tanh() pour HV"),

            # Vérifier qu'il n'y a pas de scaling incorrect
            (r"\*\s*127|\*127|/\s*127|/127", "ABSENT", "Scaling * 127 ou / 127"),
        ]

        self.check_file(decoder_path, checks)

        # Check spécifique: Lire la définition de hv_head
        if decoder_path.exists():
            content = decoder_path.read_text()

            # Chercher la définition de hv_head
            hv_head_match = re.search(
                r"self\.hv_head\s*=\s*([^\n]+)",
                content
            )

            if hv_head_match:
                hv_head_def = hv_head_match.group(1)
                print(f"\n📋 Définition hv_head:\n   {hv_head_def}")

                if "Tanh" not in hv_head_def and "tanh" not in hv_head_def:
                    self.warnings.append(
                        "⚠️  hovernet_decoder.py: hv_head sans activation tanh() explicite\n"
                        "   Le modèle apprend naturellement à produire [-1, 1] via SmoothL1 loss,\n"
                        "   mais tanh() explicite serait plus robuste selon HoVer-Net paper."
                    )

    def audit_inference_files(self):
        """Vérifie les fichiers d'inférence."""
        print("\n" + "="*70)
        print("2. AUDIT FICHIERS D'INFÉRENCE")
        print("="*70)

        inference_files = [
            "src/inference/hoptimus_hovernet.py",
            "src/inference/optimus_gate_inference.py",
            "src/inference/optimus_gate_inference_multifamily.py",
        ]

        checks = [
            # Vérifier qu'il n'y a pas de scaling incorrect
            (r"hv.*\*\s*127|hv.*\*127|hv.*/\s*127|hv.*/127", "ABSENT", "HV scaling * 127 ou / 127"),

            # Vérifier forward_features() utilisé
            (r"forward_features", "PRESENT", "forward_features() (pas blocks[X])"),

            # Vérifier qu'il n'y a pas de hooks sur blocks
            (r"\.blocks\[", "ABSENT", "Hooks sur blocks[X]"),
        ]

        for filepath in inference_files:
            full_path = self.project_root / filepath
            self.check_file(full_path, checks)

    def audit_visualizations(self):
        """Vérifie les visualisations."""
        print("\n" + "="*70)
        print("3. AUDIT VISUALISATIONS")
        print("="*70)

        viz_files = [
            "scripts/demo/gradio_demo.py",
            "src/inference/hoptimus_hovernet.py",
        ]

        checks = [
            # Vérifier vmin/vmax pour HV maps
            (r"vmin\s*=\s*-127|vmax\s*=\s*127", "ABSENT", "vmin/vmax avec valeurs [-127, 127]"),
            (r"vmin\s*=\s*-1|vmax\s*=\s*1", "CHECK", "vmin/vmax avec valeurs [-1, 1]"),

            # Vérifier imshow avec HV
            (r"imshow.*hv", "CHECK", "Visualisation HV"),
        ]

        for filepath in viz_files:
            full_path = self.project_root / filepath
            if full_path.exists():
                self.check_file(full_path, checks)

    def audit_watershed(self):
        """Vérifie les seuils watershed."""
        print("\n" + "="*70)
        print("4. AUDIT POST-PROCESSING WATERSHED")
        print("="*70)

        watershed_files = [
            "src/inference/hoptimus_hovernet.py",
        ]

        checks = [
            # Chercher edge_threshold ou dist_threshold
            (r"edge_threshold\s*=\s*([\d.]+)", "CHECK", "edge_threshold"),
            (r"dist_threshold\s*=\s*([\d.]+)", "CHECK", "dist_threshold"),

            # Chercher Sobel sur HV
            (r"Sobel.*hv", "CHECK", "Sobel sur HV maps"),
        ]

        for filepath in watershed_files:
            full_path = self.project_root / filepath
            self.check_file(full_path, checks)

    def print_report(self):
        """Affiche le rapport final."""
        print("\n" + "="*70)
        print("RAPPORT D'AUDIT IHM - NORMALISATION HV")
        print("="*70)

        print(f"\n✅ SUCCÈS ({len(self.successes)}):")
        for success in self.successes:
            print(f"   {success}")

        print(f"\n⚠️  AVERTISSEMENTS ({len(self.warnings)}):")
        if self.warnings:
            for warning in self.warnings:
                print(f"   {warning}")
        else:
            print("   Aucun")

        print(f"\n❌ PROBLÈMES ({len(self.issues)}):")
        if self.issues:
            for issue in self.issues:
                print(f"   {issue}")
        else:
            print("   Aucun")

        print("\n" + "="*70)
        print("RECOMMANDATIONS")
        print("="*70)

        if not self.issues and not self.warnings:
            print("\n🎉 AUDIT COMPLET: Aucun problème détecté!")
            print("   L'IHM est prête pour les modèles FIXED.")
        elif not self.issues:
            print("\n✅ AUDIT OK avec avertissements mineurs")
            print("   Vous pouvez procéder, mais vérifiez les avertissements ci-dessus.")
        else:
            print("\n⚠️  AUDIT ÉCHOUÉ: Des problèmes doivent être corrigés")
            print("   Suivez le plan d'intégration: INTEGRATION_PLAN_HV_NORMALIZATION.md")

        print("\n" + "="*70)

        return len(self.issues) == 0


def main():
    project_root = Path(__file__).parent.parent.parent

    auditor = IHMAuditor(project_root)

    auditor.audit_decoder()
    auditor.audit_inference_files()
    auditor.audit_visualizations()
    auditor.audit_watershed()

    success = auditor.print_report()

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
