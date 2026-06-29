"""LoRA_Lab — Text-to-LoRA hypernetwork.

Train + evaluate a text-conditioned hypernetwork that generates LoRA adapters,
under two objectives: (1) reconstruction of existing library LoRAs, and (2) SFT
(the Text-to-LoRA recipe) through a frozen base. See docs/phase-2-sprint-plan.md.
"""

__version__ = "0.0.1"
