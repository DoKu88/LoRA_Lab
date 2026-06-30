"""LoRA_Lab — Text-to-LoRA hypernetwork.

Train and evaluate a text-conditioned hypernetwork that generates LoRA adapters,
under two objectives: (1) reconstruction of existing library LoRAs, and (2)
generalization (generate a LoRA from a task description and backprop the task
loss through a frozen base into the hypernetwork).
"""

__version__ = "0.0.1"
