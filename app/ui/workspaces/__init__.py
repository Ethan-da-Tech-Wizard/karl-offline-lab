# ==============================================================================
# Program: __init__.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Init .
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================


# This package groups the ten top-level workspaces shown in the sidebar
# (workbench, prompt_lab, knowledge_base, vision_workbench, training_studio,
# eval_suite, swarm_studio, system_config, docs, flywheel_studio) plus
# supporting studios. It intentionally exports nothing: MainWindow imports each
# workspace class directly from its own module, which keeps startup imports
# explicit and avoids circular imports between workspaces.
