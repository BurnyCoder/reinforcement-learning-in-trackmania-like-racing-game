"""RacingMania: GPU simulator port, RL training, evaluation and reporting.

Global context: the browser game lives in ``web/`` and uses the original cannon-es car physics.
This package holds the batched PyTorch port of that physics (``sim``), the tracks (``tracks``),
the RL environment and algorithms (``env``, ``rl``), the non-learning drivers (``drivers``),
the evaluation (``eval``) and the report pipeline (``report``).
"""

__version__ = "0.1.0"
