"""Benchmark harness for the four conditions (A) status quo, (B) continued
pretraining, (C) zero-shot transfer, (D) Parity.

``benchmarks.tasks`` holds the metrics, ``benchmarks.cost`` the FLOP accounting,
``benchmarks.report`` the tables, and ``benchmarks.run`` the driver.
"""

__all__ = ["tasks", "cost", "report"]
