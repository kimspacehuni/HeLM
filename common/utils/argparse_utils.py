"""Argument types shared by training and evaluation entrypoints."""
import argparse


def str_to_bool(value):
    """Accept explicit booleans, including the existing ``--flag False`` syntax."""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, on/off, or 1/0.")
