"""Shared fixtures for Phase 0 math tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def dtype():
    return torch.float64


@pytest.fixture
def seed():
    return 42
