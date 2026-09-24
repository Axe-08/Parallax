"""
Smoke tests for parallax.
"""

from parallax.core import get_health


def test_health_smoke():
    health = get_health()
    assert health.status == "healthy"
    assert health.version == "0.1.0"
