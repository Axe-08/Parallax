"""
parallax core module.
"""

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str = Field(default="healthy", description="System health status")
    version: str = Field(default="0.1.0", description="System version")


def get_health() -> HealthStatus:
    """Return initial health status."""
    return HealthStatus()
