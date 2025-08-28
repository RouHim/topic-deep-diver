"""
Utility functions and constants for Topic Deep Diver.
"""

# UTC timezone compatibility for Python <3.11
try:
    from datetime import UTC as UTC_TZ
except ImportError:
    from datetime import timezone

    UTC_TZ = timezone.utc

__all__ = ["UTC_TZ"]
