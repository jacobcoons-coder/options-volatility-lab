from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from options_volatility_lab.time_utils import time_to_expiration

NY_TIME = ZoneInfo("America/New_York")


def test_half_year_to_expiration():
    valuation_time = datetime(
        2026, 1, 1, tzinfo=NY_TIME
    )
    expiration = datetime(
        2026, 7, 2, 12, tzinfo=NY_TIME
    )

    T = time_to_expiration(expiration, valuation_time)

    assert np.isclose(T, 0.5)


def test_one_year_to_expiration():
    valuation_time = datetime(
        2025, 1, 1, tzinfo=NY_TIME
    )
    expiration = datetime(
        2026, 1, 1, tzinfo=NY_TIME
    )

    T = time_to_expiration(expiration, valuation_time)

    assert np.isclose(T, 1.0)


def test_expired_option_raises_error():
    valuation_time = datetime(
        2026, 1, 2, tzinfo=NY_TIME
    )
    expiration = datetime(
        2026, 1, 1, tzinfo=NY_TIME
    )

    with pytest.raises(ValueError):
        time_to_expiration(expiration, valuation_time)