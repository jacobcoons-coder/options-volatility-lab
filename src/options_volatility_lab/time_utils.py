def time_to_expiration(expiration, valuation_time):
    seconds_remaining = (expiration - valuation_time).total_seconds()

    if seconds_remaining <= 0:
        raise ValueError("Expiration must be after valuation time")

    seconds_per_year = 365.0 * 24 * 60 * 60

    return seconds_remaining / seconds_per_year