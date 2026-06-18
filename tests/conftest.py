"""Shared fixtures and synthetic data for the test suite.

The synthetic frames mirror the real extract schemas (see db.py) but are small
enough that every expected value can be computed by hand in the tests.
"""

import polars as pl
import pytest


@pytest.fixture
def airports() -> pl.DataFrame:
    """Minimal AIRPORTS reference: two origins, two destinations."""
    return pl.DataFrame(
        {
            "iata_code": ["A1", "A2", "B1", "B2"],
            "continent": ["Europe", "Asia", "Europe", "Asia"],
            "city": ["CityA", "CityB", "CityC", "CityD"],
        }
    )


@pytest.fixture
def revenue_raw() -> pl.DataFrame:
    """Revenue at (route, year, class) grain, matching extract_revenue output.

    R1 (A1→B1, Europe origin): 2020 and 2021, Economy.
    R2 (A2→B2, Asia origin): 2020, Business.
    """
    return pl.DataFrame(
        {
            "route_code": ["R1", "R1", "R2"],
            "origin": ["A1", "A1", "A2"],
            "destination": ["B1", "B1", "B2"],
            "yr": [2020, 2021, 2020],
            "class": ["E", "E", "B"],
            "ticket_count": [100, 100, 50],
            "net_revenue": [1000.0, 1200.0, 2000.0],
            "total_taxes": [200.0, 200.0, 500.0],
            "gross_revenue": [1200.0, 1400.0, 2500.0],
        }
    )


@pytest.fixture
def fuel() -> pl.DataFrame:
    """Fuel at (route, model, year) grain, matching extract_fuel output.

    R1 flown by M1 (2020, 2021) and M2 (2020); R2 flown by M2 (2020).
    total_fuel_gallons is set directly so cost math is exact.
    """
    return pl.DataFrame(
        {
            "route_code": ["R1", "R1", "R1", "R2"],
            "model": ["M1", "M2", "M1", "M2"],
            "yr": [2020, 2020, 2021, 2020],
            "flight_minutes": [60, 60, 60, 120],
            "fuel_gallons_hour": [100.0, 100.0, 100.0, 100.0],
            "flights_operated": [10, 10, 10, 5],
            "total_fuel_gallons": [100.0, 100.0, 100.0, 200.0],
        }
    )


@pytest.fixture
def revenue_enriched(revenue_raw, airports) -> pl.DataFrame:
    """Revenue with continent/city joined, as the dashboard uses it."""
    from analysis import enrich_revenue_with_airports

    return enrich_revenue_with_airports(revenue_raw, airports)
