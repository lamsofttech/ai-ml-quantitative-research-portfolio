import numpy as np
import pytest

from quant_mc_options.black_scholes import bs_greeks, bs_price


def test_matches_textbook_reference_value():
    # Hull, "Options, Futures, and Other Derivatives": S0=42, K=40, r=10%,
    # sigma=20%, T=0.5y -> European call ~= 4.76.
    price = bs_price(42.0, 40.0, 0.10, 0.20, 0.5, option_type="call")
    assert price == pytest.approx(4.76, abs=0.01)


@pytest.mark.parametrize(
    ("s0", "k", "r", "sigma", "maturity", "q"),
    [(100.0, 100.0, 0.03, 0.2, 1.0, 0.0), (80.0, 90.0, 0.05, 0.35, 2.0, 0.02)],
)
def test_put_call_parity_holds(s0, k, r, sigma, maturity, q):
    call = bs_price(s0, k, r, sigma, maturity, "call", q)
    put = bs_price(s0, k, r, sigma, maturity, "put", q)
    parity_rhs = s0 * np.exp(-q * maturity) - k * np.exp(-r * maturity)
    assert (call - put) == pytest.approx(parity_rhs, abs=1e-9)


def test_deep_itm_call_approaches_intrinsic_forward_value():
    price = bs_price(1000.0, 10.0, 0.05, 0.2, 1.0, option_type="call")
    intrinsic = 1000.0 - 10.0 * np.exp(-0.05)
    assert price == pytest.approx(intrinsic, rel=1e-4)


def test_deep_otm_call_is_near_zero():
    price = bs_price(10.0, 1000.0, 0.05, 0.2, 1.0, option_type="call")
    assert price == pytest.approx(0.0, abs=1e-6)


def test_delta_matches_finite_difference_of_price():
    s0, k, r, sigma, maturity = 100.0, 100.0, 0.03, 0.2, 1.0
    eps = 1e-4
    numeric_delta = (
        bs_price(s0 + eps, k, r, sigma, maturity, "call")
        - bs_price(s0 - eps, k, r, sigma, maturity, "call")
    ) / (2 * eps)
    assert bs_greeks(s0, k, r, sigma, maturity, "call").delta == pytest.approx(
        numeric_delta, abs=1e-4
    )


def test_vega_is_identical_for_call_and_put():
    greeks = dict(s0=100.0, k=95.0, r=0.02, sigma=0.25, maturity=0.75)
    call_vega = bs_greeks(**greeks, option_type="call").vega
    put_vega = bs_greeks(**greeks, option_type="put").vega
    assert call_vega == pytest.approx(put_vega, rel=1e-10)


@pytest.mark.parametrize(
    ("s0", "k", "sigma", "maturity"),
    [
        (0.0, 100.0, 0.2, 1.0),
        (100.0, 0.0, 0.2, 1.0),
        (100.0, 100.0, -0.1, 1.0),
        (100.0, 100.0, 0.2, 0.0),
    ],
)
def test_rejects_invalid_inputs(s0, k, sigma, maturity):
    with pytest.raises(ValueError):
        bs_price(s0, k, 0.05, sigma, maturity, "call")


def test_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.05, 0.2, 1.0, option_type="exotic")
