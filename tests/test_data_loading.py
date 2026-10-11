"""
Tests for data_loading.add_date_to_time.

Issue #8: LOBSTER message files occasionally contain times with more than 9 fractional digits
(binary float artefacts such as 36754.716797047004). add_date_to_time padded the fraction to 9
digits but never truncated it, so the fraction was read as 716797047004 ns (~12 min) instead of
716797047 ns, the next inter-arrival time went negative and its log became NaN.
"""
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd
import pytest

import data_loading
import eval as lob_eval

DATE = '2012-06-21'

# rows 29330-29335 of the free LOBSTER sample AMZN_2012-06-21_34200000_57600000_message_10.csv,
# the third one is the float artefact quoted in issue #8
AMZN_ROWS = [
    '36754.714497192,1,61443589,19,2234000,1',
    '36754.71674558,1,61443841,100,2233300,1',
    '36754.716797047004,3,61433394,100,2233200,1',
    '36754.724623741,3,61441747,200,2236100,-1',
    '36754.724623741,1,61444766,200,2236200,-1',
    '36754.876166169,3,61443589,19,2234000,1',
]

# every >9-digit time in the free AMZN (level 10) and MSFT (level 5) sample files: 2 and 3 rows
FLOAT_ARTEFACTS = [
    '36754.716797047004',
    '43002.632812672004',
    '34397.878728270996',
    '42734.846057878996',
    '51027.166903456004',
]


def _time_df(times):
    # load_message_df parses the time column to Decimal before add_date_to_time is called
    return pd.DataFrame({'time': [Decimal(t) for t in times]})


def _expected(t):
    # seconds after midnight, truncated (not rounded) to nanoseconds
    ns = int((Decimal(t) * 10**9).to_integral_value(rounding=ROUND_DOWN))
    return pd.Timestamp(DATE) + pd.Timedelta(ns, unit='ns')


def test_add_date_to_time_issue_8_example():
    df = data_loading.add_date_to_time(_time_df(['36754.716797047004']), DATE)
    assert df.time.iloc[0] == pd.Timestamp('2012-06-21 10:12:34.716797047')


@pytest.mark.parametrize('t', FLOAT_ARTEFACTS)
def test_add_date_to_time_truncates_fractional_seconds_to_ns(t):
    df = data_loading.add_date_to_time(_time_df([t]), DATE)
    assert df.time.iloc[0] == _expected(t)


@pytest.mark.parametrize('t', ['34200.017459617', '34200.18960767', '36754.5', '34200.0'])
def test_add_date_to_time_pads_short_fractions(t):
    df = data_loading.add_date_to_time(_time_df([t]), DATE)
    assert df.time.iloc[0] == _expected(t)


def test_loaded_message_times_stay_monotone_with_float_artefact(tmp_path):
    m_f = tmp_path / 'AMZN_2012-06-21_34200000_57600000_message_10.csv'
    m_f.write_text('\n'.join(AMZN_ROWS) + '\n')

    m = data_loading.add_date_to_time(data_loading.load_message_df(str(m_f)), DATE)

    assert m.time.iloc[2] == pd.Timestamp('2012-06-21 10:12:34.716797047')
    assert m.time.is_monotonic_increasing
    # the log_inter_arrival_time score in run_bench.DEFAULT_SCORING_CONFIG
    log_iat = np.log(lob_eval.inter_arrival_time(m).replace({0: 1e-9}).values.astype(float))
    assert np.isfinite(log_iat).all()
