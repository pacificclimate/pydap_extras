import pytest

from pydap_extras.handlers.sql import SQLHandler, parse_queries
from webob.request import Request


def test_data(testconfig):
    handler = SQLHandler(testconfig)
    req = Request.blank("/foo.sql.das")
    assert handler
    assert handler.dataset

    resp = req.get_response(handler)
    assert resp.status == "200 OK"
    assert (
        resp.body.decode("utf-8")
        == """Attributes {
    NC_GLOBAL {
        String name "test dataset";
    }
    a_sequence {
        foo {
            String type "Integer";
        }
    }
}
"""
    )


def test_multiple_selections_for_one_variable():
    selection = [
        "station_observations.time>='1970-01-01 00:00:00'",
        "station_observations.time<='2000-12-31 00:00:00'",
    ]
    out, params = parse_queries(selection, {"time": "obs_time"})

    assert len(out) == 2
    assert len(params.keys()) == 2

    assert out == ["(obs_time >= :0)", "(obs_time <= :1)"]
    assert set(params.values()) == set(["2000-12-31 00:00:00", "1970-01-01 00:00:00"])

def test_basics_data(testconfig_basics):
    handler = SQLHandler(testconfig_basics)
    req = Request.blank("/foo.sql.das")
    assert handler
    assert handler.dataset

    resp = req.get_response(handler)
    assert resp.status == "200 OK"

    # Sensitive to whitespace comparison, be wary if reformatting this string
    assert (
        resp.body.decode("utf-8")
        == """Attributes {
    NC_GLOBAL {
        String name "test_values dataset";
    }
    a_sequence {
        idx {
            String type "Integer";
        }
        temperature {
            String type "Float";
        }
        site {
            String type "String";
        }
    }
}
""")


def test_basics_full_dataset(testconfig_basics):
    handler = SQLHandler(testconfig_basics)
    assert handler
    assert handler.dataset

    seq = handler.dataset['a_sequence']

    assert list(seq) == [
        (10, 15.2, 'Diamond_St'),
        (11, 13.1, 'Blacktail_Loop'),
        (12, 13.3, 'Platinum_St'),
        (13, 12.1, 'Kodiak_Trail')
    ]

def test_basics_reordered_columns(testconfig_basics):
    handler = SQLHandler(testconfig_basics)
    assert handler
    assert handler.dataset

    seq = handler.dataset['a_sequence']

    assert list(seq[['site', 'temperature', 'idx']]) == [
        ('Diamond_St', 15.2, 10),
        ('Blacktail_Loop', 13.1, 11),
        ('Platinum_St', 13.3, 12),
        ('Kodiak_Trail', 12.1, 13)
    ]

def test_basics_single_column(testconfig_basics):
    handler = SQLHandler(testconfig_basics)
    assert handler
    assert handler.dataset

    seq = handler.dataset['a_sequence']

    assert list(seq['temperature']) == [15.2, 13.1, 13.3, 12.1]

def test_basics_filtered_dataset(testconfig_basics):
    handler = SQLHandler(testconfig_basics)
    assert handler
    assert handler.dataset

    seq = handler.dataset['a_sequence']

    assert list(seq[ seq.idx > 10 ]) == [
        (11, 13.1, 'Blacktail_Loop'),
        (12, 13.3, 'Platinum_St'),
        (13, 12.1, 'Kodiak_Trail')
    ]

def test_basics_filtered_single_column(testconfig_basics):
    handler = SQLHandler(testconfig_basics)
    assert handler
    assert handler.dataset

    seq = handler.dataset['a_sequence']

    assert list(seq[ seq.idx > 10 ]['site']) == ['Blacktail_Loop', 'Platinum_St', 'Kodiak_Trail']
    assert list(seq[ seq.idx > 10 ]['temperature']) == [13.1, 13.3, 12.1]

@pytest.mark.xfail
def test_basics_filter_on_unselected_column(testconfig_basics):
    "Filtering won't work on columns that don't exist in the selection"
    handler = SQLHandler(testconfig_basics)
    assert handler
    assert handler.dataset

    seq = handler.dataset['a_sequence']

    assert list(seq['site'][ seq.idx > 10 ]) == ['Blacktail_Loop', 'Platinum_St', 'Kodiak_Trail']
