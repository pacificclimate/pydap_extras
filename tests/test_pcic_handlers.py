import pytest
from webob.request import Request

from pydap_extras.handlers.pcic import RawPcicSqlHandler, ClimoPcicSqlHandler


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        # Raw
        (
            RawPcicSqlHandler(""),
            [
                (
                    "air-temperature",
                    "degC",
                    "air_temperature",
                    "time: point",
                    "Instantaneous air temperature",
                    "Temperature (Point)",
                )
            ],
        ),
        # Climo
        (
            ClimoPcicSqlHandler(""),
            [
                (
                    "T_mean_Climatology",
                    "celsius",
                    "air_temperature",
                    "t: mean within days t: mean within months t: mean over years",
                    "Climatological mean of monthly mean of mean daily temperature",
                    "Temperature Climatology (Mean)",
                )
            ],
        ),
    ],
)
def test_get_vars(test_db_with_variables, handler, expected):
    assert set(handler.get_vars(1, test_db_with_variables)) == set(expected)


@pytest.mark.parametrize(
    ("net_name", "native_id", "expected"),
    [
        (
            "MoTI",
            "invermere",
            [
                'station_id: "invermere"',
                'station_name: "Invermere"',
                'network: "MoTI"',
                'standard_name: "air_temperature"',
                "latitude: 50.4989",
            ],
        ),
        (
            "MoE",
            "masset",
            [
                'station_id: "masset"',
                'station_name: "Masset"',
                'network: "MoE"',
                'standard_name: "air_pressure"',
                "longitude: -132.14255",
            ],
        ),
    ],
)
def test_create_ini(
    raw_handler, net_name, native_id, expected, monkeypatch, test_db_with_variables
):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, "get_full_query", lambda x, y: "")
    s = raw_handler.create_ini(test_db_with_variables, net_name, native_id)

    for substr in expected:
        assert substr in s


def test_monkey(raw_handler, test_db_with_variables):
    s = raw_handler.get_full_query(1, test_db_with_variables)
    rv = test_db_with_variables.execute(s)


@pytest.mark.parametrize(
    "url",
    ["/EC/913/junk", "/EC/913.sql.html"],  # unparseable path  # non-existant station
)
def test_404s(raw_handler, url):
    req = Request.blank(url)
    resp = req.get_response(raw_handler)

    assert "404" in resp.status


@pytest.mark.poor_unittest
def test_returns_content(raw_handler_get_vars_mock, monkeypatch):
    """This is not a good 'unit' test in that it relies on some intergration with Pydap
    Unfortunately this is the case... this whole _package_ relies heavily on Pydap!
    """
    url = "/MoE/masset.rsql.das"
    req = Request.blank(url)
    resp = req.get_response(raw_handler_get_vars_mock)
    assert resp.status == "200 OK"

    s = """Attributes {
    NC_GLOBAL {
        String name "CRMP/MoE";
        String owner "PCIC";
        String contact "Faron Anslow <fanslow@uvic.ca>";
        Float64 version 0.2;
        String station_id "masset";
        String station_name "Masset";
        String network "MoE";
        Float64 latitude 54.0195;
        Float64 longitude -132.143;
        String elevation "nan";
        String history "Created dynamically by the Pydap SQL handler, the Pydap PCIC SQL handler, and the PCIC/CRMP database";
    }
    station_observations {
        time {
            String axis "T";
            String long_name "observation time";
            String name "time";
            String type "Float64";
            String units "days since 1970-01-01";
        }
    }
}
"""
    for x in s.split("\n"):
        assert x.encode() in resp.body


# Test Corner Cases


def test_create_ini_with_bad_station_id(
    raw_handler, monkeypatch, session_with_duplicate_station
):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, "get_full_query", lambda x, y: "")

    s = raw_handler.create_ini(session_with_duplicate_station, "EC_raw", "1106200")

    assert (
        """station_id: "1106200"
    station_name: ""
    network: "EC_raw"
    latitude: nan
    longitude: nan"""
        in s
    )


def test_create_ini_with_multiple_hist_ids(
    raw_handler, monkeypatch, session_with_multiple_hist_ids_for_one_station
):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, "get_full_query", lambda x, y: "")

    s = raw_handler.create_ini(
        session_with_multiple_hist_ids_for_one_station, "test_network", "some_station"
    )

    assert (
        """station_id: "some_station"
    station_name: "The same station"
    network: "test_network"
    latitude: 49.0
    longitude: -118.0"""
        in s
    )


def test_handles_missing_sdates(
    raw_handler, monkeypatch, session_multiple_hist_ids_null_dates
):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, "get_full_query", lambda x, y: "")

    with pytest.raises(ValueError) as excinfo:
        raw_handler.create_ini(
            session_multiple_hist_ids_null_dates, "test_network", "some_station"
        )

    assert "multiple history entries" in str(excinfo.value)
