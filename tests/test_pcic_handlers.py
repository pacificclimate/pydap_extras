import pytest
from collections import namedtuple
from webob.request import Request
from sqlalchemy.orm import sessionmaker

from pycds import Network, Station
from pydap_extras.handlers.sql import Engines
from pydap_extras.handlers.pcic import PcicSqlHandler, RawPcicSqlHandler, ClimoPcicSqlHandler


@pytest.mark.parametrize(('input', 'expected'), [
    # Raw
    (RawPcicSqlHandler(''), [('air-temperature', 'degC', 'air_temperature', 'time: point', 'Instantaneous air temperature', 'Temperature (Point)'),]),
    # Climo
    (ClimoPcicSqlHandler(''),[('T_mean_Climatology',
                                'celsius',
                                'air_temperature',
                                't: mean within days t: mean within months t: mean over years',
                                'Climatological mean of monthly mean of mean daily temperature',
                                'Temperature Climatology (Mean)'),
                                ])])

def test_get_vars(test_db_with_variables, input, expected):
    assert set(input.get_vars(1, test_db_with_variables)) == set(expected)

@pytest.mark.parametrize(('net_name', 'native_id', 'expected'), [
                         ('MoTI','invermere',
                          ['station_id: "invermere"',
                           'station_name: "Invermere"',
                           'network: "MoTI"',
                           'standard_name: "air_temperature"',
                           'latitude: 50.4989',
                           ]
                          ),
                         ('MoE','masset',
                          ['station_id: "masset"',
                           'station_name: "Masset"',
                           'network: "MoE"',
                           'standard_name: "air_pressure"',
                           'longitude: -132.14255',
                           ]
                         )])
def test_create_ini(raw_handler, net_name, native_id, expected, monkeypatch, test_db_with_variables):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, 'get_full_query', lambda x, y: '' )
    s = raw_handler.create_ini(test_db_with_variables, net_name, native_id)

    for substr in expected:
        assert substr in s

def test_monkey(raw_handler, test_db_with_variables):
    s = raw_handler.get_full_query(1, test_db_with_variables)
    rv = test_db_with_variables.execute(s)

@pytest.mark.parametrize('url', [
    '/EC/913/junk', # unparseable path
    '/EC/913.sql.html' # non-existant station
    ])
def test_404s(raw_handler, url):
    req = Request.blank(url)
    resp = req.get_response(raw_handler)

    assert '404' in resp.status

@pytest.mark.poor_unittest
def test_returns_content(raw_handler):
    '''This is not a good 'unit' test in that it relies on some intergration with Pydap
       Unfortunately this is the case... this whole _package_ relies heavily on Pydap!
    '''
    url = '/moe/masset.rsql.das'
    req = Request.blank(url)
    resp = req.get_response(raw_handler)
    assert resp.status == '200 OK'

    s = '''Attributes {
    NC_GLOBAL {
        String network "EC";
        String contact "Faron Anslow <fanslow@uvic.ca>";
        String name "CRMP/EC";
        String owner "PCIC";
        Float64 version 0.2;
        String station_id "1106200";
        Float64 latitude 49.3303;
        String station_name "POINT ATKINSON";
        Float64 longitude -123.265;
        String history "Created dynamically by the Pydap SQL handler, the Pydap PCIC SQL handler, and the PCIC/CRMP database";
    }
    station_observations {
        MAX_TEMP {
            String name "MAX_TEMP";
            Int32 missing_value -9999;
            String axis "Y";
            String long_name "Maximum daily temperature";
            String standard_name "air_temperature";
            String display_name "Temperature (Max.)";
            String units "celsius";
            String type "Float64";
            String cell_method "time: maximum";
        }
        time {
            String long_name "observation time";
            String type "String";
            String name "time";
            String axis "T";
            String units "days since 1970-01-01";
        }
    }
}'''
    assert all([x in resp.body for x in s.split('\n')])

@pytest.mark.poor_unittest
def test_returns_html_content(raw_handler):
    '''This is not a good 'unit' test in that it relies on some intergration with Pydap
       Unfortunately this is the case... this whole _package_ relies heavily on Pydap!
    '''
    url = '/EC/1106200.rsql.html'
    req = Request.blank(url)
    resp = req.get_response(raw_handler)
    assert resp.status == '200 OK'

# Test Corner Cases

def test_create_ini_with_bad_station_id(raw_handler, monkeypatch, session_with_duplicate_station):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, 'get_full_query', lambda x, y: '' )

    s = raw_handler.create_ini(session_with_duplicate_station, 'EC_raw', '1106200')

    assert '''station_id: "1106200"
    station_name: ""
    network: "EC_raw"
    latitude: nan
    longitude: nan''' in s

def test_create_ini_with_multiple_hist_ids(raw_handler, monkeypatch, session_with_multiple_hist_ids_for_one_station):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, 'get_full_query', lambda x, y: '' )

    s = raw_handler.create_ini(session_with_multiple_hist_ids_for_one_station, 'test_network', 'some_station')

    assert '''station_id: "some_station"
    station_name: "The same station"
    network: "test_network"
    latitude: 49.000000
    longitude: -118.000000''' in s

def test_handles_missing_sdates(raw_handler, monkeypatch, session_multiple_hist_ids_null_dates):
    # get_full_query is not important for this test
    monkeypatch.setattr(raw_handler, 'get_full_query', lambda x, y: '' )

    with pytest.raises(ValueError) as excinfo:
        raw_handler.create_ini(session_multiple_hist_ids_null_dates, 'test_network', 'some_station')

    assert "multiple history entries" in str(excinfo.value)