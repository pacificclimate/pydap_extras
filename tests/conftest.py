import pytest
import csv
import os
from datetime import datetime
from collections import namedtuple
from tempfile import NamedTemporaryFile
from pkg_resources import resource_filename

import testing.postgresql
from pycds.util import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema

import h5py
import pycds
import numpy as np
from pycds import *
from netCDF4 import Dataset
from pydap.model import DatasetType, BaseType, SequenceType
from pydap.handlers.netcdf import NetCDFHandler
from pydap_extras.handlers.pcic import RawPcicSqlHandler
from pydap_extras.handlers.hdf5 import Hdf5Data


TestNetwork = namedtuple("TestNetwork", "name long_name color")


@pytest.fixture
def handler():
    fname = resource_filename("tests", "data/tiny_bccaq2_wo_recvars.nc")
    return NetCDFHandler(fname)


@pytest.fixture(scope="session")
def simple_data():
    data = [
        (10, 15.2, "Diamond_St"),
        (11, 13.1, "Blacktail_Loop"),
        (12, 13.3, "Platinum_St"),
        (13, 12.1, "Kodiak_Trail"),
    ]
    return data


@pytest.fixture(scope="session")
def simple_data_file(tmpdir_factory, simple_data):
    temp_file = str(tmpdir_factory.mktemp("data").join("simple_data.csv"))
    with open(temp_file, "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["index", "temperature", "site"])
        for row in simple_data:
            writer.writerow(row)
    return temp_file


@pytest.fixture
def simple_dataset():
    dataset = DatasetType("VerySimpleSequence")
    dataset["sequence"] = SequenceType("sequence")
    dataset["sequence"]["byte"] = BaseType("byte")
    dataset["sequence"]["int"] = BaseType("int")
    dataset["sequence"]["float"] = BaseType("float")
    dataset["sequence"].data = np.array(
        [
            (0, 1, 10.0),
            (1, 2, 20.0),
            (2, 3, 30.0),
            (3, 4, 40.0),
            (4, 5, 50.0),
            (5, 6, 60.0),
            (6, 7, 70.0),
            (7, 8, 80.0),
        ],
        dtype=[("byte", np.byte), ("int", np.int32), ("float", np.float32)],
    )

    return dataset


@pytest.fixture
def testconfig(testdb, request):
    config = f"""database:
  dsn: "sqlite:///{testdb}"
  id: "mytable"
  table: "mytable"

dataset:
  NC_GLOBAL:
    name: "test dataset"

sequence:
  name: "a_sequence"

foo:
  col: "foo"
  type: Integer
"""

    with NamedTemporaryFile("w", delete=False) as myconfig:
        myconfig.write(config)
        fname = myconfig.name

    def fin():
        os.remove(fname)

    request.addfinalizer(fin)
    return fname


@pytest.fixture
def testdb(request):
    with NamedTemporaryFile("w", delete=False) as f:
        engine = create_engine("sqlite:///" + f.name, echo=True)
        engine.execute("CREATE TABLE mytable (foo INTEGER, bar VARCHAR(50));")
        engine.execute("INSERT INTO mytable (foo, bar) VALUES (1, 'hello world');")
        fname = f.name

    def fin():
        os.remove(fname)

    request.addfinalizer(fin)
    return fname


@pytest.fixture(scope="session")
def engine():
    """Test-session-wide database engine"""
    with testing.postgresql.Postgresql() as pg:
        engine = create_engine(pg.url())
        engine.execute("create extension postgis")
        engine.execute(CreateSchema("crmp"))
        pycds.Base.metadata.create_all(bind=engine)
        yield engine


@pytest.fixture(scope="function")
def session(engine):
    """Single-test database session. All session actions are rolled back on teardown"""
    session = sessionmaker(bind=engine)()
    # Default search path is `"$user", public`. Need to reset that to search crmp (for our db/orm content) and
    # public (for postgis functions)
    session.execute("SET search_path TO crmp, public")
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="module")
def mod_blank_postgis_session():
    with testing.postgresql.Postgresql() as pg:
        engine = create_engine(pg.url())
        engine.execute("create extension postgis")
        engine.execute(CreateSchema("crmp"))
        sesh = sessionmaker(bind=engine)()
        yield sesh


@pytest.fixture(scope="module")
def mod_empty_database_session(mod_blank_postgis_session):
    sesh = mod_blank_postgis_session
    engine = sesh.get_bind()
    pycds.Base.metadata.create_all(bind=engine)
    pycds.weather_anomaly.Base.metadata.create_all(bind=engine)
    yield sesh


@pytest.fixture(scope="function")
def blank_postgis_session():
    with testing.postgresql.Postgresql() as pg:
        engine = create_engine(pg.url())
        engine.execute("create extension postgis")
        engine.execute(CreateSchema("crmp"))
        sesh = sessionmaker(bind=engine)()

        yield sesh


@pytest.fixture(scope="function")
def test_session(blank_postgis_session):

    engine = blank_postgis_session.get_bind()
    pycds.Base.metadata.create_all(bind=engine)

    yield blank_postgis_session


@pytest.fixture(scope="function")
def test_db_with_variables(test_session):
    sesh = test_session

    moti = Network(
        **TestNetwork(
            "MoTI", "Ministry of Transportation and Infrastructure", "000000"
        )._asdict()
    )
    moe = Network(**TestNetwork("MoE", "Ministry of Environment", "000000")._asdict())
    sesh.add_all([moti, moe])

    histories = [
        History(
            station_name="Invermere",
            elevation=1000,
            the_geom="SRID=4326;POINT(-116.0274 50.4989)",
            province="BC",
            freq="1-hourly",
        ),
        History(
            station_name="Masset",
            elevation=0,
            the_geom="SRID=4326;POINT(-132.14255 54.01950)",
            province="BC",
            freq="1-year",
        ),
    ]

    invermere = Station(native_id="invermere", network=moti, histories=[histories[0]])
    masset = Station(native_id="masset", network=moe, histories=[histories[1]])
    sesh.add_all([invermere, masset])

    variables = [
        Variable(
            name="air-temperature",
            unit="degC",
            standard_name="air_temperature",
            cell_method="time: point",
            description="Instantaneous air temperature",
            display_name="Temperature (Point)",
            network=moti,
        ),
        Variable(
            name="T_mean_Climatology",
            unit="celsius",
            standard_name="air_temperature",
            cell_method="t: mean within days t: mean within months t: mean over years",
            description="Climatological mean of monthly mean of mean daily temperature",
            display_name="Temperature Climatology (Mean)",
            network=moti,
        ),
        Variable(
            name="dew-point",
            unit="degC",
            standard_name="dew_point_temperature",
            cell_method="time: point",
            display_name="Dew Point Temperature (Mean)",
            network=moti,
        ),
        Variable(
            name="BAR_PRESS_HOUR",
            unit="millibar",
            standard_name="air_pressure",
            cell_method="time:point",
            description="Instantaneous air pressure",
            display_name="Air Pressure (Point)",
            network=moe,
        ),
    ]
    sesh.add_all(variables)
    sesh.commit()

    vars_per_history = [
        VarsPerHistory(history_id=histories[0].id, vars_id=variables[0].id),
        VarsPerHistory(history_id=histories[1].id, vars_id=variables[-1].id),
    ]
    sesh.add_all(vars_per_history)

    sesh.commit()

    yield sesh


@pytest.fixture(scope="module")
def conn_params(mod_blank_postgis_session):
    mod_blank_postgis_session.get_bind()
    return mod_blank_postgis_session.get_bind()


ObsTuple = namedtuple("ObsTuple", "time datum history variable")


def ObsMaker(*args):
    return Obs(**ObsTuple(*args)._asdict())


@pytest.fixture(scope="function")
def test_db_with_met_obs(test_db_with_variables):
    sesh = test_db_with_variables

    hist = sesh.query(History).filter(History.station_name == "Masset").first()
    var = hist.station.network.variables[0]

    timeseries = [
        (datetime(2015, 1, 1, 10), 1, hist, var),
        (datetime(2015, 1, 1, 11), 2, hist, var),
        (datetime(2015, 1, 1, 12), 2, hist, var),
        (datetime(2015, 1, 1, 13), 1, hist, var),
    ]

    for obs in timeseries:
        sesh.add(ObsMaker(*obs))

    sesh.commit()
    yield sesh


@pytest.fixture(scope="function")
def session_with_duplicate_station(test_session):
    """In 0.0.5, if there's bad data in the database where there's a spurrious station
    without a corresponding history_id, it gets selected first and then the
    metadata request fails. Construct a test database to test for this.
    """
    s = test_session

    ecraw = Network(name="EC_raw")
    station0 = Station(native_id="1106200", network=ecraw, histories=[])
    history1 = History()
    station1 = Station(native_id="1106200", network=ecraw, histories=[history1])
    s.add_all([ecraw, station0, station1, history1])
    s.commit()

    yield s


@pytest.fixture(scope="function")
def session_with_multiple_hist_ids_for_one_station(test_session):
    s = test_session

    net = Network(name="test_network")
    history0 = History(
        station_name="Some station",
        elevation=999,
        sdate=datetime(1880, 1, 1),
        edate=datetime(2000, 1, 1),
    )
    # Empty end date... i.e. and "active station"
    history1 = History(
        station_name="The same station",
        elevation=999,
        sdate=datetime(2000, 1, 2),
        the_geom="SRID=4326;POINT(-118 49)",
    )
    station0 = Station(
        native_id="some_station", network=net, histories=[history0, history1]
    )
    s.add(station0)
    s.commit()

    yield s


@pytest.fixture(scope="function")
def session_multiple_hist_ids_null_dates(test_session):
    s = test_session

    net = Network(name="test_network")
    history0 = History(station_name="Some station", elevation=999)
    history1 = History(station_name="The same station", elevation=999)
    station0 = Station(
        native_id="some_station", network=net, histories=[history0, history1]
    )
    s.add(station0)
    s.commit()

    yield s


@pytest.fixture(scope="function")
def raw_handler(monkeypatch, test_db_with_met_obs):
    conn_params = test_db_with_met_obs.get_bind()
    handler = RawPcicSqlHandler(conn_params, test_db_with_met_obs)

    def my_get_full_query(self, stn_id, sesh):
        return "SELECT * FROM crmp.obs_raw"

    monkeypatch.setattr(RawPcicSqlHandler, "get_full_query", my_get_full_query)

    return handler


@pytest.fixture(scope="function")
def raw_handler_get_vars_mock(monkeypatch, test_db_with_met_obs):
    conn_params = test_db_with_met_obs.get_bind()
    handler = RawPcicSqlHandler(conn_params, test_db_with_met_obs)

    def my_get_full_query(self, stn_id, sesh):
        return "SELECT * FROM crmp.obs_raw"

    monkeypatch.setattr(RawPcicSqlHandler, "get_full_query", my_get_full_query)

    def my_get_vars(self, stn_id, sesh):
        return ("") * 6

    monkeypatch.setattr(RawPcicSqlHandler, "get_vars", my_get_vars)

    return handler


test_h5 = resource_filename("tests", "data/test.h5")


@pytest.fixture(scope="function", params=["/tasmax", "/tasmin", "/pr"])
def hdf5data_instance_3d(request):
    f = h5py.File(test_h5, "r")
    dst = f[request.param]
    return Hdf5Data(dst)


@pytest.fixture(scope="module", params=["/lat", "/lon", "/time"])
def hdf5data_instance_1d(request):
    f = h5py.File(test_h5, "r")
    dst = f[request.param]
    return Hdf5Data(dst)


# _All_ the variables should be iterable
@pytest.fixture(
    scope="module", params=["/tasmax", "/tasmin", "/pr", "/lat", "/lon", "/time"]
)
def hdf5data_iterable(request):
    f = h5py.File(test_h5, "r")
    dst = f[request.param]
    return Hdf5Data(dst)


@pytest.fixture(scope="function")
def hdf5_dst(request):
    f = NamedTemporaryFile()
    hf = h5py.File(f, "w", driver="fileobj", backing_store=False)
    group = hf.create_group("foo")
    dst = group.create_dataset("bar", (10, 10, 10), "=f8", maxshape=(None, 10, 10))
    dst[:, :, :] = np.random.rand(10, 10, 10)

    def fin():
        hf.close()
        os.remove(f.name)

    request.addfinalizer(fin)

    return dst
