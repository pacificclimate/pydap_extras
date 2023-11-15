from collections import namedtuple
from datetime import datetime

from pkg_resources import resource_filename

import pytest
import csv
import os
from tempfile import NamedTemporaryFile

import testing.postgresql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema

from alembic.config import Config
from alembic import command
import alembic.config
import alembic.command
import pycds.alembic

import h5py
import pycds
import numpy as np
from pycds import *
from pydap.model import DatasetType, BaseType, SequenceType
from pydap.handlers.netcdf import NetCDFHandler
from pydap_extras.handlers.pcic import RawPcicSqlHandler
from pydap_extras.handlers.hdf5 import Hdf5Data


@pytest.fixture
def netcdf_handler():
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


@pytest.fixture(scope="session")
def schema_name():
    return pycds.get_schema_name()


@pytest.fixture(scope="function")  # TODO: Can scope be broader?
def database_uri():
    """URI of test PG database"""
    with testing.postgresql.Postgresql() as pg:
        yield pg.url()


@pytest.fixture
def base_engine(database_uri):
    """Plain vanilla database engine, with nothing added."""
    yield create_engine(database_uri)


@pytest.fixture
def testdb(base_engine):
    base_engine.execute("CREATE TABLE mytable (foo INTEGER, bar VARCHAR(50));")
    base_engine.execute("INSERT INTO mytable (foo, bar) VALUES (1, 'hello world');")
    yield base_engine


@pytest.fixture
def testconfig(testdb, database_uri):
    config = f"""database:
  dsn: "{database_uri}"
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
    with NamedTemporaryFile("w") as myconfig:
        myconfig.write(config)
        myconfig.flush()
        yield myconfig.name


def initialize_database(engine, schema_name):
    """Initialize an empty database"""
    # Add role required by PyCDS migrations for privileged operations.
    engine.execute(f"CREATE ROLE {pycds.get_su_role_name()} WITH SUPERUSER NOINHERIT;")
    # Add extensions required by PyCDS.
    engine.execute("CREATE EXTENSION postgis")
    engine.execute("CREATE EXTENSION plpython3u")
    engine.execute("CREATE EXTENSION IF NOT EXISTS citext")
    # Add schema.
    engine.execute(CreateSchema(schema_name))


@pytest.fixture(scope="function")  # TODO: Can scope be broader?
def pycds_engine(base_engine, database_uri, schema_name):
    initialize_database(base_engine, schema_name)
    yield base_engine


@pytest.fixture(scope="package")
def alembic_script_location():
    """
    This fixture extracts the filepath to the installed pycds Alembic content.
    The filepath is typically like
    `/usr/local/lib/python3.6/dist-packages/pycds/alembic`.
    """
    try:
        import importlib_resources

        source = importlib_resources.files(pycds.alembic)
    except ModuleNotFoundError:
        import importlib.resources

        if hasattr(importlib.resources, "files"):
            source = importlib.resources.files(pycds.alembic)
        else:
            with importlib.resources.path("pycds", "alembic") as path:
                source = path

    yield str(source)


def migrate_database(script_location, database_uri, revision="head"):
    """
    Migrate a database to a specified revision using Alembic.
    This requires a privileged role to be added in advance to the database.
    """
    alembic_config = alembic.config.Config()
    alembic_config.set_main_option("script_location", script_location)
    alembic_config.set_main_option("sqlalchemy.url", database_uri)
    alembic.command.upgrade(alembic_config, revision)


@pytest.fixture(scope="function")
def pycds_session(pycds_engine, alembic_script_location, database_uri):
    migrate_database(alembic_script_location, database_uri)
    Session = sessionmaker(bind=pycds_engine)
    with Session() as session:
        yield session


@pytest.fixture(scope="function")
def test_db_with_variables(pycds_session):
    sesh = pycds_session

    nw_moti = Network(
        name="MoTI",
        long_name="Ministry of Transportation and Infrastructure",
        color="000000",
    )
    nw_moe = Network(
        name="MoE",
        long_name="Ministry of Environment",
        color="000000",
    )
    sesh.add_all([nw_moti, nw_moe])

    stn_invermere = Station(native_id="invermere", network=nw_moti)  # id == 1
    stn_masset = Station(native_id="masset", network=nw_moe)  # id == 2
    sesh.add_all([stn_invermere, stn_masset])

    hx_invermere = History(
        station_name="Invermere",
        elevation=1000,
        the_geom="SRID=4326;POINT(-116.0274 50.4989)",
        province="BC",
        freq="1-hourly",
        station=stn_invermere,
    )
    hx_masset = History(
        station_name="Masset",
        elevation=0,
        the_geom="SRID=4326;POINT(-132.14255 54.01950)",
        province="BC",
        freq="1-year",
        station=stn_masset,
    )
    sesh.add_all([hx_invermere, hx_masset])

    var_air_temperature = Variable(
        name="air-temperature",
        unit="degC",
        standard_name="air_temperature",
        cell_method="time: point",
        description="Instantaneous air temperature",
        display_name="Temperature (Point)",
        network=nw_moti,
    )
    var_T_mean_Climatology = Variable(
        name="T_mean_Climatology",
        unit="celsius",
        standard_name="air_temperature",
        cell_method="t: mean within days t: mean within months t: mean over years",
        description="Climatological mean of monthly mean of mean daily temperature",
        display_name="Temperature Climatology (Mean)",
        network=nw_moti,
    )
    var_dew_point = Variable(
        name="dew-point",
        unit="degC",
        standard_name="dew_point_temperature",
        cell_method="time: point",
        display_name="Dew Point Temperature (Mean)",
        network=nw_moti,
    )
    var_BAR_PRESS_HOUR = Variable(
        name="BAR_PRESS_HOUR",
        unit="millibar",
        standard_name="air_pressure",
        cell_method="time:point",
        description="Instantaneous air pressure",
        display_name="Air Pressure (Point)",
        network=nw_moe,
    )
    sesh.add_all(
        [var_air_temperature, var_T_mean_Climatology, var_dew_point, var_BAR_PRESS_HOUR]
    )
    sesh.commit()

    observations = [
        Obs(history=hx_invermere, variable=var_air_temperature, datum=99),
        Obs(history=hx_invermere, variable=var_T_mean_Climatology, datum=99),
        Obs(history=hx_masset, variable=var_BAR_PRESS_HOUR, datum=99),
    ]
    sesh.add_all(observations)
    sesh.commit()

    sesh.execute(VarsPerHistory.refresh())
    sesh.commit()

    yield sesh


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
def session_with_duplicate_station(pycds_session):
    """In 0.0.5, if there's bad data in the database where there's a spurrious station
    without a corresponding history_id, it gets selected first and then the
    metadata request fails. Construct a test database to test for this.
    """
    s = pycds_session

    ecraw = Network(name="EC_raw")
    station0 = Station(native_id="1106200", network=ecraw, histories=[])
    history1 = History()
    station1 = Station(native_id="1106200", network=ecraw, histories=[history1])
    s.add_all([ecraw, station0, station1, history1])
    s.commit()

    yield s


@pytest.fixture(scope="function")
def session_with_multiple_hist_ids_for_one_station(pycds_session):
    s = pycds_session

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
def session_multiple_hist_ids_null_dates(pycds_session):
    s = pycds_session

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
