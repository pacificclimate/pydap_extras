import csv
import os
from datetime import datetime
from collections import namedtuple
from tempfile import NamedTemporaryFile

import pytest
import pycds
from pycds import *
from pydap_extras.handlers.pcic import RawPcicSqlHandler

import testing.postgresql
from pycds.util import *
from sqlalchemy.ext.declarative import declarative_base, DeferredReflection
from sqlalchemy import Column, Integer, ForeignKey, String, DateTime, Float, create_engine, event, not_, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import DDL, CreateSchema

TestNetwork = namedtuple('TestNetwork', 'name long_name color')

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


@pytest.fixture(scope='session')
def engine():
    """Test-session-wide database engine"""
    with testing.postgresql.Postgresql() as pg:
        engine = create_engine(pg.url())
        engine.execute("create extension postgis")
        engine.execute(CreateSchema('crmp'))
        pycds.Base.metadata.create_all(bind=engine)
        # sqlalchemy.event.listen(
        #     pycds.weather_anomaly.Base.metadata,
        #     'before_create',
        #     DDL('''
        #         CREATE OR REPLACE FUNCTION crmp.DaysInMonth(date) RETURNS double precision AS
        #         $$
        #             SELECT EXTRACT(DAY FROM CAST(date_trunc('month', $1) + interval '1 month' - interval '1 day'
        #             as timestamp));
        #         $$ LANGUAGE sql;
        #     ''')
        # )
        # pycds.weather_anomaly.Base.metadata.create_all(bind=engine)
        yield engine


@pytest.fixture(scope='function')
def session(engine):
    """Single-test database session. All session actions are rolled back on teardown"""
    session = sessionmaker(bind=engine)()
    # Default search path is `"$user", public`. Need to reset that to search crmp (for our db/orm content) and
    # public (for postgis functions)
    session.execute('SET search_path TO crmp, public')
    # print('\nsearch_path', [r for r in session.execute('SHOW search_path')])
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope='module')
def mod_blank_postgis_session():
    with testing.postgresql.Postgresql() as pg:
        engine = create_engine(pg.url())
        engine.execute("create extension postgis")
        engine.execute(CreateSchema('crmp'))
        sesh = sessionmaker(bind=engine)()
        yield sesh


@pytest.fixture(scope='module')
def mod_empty_database_session(mod_blank_postgis_session):
    sesh = mod_blank_postgis_session
    engine = sesh.get_bind()
    pycds.Base.metadata.create_all(bind=engine)
    pycds.weather_anomaly.Base.metadata.create_all(bind=engine)
    yield sesh


@pytest.yield_fixture(scope='function')
def blank_postgis_session():
    with testing.postgresql.Postgresql() as pg:
        engine = create_engine(pg.url())
        engine.execute("create extension postgis")
        engine.execute(CreateSchema('crmp'))
        sesh = sessionmaker(bind=engine)()

        yield sesh


@pytest.fixture(scope="function")
def test_session(blank_postgis_session):

    engine = blank_postgis_session.get_bind()
    pycds.Base.metadata.create_all(bind=engine)

    # Make sure spatial extensions are loaded for each connection, not just the current session
    # https://groups.google.com/d/msg/sqlalchemy/eDpJ-yZEnqU/_XJ4Pmd712QJ
    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_rec):
        dbapi_connection.enable_load_extension(True)
        dbapi_connection.execute("select load_extension('mod_spatialite')")

    yield blank_postgis_session

@pytest.fixture(scope="function")
def test_db_with_variables(test_session):
    sesh = test_session

    moti = Network(**TestNetwork('MoTI', 'Ministry of Transportation and Infrastructure', '000000')._asdict())
    moe = Network(**TestNetwork('MoE', 'Ministry of Environment', '000000')._asdict())
    sesh.add_all([moti, moe])

    histories = [History(station_name="Invermere",
                         elevation=1000,
                         the_geom='SRID=4326;POINT(-116.0274 50.4989)',
                         province='BC',
                         freq='1-hourly'),
                 History(station_name="Masset",
                         elevation=0,
                         the_geom='SRID=4326;POINT(-132.14255 54.01950)',
                         province='BC',
                         freq='1-year'),
                 ]
    
    invermere = Station(native_id='invermere', network=moti, histories=[histories[0]])
    masset = Station(native_id='masset', network=moe, histories=[histories[1]])
    sesh.add_all([invermere, masset])
    
    variables = [Variable(name='air-temperature',
                          unit='degC',
                          standard_name='air_temperature',
                          cell_method='time: point',
                          description='Instantaneous air temperature',
                          display_name='Temperature (Point)',
                          network=moti),
                 Variable(name='T_mean_Climatology',
                          unit='celsius',
                          standard_name='air_temperature',
                          cell_method='t: mean within days t: mean within months t: mean over years',
                          description='Climatological mean of monthly mean of mean daily temperature',
                          display_name='Temperature Climatology (Mean)',
                          network=moti),
                 Variable(name='dew-point',
                          unit='degC',
                          standard_name='dew_point_temperature',
                          cell_method='time: point',
                          display_name='Dew Point Temperature (Mean)',
                          network=moti),
                 Variable(name='BAR_PRESS_HOUR',
                          unit='millibar',
                          standard_name='air_pressure',
                          cell_method='time:point',
                          description='Instantaneous air pressure',
                          display_name='Air Pressure (Point)',
                          network=moe),
                ]
    sesh.add_all(variables)
    sesh.commit()

    vars_per_history = [VarsPerHistory(history_id=histories[0].id, vars_id=variables[0].id),
                        VarsPerHistory(history_id=histories[1].id, vars_id=variables[-1].id)]
    sesh.add_all(vars_per_history)

    sesh.commit()
    
    yield sesh

@pytest.fixture(scope="module")
def conn_params(mod_blank_postgis_session):
    return mod_blank_postgis_session.get_bind()

ObsTuple = namedtuple('ObsTuple', 'time datum history variable')
def ObsMaker(*args):
    return Obs(**ObsTuple(*args)._asdict())

@pytest.fixture(scope="function")
def test_db_with_met_obs(test_db_with_variables):
    sesh = test_db_with_variables

    hist = sesh.query(History).filter(History.station_name == "Masset").first()
    var = hist.station.network.variables[0]

    timeseries = [(datetime(2015, 1, 1, 10), 1, hist, var),
                  (datetime(2015, 1, 1, 11), 2, hist, var),
                  (datetime(2015, 1, 1, 12), 2, hist, var),
                  (datetime(2015, 1, 1, 13), 1, hist, var)]

    for obs in timeseries:
        sesh.add(ObsMaker(*obs))

    sesh.commit()
    yield sesh

@pytest.fixture(scope="function")
def session_with_duplicate_station(test_session):
    '''In 0.0.5, if there's bad data in the database where there's a spurrious station
       without a corresponding history_id, it gets selected first and then the
       metadata request fails. Construct a test database to test for this.
    '''
    s = test_session

    ecraw = Network(name='EC_raw')
    station0 = Station(native_id='1106200', network=ecraw, histories=[])
    history1 = History()
    station1 = Station(native_id='1106200', network=ecraw, histories=[history1])
    s.add_all([ecraw, station0, station1, history1])
    s.commit()

    yield s


@pytest.fixture(scope="function")
def session_with_multiple_hist_ids_for_one_station(test_session):
    s = test_session

    net = Network(name='test_network')
    history0 = History(station_name='Some station', elevation=999,
                             sdate = datetime(1880, 1, 1),
                             edate = datetime(2000, 1, 1))
    # Empty end date... i.e. and "active station"
    history1 = History(station_name='The same station', elevation=999,
                             sdate = datetime(2000, 1, 2),
                             the_geom = 'SRID=4326;POINT(-118 49)')
    station0 = Station(native_id='some_station', network=net, histories=[history0, history1])
    s.add(station0)
    s.commit()

    yield s


@pytest.fixture(scope="function")
def session_multiple_hist_ids_null_dates(test_session):
    s = test_session

    net = Network(name='test_network')
    history0 = History(station_name='Some station', elevation=999)
    history1 = History(station_name='The same station', elevation=999)
    station0 = Station(native_id='some_station', network=net, histories=[history0, history1])
    s.add(station0)
    s.commit()

    yield s


@pytest.fixture(scope="function")
def raw_handler(monkeypatch, conn_params, test_session):
    handler = RawPcicSqlHandler(conn_params, test_session)

    def my_get_full_query(self, stn_id, sesh):
        return sql_station_table(sesh, stn_id)
    monkeypatch.setattr(RawPcicSqlHandler, 'get_full_query', my_get_full_query)

    return handler


# http://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query
def compile_query(statement, bind=None):
    """
    print a query, with values filled in
    for debugging purposes *only*
    for security, you should always separate queries from their values
    please also note that this function is quite slow
    """
    import sqlalchemy.orm
    if isinstance(statement, sqlalchemy.orm.Query):
        if bind is None:
            bind = statement.session.get_bind(
                statement.column_descriptions
            )
            statement = statement.statement
        elif bind is None:
            bind = statement.bind

        dialect = bind.dialect
        compiler = statement._compiler(dialect)

        class LiteralCompiler(compiler.__class__):

            def visit_bindparam(
                    self, bindparam, within_columns_clause=False,
                    literal_binds=False, **kwargs
            ):
                return super(LiteralCompiler, self).render_literal_bindparam(
                    bindparam, within_columns_clause=within_columns_clause,
                    literal_binds=literal_binds, **kwargs
                )

    compiler = LiteralCompiler(dialect, statement)
    return compiler.process(statement)


def orm_station_table(sesh, stn_id, raw=True):
    '''Construct a 'station table' i.e. a table such that each row
       corresponds to a single timestep and each column corresponds to
       a separate variable or flag
       :param sesh: sqlalchemy session
       :param stn_id: id corresponding to meta_station.station_id or Station.id
       :type stn_id: int
       :param raw: Should this query be for raw observations? Setting this to False will fetch climatologies.
       :type raw: bool
       :rtype: :py:class:`sqlalchemy.orm.query.Query`
    '''
    if raw:
        raw_filter = not_(and_(ObsWithFlags.cell_method.like(
            '%within%'), ObsWithFlags.cell_method.like('%over%')))
    else:
        raw_filter = or_(ObsWithFlags.cell_method.like(
            '%within%'), ObsWithFlags.cell_method.like('%over%'))

    # Get all of the variables for which observations exist
    # and iterate over them
    vars_ = sesh.query(ObsWithFlags.vars_id, ObsWithFlags.net_var_name)\
        .filter(ObsWithFlags.station_id == stn_id).filter(raw_filter)\
        .distinct().order_by(ObsWithFlags.vars_id)

    # Start with all of the times for which observations exist
    # and then use this as a basis for a left join
    # (sqlite doesn't support full outer joins
    times = sesh.query(ObsWithFlags.obs_time.label('flag_time'))\
        .filter(ObsWithFlags.station_id == stn_id)\
        .order_by(ObsWithFlags.obs_time).distinct()
    stmt = times.subquery()

    for vars_id, var_name in vars_.all():

        # Construct a query for all values of this variable
        right = sesh.query(
            ObsWithFlags.obs_time.label('obs_time'),
            ObsWithFlags.datum.label(var_name),
            ObsWithFlags.flag_name.label(var_name + '_flag')
        ).filter(ObsWithFlags.vars_id == vars_id)\
            .filter(ObsWithFlags.station_id == stn_id).subquery()

        # Then join it to the query we're already building
        join_query = sesh.query(stmt, right).outerjoin(
            right, stmt.c.obs_time == right.c.obs_time)

        stmt = join_query.subquery()

    return sesh.query(stmt)


def sql_station_table(sesh, stn_id):
    return compile_query(orm_station_table(sesh, stn_id))


class ObsWithFlags(Base):
    '''This class maps to a convenience view that is used to construct a
    table of flagged observations; i.e. one row per observation with
    additional columns for each attached flag.
    '''
    __tablename__ = 'obs_with_flags'
    vars_id = Column(Integer, ForeignKey('meta_vars.vars_id'))
    network_id = Column(Integer, ForeignKey('meta_network.network_id'))
    unit = Column(String)
    standard_name = Column(String)
    cell_method = Column(String)
    net_var_name = Column(String)
    obs_raw_id = Column(Integer, ForeignKey(
        'obs_raw.obs_raw_id'), primary_key=True)
    station_id = Column(Integer, ForeignKey('meta_station.station_id'))
    obs_time = Column(DateTime)
    mod_time = Column(DateTime)
    datum = Column(Float)
    native_flag_id = Column(Integer, ForeignKey(
        'meta_native_flag.native_flag_id'))
    flag_name = Column(String)
    description = Column(String)
    flag_value = Column(String)