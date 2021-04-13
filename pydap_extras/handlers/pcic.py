"""This module provides an Pydap handler which reads in-situ observations out of the BC Provincial Climate Data Set. It is implemented as a subclass of :class:`pydap.handlers.sql.Handler` (the Pydap SQL handlers). Since the Pydap SQL handler is written to use an on-disk config file for each dataset, this handlers generates the config file dynamically in memory and then uses it to instantiate the base class.
The handler will configure a different dataset for each station based on the file path of the request. In general the file path is assumed to be ::
.../(raw|climo)/[network_name]/[native_id]/
Each dataset will contain a variety of global attributes such as the station and network names, latitude and longitide of the station and some contact information. Each dataset will contain one sequence named ``station_observations`` and some number of variables (including time) attached to that sequence. Each variable will be attributed with its name, long_name, CF standard_name, CF cell_method and the units.
"""

import os
import sys
import re
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import logging

from sqlalchemy import or_, not_, func
from sqlalchemy.orm import sessionmaker
from paste.httpexceptions import HTTPNotFound
from geoalchemy2.functions import ST_X, ST_Y

from pydap.wsgi.app import DapServer
from pydap_extras.handlers.sql import SQLHandler, Engines
from pycds import *

logger = logging.getLogger(__name__)

# From http://docs.sqlalchemy.org/en/rel_0_9/orm/session.html#session-faq-whentocreate
@contextmanager
def session_scope(dsn):
    """Provide a transactional scope around a series of operations."""
    factory = sessionmaker(bind=Engines[dsn])
    session = factory()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


class PcicSqlHandler(object):
    """A Pydap handler which reads in-situ observations from the BC Provincial Climate Data Set."""

    extensions = re.compile(r"^.*\.psql$", re.IGNORECASE)

    def __init__(self, dsn, sesh=None):
        self.dsn = dsn

        def session_scope_factory():
            return session_scope(dsn)

        self.session_scope_factory = session_scope_factory

        if sesh:
            # Stash a copy of our engine in pydap.handlers.sql so that it will use it for data queries
            Engines[self.dsn] = sesh.get_bind()

    def __call__(self, environ, start_response):
        """:param environ: WSGI environment such that PATH_INFO is set to something that matches the pattern /[network_name]/[native_id].sql.[response]
        :rtype: iterable WSGI response
        """
        filepath = environ.get("PATH_INFO")
        match = re.search(r"/([-a-zA-Z0-9_]+)/([a-zA-Z0-9_]+)\..sql", filepath)
        if not match:
            return HTTPNotFound(
                "Could not make sense of path {0}{1}".format(
                    environ.get("SCRIPT_NAME", ""), environ.get("PATH_INFO", "")
                )
            )(environ, start_response)

        net_name, native_id = match.groups()

        try:
            with self.session_scope_factory() as sesh:
                s = self.create_ini(sesh, net_name, native_id)
        except ValueError as e:
            return HTTPNotFound(e)(environ, start_response)  # 404
        f = NamedTemporaryFile("w", suffix=self.suffix, delete=False)

        eng_name = re.compile(r"(Engine\(([^\)]*)\))").findall(s)
        s = s.replace(eng_name[0][0], eng_name[0][1]) if eng_name else s

        f.write(s)
        f.close()

        app = SQLHandler(f.name)
        response = app(environ, start_response)
        os.remove(f.name)
        return response

    def create_ini(self, sesh, net_name, native_id):
        """Creates the actual text of a pydap SQL handler config file and returns it as a StringIO. `self.filepath` should be set before this is called. It will typically be something like ``.../[network_name]/[native_id].rsql``. The database station_id is looked up from that.

        :param environ: WSGI environment which *must* contain a dsn string under the key pydap.handlers.pcic.dsn
        :rtype: StringIO.StringIO
        """
        q = (
            sesh.query(Station.id)
            .join(Network)
            .filter(Station.native_id == native_id)
            .filter(Network.name == net_name)
        )
        if not q.first():
            raise ValueError(
                "No such station {net_name}/{native_id}".format(**locals())
            )
        (station_id,) = q.first()

        full_query = self.get_full_query(station_id, sesh)

        q = (
            sesh.query(
                Station.native_id,
                History.station_name,
                Network.name,
                History.the_geom,
                History.elevation,
            )
            .join(History)
            .join(Network)
            .filter(Station.id == station_id)
        )
        if q.count() < 1:
            native_id, station_name, network, lat, lon, elevation = (
                native_id,
                "",
                net_name,
                float("nan"),
                float("nan"),
                float("nan"),
            )
        else:
            if q.count() > 1:
                logger.warning(
                    "Multiple history entries (ids {}) were found for a single station_id, but we're reporting locations for the most recent".format(
                        []
                    )
                )
                (sdate,) = (
                    sesh.query(func.max(History.sdate))
                    .filter(History.station_id == station_id)
                    .first()
                )
                # This should never happen
                if not sdate:
                    raise ValueError(
                        "Found multiple history entries for station_id {}, but none have a valid record start date!".format(
                            station_id
                        )
                    )
                q = q.filter(History.sdate == sdate)

            _, station_name, network, geom, elevation = q.first()
            elevation = elevation if elevation else float("nan")
            lat, lon = (
                (sesh.scalar(ST_Y(geom)), sesh.scalar(ST_X(geom)))
                if geom is not None
                else (float("nan"), float("nan"))
            )

        dsn = self.dsn
        full_query = full_query.replace('"', '\\"')

        s = f"""database:
  dsn: "{dsn}"
  id: "obs_time"
  table: "({full_query}) as foo"
dataset:
  NC_GLOBAL:
    name: "CRMP/{network}"
    owner: "PCIC"
    contact: "Faron Anslow <fanslow@uvic.ca>"
    version: 0.2
    station_id: "{native_id}"
    station_name: "{station_name}"
    network: "{network}"
    latitude: {lat}
    longitude: {lon}
    elevation: {elevation}
    history: "Created dynamically by the Pydap SQL handler, the Pydap PCIC SQL handler, and the PCIC/CRMP database"
sequence:
  name: "station_observations"
time:
  name: "time"
  axis: "T"
  col: "obs_time"
  long_name: "observation time"
  units: "days since 1970-01-01"
  type: Float64
"""

        stn_vars = self.get_vars(station_id, sesh)

        for (
            var_name,
            unit,
            standard_name,
            cell_method,
            long_description,
            display_name,
        ) in stn_vars:
            s = (
                s
                + f"""{var_name}:
  name: "{var_name}"
  display_name: "{display_name}"
  long_name: "{long_description}"
  standard_name: "{standard_name}"
  units: "{unit}"
  cell_method: "{cell_method}"
  col: "{var_name}"
  axis: "Y"
  missing_value: -9999
  type: Float64
"""
            )

        return str(s)

    def get_full_query(self, stn_id, sesh):
        raise NotImplementedError

    def get_vars(self, stn_id, sesh):
        raise NotImplementedError


class RawPcicSqlHandler(PcicSqlHandler):
    """Subclass of PcicSqlHandler which handles the raw observations"""

    extensions = re.compile(r"^.*\.rsql$", re.IGNORECASE)
    suffix = ".rsql"
    virtual = True

    def get_full_query(self, stn_id, sesh):
        """Sends a special query to the database that actually retrieves generated SQL for constructing an observation table (time by variable) for a single station. The query needs to return at least one column (obs_time) with additional columns for each available variable, if any. Uses the ``query_one_station`` stored procedure.
        :param stn_id: the *database* station_id of the desired station
        :type stn_id: int or str
        :param sesh: an sqlalchemy session
        """

        if not self.get_vars(stn_id, sesh):
            return "SELECT obs_time FROM obs_raw WHERE NULL"

        query_string = "SELECT query_one_station(%s)" % stn_id
        return sesh.execute(query_string).fetchone()[0]

    def get_vars(self, stn_id, sesh):
        """Makes a database query to retrieve all of the raw variables for a particular station"""
        q = (
            sesh.query(Variable)
            .join(VarsPerHistory)
            .join(History)
            .join(Station)
            .join(Network)
            .filter(Station.id == stn_id)
            .filter(
                not_(
                    or_(
                        Variable.cell_method.like("%within%"),
                        Variable.cell_method.like("%over%"),
                    )
                )
            )
        )
        return [
            (
                x.name,
                x.unit,
                x.standard_name,
                x.cell_method,
                x.description,
                x.display_name,
            )
            for x in q.all()
        ]


class ClimoPcicSqlHandler(PcicSqlHandler):
    """Subclass of PcicSqlHandler which handles the climatological observations"""

    extensions = re.compile(r"^.*\.csql$", re.IGNORECASE)
    suffix = ".csql"
    virtual = True

    def get_full_query(self, stn_id, sesh):
        """Sends a special query to the database that actually retrieves generated SQL for constructing an observation table (time by variable) for a single station. Uses the ``query_one_station`` stored procedure.
        :param stn_id: the *database* station_id of the desired station
        :type stn_id: int or str
        :param sesh: sqlalchemy session
        """
        query_string = "SELECT query_one_station_climo(%s)" % stn_id
        return sesh.execute(query_string).first()[0]

    def get_vars(self, stn_id, sesh):
        """Makes a database query to retrieve all of the climatological variables for a particular station"""
        q = (
            sesh.query(Variable)
            .join(Network)
            .join(Station)
            .join(History)
            .join(VarsPerHistory)
            .filter(Station.id == stn_id)
            .filter(
                or_(
                    Variable.cell_method.like("%within%"),
                    Variable.cell_method.like("%over%"),
                )
            )
        )

        return [
            (
                x.name,
                x.unit,
                x.standard_name,
                x.cell_method,
                x.description,
                x.display_name,
            )
            for x in q.all()
        ]
