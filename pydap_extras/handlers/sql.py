"""
Pydap SQL handler.
This handler allows Pydap to server data from any relational database supported
by SQLAlchemy. Each dataset is represented by a YAML file that defines the 
database connection, variables and other associated metadata. Here's a simple
example:
    database:
        dsn: 'sqlite:///simple.db'
        table: test
    dataset:
        NC_GLOBAL:
            history: Created by the Pydap SQL handler
        contact: roberto@dealmeida.net
        name: test_dataset
        owner: Roberto De Almeida
        version: 1.0
        last_modified: !Query 'SELECT time FROM test ORDER BY time DESC LIMIT 1;'
    sequence:
        name: simple
        items: !Query 'SELECT COUNT(id) FROM test'
    _id:
        col: id
        long_name: sequence id
        missing_value: -9999
    lon:
        col: lon
        axis: X
        grads_dim: x
        long_name: longitude
        units: degrees_east
        missing_value: -9999
        global_range: [-180, 180]
        valid_range: !Query 'SELECT min(lon), max(lon) FROM test'
"""
import sys
import os
import itertools
import re
import ast
import copy
from datetime import datetime
import time
from email.utils import formatdate
import operator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml
import numpy as np

from pydap.model import *
from pydap.lib import quote
from pydap.handlers.lib import BaseHandler
from pydap.exceptions import OpenFileError, ConstraintExpressionError
from pydap_extras.handlers.csv import CSVData


# module level engines, using connection pool
class EngineCreator(dict):
    def __missing__(self, key):
        self[key] = create_engine(key)
        return self[key]


Engines = EngineCreator()

# From http://docs.sqlalchemy.org/en/rel_0_9/orm/session.html#session-faq-whentocreate
@contextmanager
def session_scope(dsn):
    """Provide a transactional scope around a series of operations. Cleans up the connection even in the case of failure"""
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


class SQLHandler(BaseHandler):

    extensions = re.compile(r"^.*\.sql$", re.IGNORECASE)

    def __init__(self, filepath):
        """
        Prepare dataset.
        The `__init__` method of handlers is responsible for preparing the dataset
        for incoming requests.
        """
        BaseHandler.__init__(self)

        # open the YAML file and parse configuration
        try:
            with open(filepath, "r") as fp:
                fp = open(filepath, "r")
                config = yaml.safe_load(fp)
        except Exception as exc:
            raise OpenFileError(f"Unable to open file {filepath}: {exc}")

        # add last-modified header from config, if available
        try:
            last_modified = config["dataset"]["last_modified"]
            if isinstance(last_modified, tuple):
                last_modified = last_modified[0]
            if isinstance(last_modified, basestring):
                last_modified = datetime.strptime(last_modified, "%Y-%m-%d %H:%M:%S")
            self.additional_headers.append(
                ("Last-modified", formatdate(time.mktime(last_modified.timetuple())))
            )
        except KeyError:
            pass

        # Peek types. I'm trying to avoid having the user know about Opendap types,
        # so they are not specified in the config file. Instead, we request a single
        # row of data to inspect the data types.
        # FIXME: type peeking does not work if there are NA values in the sequence!!!
        cols = tuple(key for key in config if "col" in config[key])
        with session_scope(config["database"]["dsn"]) as conn:
            query = "SELECT {cols} FROM {table} LIMIT 1".format(
                cols=", ".join(config[key]["col"] for key in cols),
                table=config["database"]["table"],
            )
            results = conn.execute(query)
            first_row = results.fetchone()

            dtypes = {}
            for col, value, description in zip(
                cols, first_row, results.cursor.description
            ):
                # FIXME: This is fraaaagile, and depends on internal, undocumented behaviour from SQLAlchemy
                if value is None and description[1]:
                    # the value is NULL... try to use the typecode
                    dtypes[col] = {
                        700: np.dtype("float64"),
                        701: np.dtype("float64"),
                        1114: np.dtype("datetime64"),
                    }[description[1]]
                elif type(value) == datetime:
                    dtypes[col] = np.dtype("datetime64")
                else:
                    dtypes[col] = np.array(value).dtype

        # create the dataset, adding attributes from the config file
        attrs = config.get("dataset", {}).copy()
        name = attrs.pop("name", os.path.split(filepath)[1])
        self.dataset = DatasetType(name, attrs)

        # and now create the sequence
        attrs = config.get("sequence", {}).copy()
        name = attrs.pop("name", "sequence")
        seq = self.dataset[quote(name)] = SequenceType(name, config, attrs)
        for var in cols:
            attrs = {k: v for k, v in config[var].items() if k != "col"}
            seq[var] = BaseType(var, attributes=attrs)

        # set the data
        seq.data = SQLData(config, seq.id, tuple(cols), dtypes, copy.copy(seq))


class SQLData(CSVData):
    """
    Emulate a Numpy structured array using an SQL database.
    Here's a standard dataset for testing sequential data:
        >>> data = [
        ... (10, 15.2, 'Diamond_St'),
        ... (11, 13.1, 'Blacktail_Loop'),
        ... (12, 13.3, 'Platinum_St'),
        ... (13, 12.1, 'Kodiak_Trail')]
        >>> import os
        >>> if os.path.exists('test.db'):
        ...     os.unlink('test.db')
        >>> import sqlite3
        >>> conn = sqlite3.connect('test.db')
        >>> c = conn.cursor()
        >>> out = c.execute("CREATE TABLE test (idx real, temperature real, site text)")
        >>> out = c.executemany("INSERT INTO test VALUES (?, ?, ?)", data)
        >>> conn.commit()
        >>> c.close()
    Iteraring over the sequence returns data:
        >>> config = {
        ...     'database': { 'dsn': 'sqlite:///test.db', 'table': 'test', 'order': 'idx' },
        ...     'index': { 'col': 'idx' },
        ...     'temperature': { 'col': 'temperature' },
        ...     'site': { 'col': 'site' }}
        >>> seq = SequenceType('example')
        >>> seq['index'] = BaseType('index')
        >>> seq['temperature'] = BaseType('temperature')
        >>> seq['site'] = BaseType('site')
        >>> seq.data = SQLData(config, seq.id, ('index', 'temperature', 'site'),
        ...     {'index': np.int32, 'temperature': np.float32, 'site': np.dtype('|S14')})
        >>> for line in seq:
        ...     print line
        (10.0, 15.2, u'Diamond_St')
        (11.0, 13.1, u'Blacktail_Loop')
        (12.0, 13.3, u'Platinum_St')
        (13.0, 12.1, u'Kodiak_Trail')
        >>> for line in seq['temperature', 'site', 'index']:
        ...     print line
        (15.2, u'Diamond_St', 10.0)
        (13.1, u'Blacktail_Loop', 11.0)
        (13.3, u'Platinum_St', 12.0)
        (12.1, u'Kodiak_Trail', 13.0)
    We can iterate over children:
        >>> for line in seq['temperature']:
        ...     print line
        15.2
        13.1
        13.3
        12.1
    We can filter the data:
        >>> for line in seq[ seq.index > 10 ]:
        ...     print line
        (11.0, 13.1, u'Blacktail_Loop')
        (12.0, 13.3, u'Platinum_St')
        (13.0, 12.1, u'Kodiak_Trail')
        >>> for line in seq[ seq.index > 10 ]['site']:
        ...     print line
        Blacktail_Loop
        Platinum_St
        Kodiak_Trail
        >>> for line in seq['site', 'temperature'][ seq.index > 10 ]:
        ...     print line
        (u'Blacktail_Loop', 13.1)
        (u'Platinum_St', 13.3)
        (u'Kodiak_Trail', 12.1)
    Or slice it:
        >>> for line in seq[::2]:
        ...     print line
        (10.0, 15.2, u'Diamond_St')
        (12.0, 13.3, u'Platinum_St')
        >>> for line in seq[ seq.index > 10 ][::2]['site']:
        ...     print line
        Blacktail_Loop
        Kodiak_Trail
        >>> for line in seq[ seq.index > 10 ]['site'][::2]:
        ...     print line
        Blacktail_Loop
        Kodiak_Trail
    """

    def __init__(
        self,
        config,
        id,
        cols,
        dtypes,
        template,
        imap=None,
        selection=None,
        slice_=None,
        level=0,
    ):
        self.template = template
        self.config = config
        self.id = id
        self.cols = cols
        self.dtypes = dtypes
        self.selection = selection or []
        self.slice = slice_ or (slice(None),)
        self.level = level
        self.imap = imap or []

        # mapping between variable names and their columns
        self.mapping = {
            key: config[key]["col"] for key in config if "col" in config[key]
        }

    @property
    def dtype(self):
        return np.dtype(
            {"names": list(self.dtypes.keys()), "formats": list(self.dtypes.values())}
        )

    @property
    def query(self):
        if "order" in self.config["database"]:
            order = "ORDER BY {order}".format(**self.config["database"])
        else:
            order = ""

        if isinstance(self.cols, tuple):
            cols = self.cols
        else:
            cols = (self.cols,)

        where, params = parse_queries(self.selection, self.mapping)
        if where:
            where = "WHERE {conditions}".format(conditions=" AND ".join(where))
        else:
            where = ""

        sql = "SELECT {cols} FROM {table} {where} {order} LIMIT {limit} OFFSET {offset}".format(
            cols=", ".join(self.config[key]["col"] for key in cols),
            table=self.config["database"]["table"],
            where=where,
            order=order,
            limit=(self.slice[0].stop or sys.maxsize) - (self.slice[0].start or 0),
            offset=self.slice[0].start or 0,
        )

        if params:
            return [sql, params]
        else:
            return [sql]

    def __len__(self):
        with session_scope(self.config["database"]["dsn"]) as conn:
            data = conn.execute(*self.query)
            rv = data.rowcount
        return rv

    def __iter__(self):
        with session_scope(self.config["database"]["dsn"]) as conn:
            data = conn.execute(*self.query)

            # there's no standard way of choosing every n result from a query using
            # SQL, so we need to filter it on Python side
            data = itertools.islice(data, 0, None, self.slice[0].step)

            # return data from a children BaseType, not a Sequence
            if not isinstance(self.cols, tuple):
                data = itertools.imap(operator.itemgetter(0), data)

            for row in data:
                yield row

    def __copy__(self):
        return self.__class__(
            self.config,
            self.id,
            self.cols[:],
            self.dtypes,
            self.template,
            self.imap,
            self.selection[:],
            self.slice[:],
            self.level,
        )


def parse_queries(selection, mapping):
    """
    Convert an Opendap selection to an SQL query.
    """
    out = []
    params = {}
    for i, expression in enumerate(selection):
        if isinstance(expression, str):
            id1, op, id2 = re.split("(<=|>=|!=|=~|>|<|=)", expression, maxsplit=1)

            # a should be a variable in the children
            name1 = id1.split(".")[-1]
            if name1 in mapping:
                a = mapping[name1]
            else:
                raise ConstraintExpressionError(
                    f'Invalid constraint expression: "{expression}" ("{id1}" is not a valid variable)'
                )

            # b could be a variable or constant
            name2 = id2.split(".")[-1]
            if name2 in mapping:
                b = mapping[name2]
            else:
                b = ast.literal_eval(id2)

            out.append(
                f"({a} {op} :{i})"
            )  # bad hack for positional args since Session.execute doesn't support them
            params[str(i)] = b

    return out, params


def yaml_query(loader, node):
    """
    Special loader for database queries.
    This is a special loader for parsing the YAML config file. The configuration
    allows queries to be embedded in the file using the `!Query` identifier.
    """
    # read DSN
    for obj in [
        obj for obj in loader.constructed_objects if isinstance(obj, yaml.MappingNode)
    ]:
        try:
            mapping = loader.construct_mapping(obj)
            dsn = mapping["dsn"]
            break
        except:
            pass

    # get/set connection
    with session_scope(dsn) as conn:
        query = loader.construct_scalar(node)
        results = conn.execute(query).fetchone()

    return tuple(results)


yaml.add_constructor("!Query", yaml_query)


def _test():
    import doctest

    doctest.testmod()
