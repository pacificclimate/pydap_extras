import pytest
import csv
import os
from tempfile import NamedTemporaryFile
from sqlalchemy import create_engine


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
