import pytest
import csv

@pytest.fixture(scope='session')
def simple_data():
    data = [(10, 15.2, 'Diamond_St'),
            (11, 13.1, 'Blacktail_Loop'),
            (12, 13.3, 'Platinum_St'),
            (13, 12.1, 'Kodiak_Trail')]
    return data


@pytest.fixture(scope='session')
def simple_data_file(tmpdir_factory, simple_data):
    temp_file = str(tmpdir_factory.mktemp('data').join('simple_data.csv'))
    with open(temp_file, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['index', 'temperature', 'site'])
        for row in simple_data:
            writer.writerow(row)
    return temp_file