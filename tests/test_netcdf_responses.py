import pytest
from pydap_extras.responses.netcdf import NCResponse
from pydap_extras.handlers.csv import CSVHandler
from webob import Request


def test_NCResponse(simple_data_file):
    dataset = CSVHandler(simple_data_file).dataset
    response = NCResponse(dataset)

    req = Request.blank("/")
    res = req.get_response(response)
    assert res.status == "200 OK"
    assert res.app_iter == response