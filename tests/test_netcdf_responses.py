import pytest
from pydap_extras.responses.netcdf import NCResponse
from webob import Request
import os
import tempfile

import pytest
import netCDF4


def test_NCResponse_simple(simple_dataset):
    response = NCResponse(simple_dataset)

    req = Request.blank("/")
    res = req.get_response(response)
    assert res.status == "200 OK"
    assert res.app_iter == response


@pytest.mark.timeout(2, method="signal")
def test_no_recvars(netcdf_handler):
    env = {
        "REQUEST_METHOD": "GET",
        "SCRIPT_NAME": "",
        "PATH_INFO": "tiny_bccaq2_wo_recvars.nc.nc",
    }
    resp = netcdf_handler(environ=env, start_response=lambda x, y: x)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        for block in iter(resp):
            tmp.write(block)

        with netCDF4.Dataset(tmp.name) as dst:
            assert "pr" in dst.variables


def test_time_val_out_of_bounds(netcdf_handler):
    env = {
        "REQUEST_METHOD": "GET",
        "SCRIPT_NAME": "",
        "PATH_INFO": "tiny_bccaq2_wo_recvars.nc.nc",
        "QUERY_STRING": "pr[0:11][][]",  # Note that this is *12* timesteps
    }
    resp = netcdf_handler(environ=env, start_response=lambda x, y: True)

    # We have to dig deep into PyDAP interals here, but long story
    # short we *should* get an array that's only 11 timesteps long
    # even though we requested one with 12 steps

    pr = [x for x in resp.dataset.children()][0]
    assert pr.array.shape[0] == 11
