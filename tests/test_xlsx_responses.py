import importlib.metadata
from pydap_extras.responses.xlsx import XLSXResponse
from pydap_extras.handlers.csv import CSVHandler
from webob import Request


def test_XLSXResponse(simple_data_file):
    dataset = CSVHandler(simple_data_file).dataset
    response = XLSXResponse(dataset)

    req = Request.blank("/")
    res = req.get_response(response)
    assert res.status == "200 OK"
    assert res.app_iter == response
    assert (
        res.content_type
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert response.headers == [
        ("XDODS-Server", f"pydap/{importlib.metadata.version('pydap')}"),
        ("Content-description", "dods_xlsx"),
        (
            "Content-type",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    ]

    assert type(response.__iter__().read()) == bytes
