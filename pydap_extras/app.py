import sys
from paste.httpserver import serve
from pydap.wsgi.ssf import ServerSideFunctions
from pydap_extras.handlers import csv

if __name__ == "__main__":
    csv._test()
    application = csv.CSVHandler(sys.argv[1])
    application = ServerSideFunctions(application)
    serve(application, port=8001)
