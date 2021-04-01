import sys
from paste.httpserver import serve
from pydap.wsgi.ssf import ServerSideFunctions
from pydap_extras.handlers import csv, sql

if __name__ == "__main__":
    handler_type = sys.argv[1]

    if handler_type == "csv":
        csv._test()
        application = csv.CSVHandler(sys.argv[2])
    elif handler_type == "sql":
        sql._test()
        application = sql.CSVHandler(sys.argv[2])
    else:
        raise KeyError(f"No handler called {handler_type}")

    application = ServerSideFunctions(application)
    serve(application, port=8001)
