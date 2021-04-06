'''This module provides a Pydap responder which returns a Excel 2010 XLSX formatted file for a certain subset of DAP requests.
'''

from StringIO import StringIO

import numpy
from xlsxwriter import Workbook
import logging

from pydap.model import GridType, SequenceType
from pydap.lib import walk
from pydap.responses.lib import BaseResponse

logger = logging.getLogger('pydap.responses.xlsx')

FORMAT = {'font_color': 'white',
          'text_wrap': 'on',
          'align': 'center',
          'valign': 'vcenter',
          'pattern': 1,
          'fg_color': 'light blue'
         }

class XLSXResponse(BaseResponse):

    __description__ = "Excel 2010 spreadsheet"

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.extend([
                ('Content-description', 'dods_xlsx'),
                ('Content-type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                ])

    def __iter__(self):
        buf = dispatch(self.dataset)
        buf.seek(0)
        return iter(buf)


def dispatch(dataset):
    buf = StringIO()
    wb = Workbook(buf,
                  {'constant_memory': True,
                   'in_memory': True,
                   'default_date_format': 'yyyy/mm/dd hh:mm:ss'}
                 )
    format_ = wb.add_format(FORMAT)

    # dataset metadata
    ws = wb.add_worksheet('Global attributes')
    write_metadata(ws, dataset, 0, 0, format_)

    # 1D grids
    for grid in [g for g in walk(dataset, GridType) if len(g.shape) == 1]:
        logger.debug('Grid {}'.format(grid.name))
        ws = wb.add_worksheet(grid.name)

        # headers
        ws.write(0, 0, grid.dimensions[0], format_)
        ws.write(0, 1, grid.name, format_)

        # data
        for j, data in enumerate(grid.data):
            for i, value in enumerate(numpy.asarray(data)):
                ws.write(i+1, 1-j, value)

        # add var metadata
        write_metadata(ws, grid, 0, 2, format_)

    # sequences
    for seq in walk(dataset, SequenceType):
        logger.debug('Sequence {}'.format(seq.name))

        ws = wb.add_worksheet(seq.name)

        # add header across the first row
        for j, var_ in enumerate(seq.keys()):
            ws.write(0, j, var_, format_)

        # add data in the subsequent rows
        for i, row in enumerate(seq.data):
            for j, value in enumerate(row):
                ws.write(i+1, j, value)

        # add var metadata in columns to the right of the data
        n = 0
        j = len(seq.keys())+1
        for child in seq.children():
            logger.debug("Child {}".format(child.name))
            ws.merge_range(n, j, n, j+1, child.name, format_)
            n = write_metadata(ws, child, n+1, j, format_)+1

    wb.close()
    return buf


def write_metadata(ws, var, i, j, format_):
    logger.debug('Metadata')
    for k, v in var.attributes.items():
        n = height(v)
        write_attr(ws, k, v, i, j, format_)
        i += n
    return i


def write_attr(ws, k, v, i, j, format_):
    col_width = ws.col_sizes[j] if j in ws.col_sizes else 0
    col_width = max(col_width, (len(k) + 1))

    ws.set_column(j, j, col_width)

    if isinstance(v, dict):
        n = height(v)
        ws.merge_range(i, j, i+n-1, j, '{}'.format(k), format_)

        for kk, vv in v.items():
            n = height(vv)
            write_attr(ws, kk, vv, i, j+1, format_)
            i += n
    else:
        ws.write(i, j, '{}'.format(k), format_)

        try:
            if len(v) == 1: v = v[0]
        except:
            pass
        ws.write(i, j+1, '{}'.format(v))

def height(v):
    """Return the number of elements in an attribute."""
    if isinstance(v, dict):
        return sum( height(o) for o in v.items() )
    else:
        return 1