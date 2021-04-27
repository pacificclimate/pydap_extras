import time

from itertools import chain
from collections import Iterator
import logging
from datetime import datetime
from functools import reduce

import numpy as np
from numpy.compat import asbytes
from pydap.model import *
from pydap.lib import walk, get_var
from pydap.responses.lib import BaseResponse
from pupynere import netcdf_file, nc_generator

logger = logging.getLogger(__name__)


class NCResponse(BaseResponse):
    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)

        self.nc = netcdf_file(None, version=2)
        if "NC_GLOBAL" in self.dataset.attributes:
            self.nc._attributes.update(self.dataset.attributes["NC_GLOBAL"])

        dimensions = [
            var.dimensions for var in walk(self.dataset) if isinstance(var, BaseType)
        ]
        dimensions = set(reduce(lambda x, y: x + y, dimensions))
        try:
            unlim_dim = self.dataset.attributes["DODS_EXTRA"]["Unlimited_Dimension"]
        except:
            unlim_dim = None

        # GridType
        for grid in walk(dataset, GridType):

            # add dimensions
            for dim, map_ in grid.maps.items():
                if dim in self.nc.dimensions:
                    continue

                n = None if dim == unlim_dim else grid[dim].data.shape[0]
                self.nc.createDimension(dim, n)
                if not n:
                    self.nc.set_numrecs(grid[dim].data.shape[0])
                var = grid[dim]

                # and add dimension variable
                self.nc.createVariable(
                    dim, var.dtype.char, (dim,), attributes=var.attributes
                )

            # finally add the grid variable itself
            base_var = grid[grid.name]
            var = self.nc.createVariable(
                base_var.name,
                base_var.dtype.char,
                base_var.dimensions,
                attributes=base_var.attributes,
            )

        # Sequence types!
        for seq in walk(dataset, SequenceType):

            self.nc.createDimension(seq.name, None)
            try:
                n = len(seq)
            except TypeError:
                # FIXME: materializing and iterating through a sequence to find the length
                # could have performance problems and could potentially consume the iterable
                # Do lots of testing here and determine the result of not calling set_numrecs()
                n = len([x for x in seq[next(seq.keys())]])
            self.nc.set_numrecs(n)

            dim = (seq.name,)

            for child in seq.children():
                dtype = child.dtype
                # netcdf does not have a date type, so remap to float
                if dtype == np.dtype("datetime64"):
                    dtype = np.dtype("float32")
                elif dtype == np.dtype("object"):
                    raise TypeError(
                        f"Don't know how to handle numpy type {dtype}"
                    )

                var = self.nc.createVariable(
                    child.name, dtype.char, dim, attributes=child.attributes
                )

        self.headers.extend([("Content-type", "application/x-netcdf")])
        # Optionally set the filesize header if possible
        try:
            self.headers.extend([("Filesize", str(self.nc.filesize))])
        except ValueError:
            pass

    def __iter__(self):
        nc = self.nc

        # Hack to find the variables if they're nested in the tree
        var2id = {}
        for recvar in nc.variables.keys():
            for dstvar in walk(self.dataset, BaseType):
                if recvar == dstvar.name:
                    var2id[recvar] = dstvar.id
                    continue

        def type_generator(input):
            epoch = datetime(1970, 1, 1)
            # is this a "scalar" (i.e. a standard python object)
            # if so, it needs to be a numpy array, or at least have 'dtype' and 'byteswap' attributes
            for value in input:
                if isinstance(value, (type(None), str, int, float, bool, datetime)):
                    # special case datetimes, since dates aren't supported by NetCDF3
                    if type(value) == datetime:
                        since_epoch = (value - epoch).total_seconds()
                        yield np.array(
                            since_epoch / 3600.0 / 24.0, dtype="Float32"
                        )  # days since epoch
                    else:
                        yield np.array(value)
                else:
                    yield value

        def nonrecord_input():
            for varname in nc.non_recvars.keys():
                logger.debug(f"Iterator for {varname}")
                dst_var = get_var(self.dataset, var2id[varname]).data
                # skip 0-d variables
                if not dst_var.shape:
                    continue

                # Make sure that all elements of the list are iterators
                for x in dst_var:
                    logger.debug(f"nonrecord_input yielding {x} from var {varname}")
                    yield x
            logger.debug("Done with nonrecord input")

        # Create a generator for the record variables
        recvars = nc.recvars.keys()

        def record_generator(nc, dst, table):
            logger.debug(f"record_generator() for dataset {dst}")
            if not nc.recvars:
                logger.debug("file has no record variables")
                return
            vars = [iter(get_var(dst, table[varname])) for varname in nc.recvars.keys()]
            while True:
                for var in vars:
                    try:
                        yield next(var)
                    except StopIteration:
                        return

        more_input = type_generator(record_generator(nc, self.dataset, var2id))

        # Create a single pipeline which includes the non-record and record variables
        pipeline = nc_generator(
            nc, chain(type_generator(nonrecord_input()), more_input)
        )

        # Generate the netcdf stream
        for block in pipeline:
            yield block
