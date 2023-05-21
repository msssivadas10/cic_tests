#!/usr/bin/python3

import sys
import logging
from mpi4py import MPI

comm = MPI.COMM_WORLD

logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s %(process)d-[%(levelname)s] %(message)s",
                    handlers = [
                        logging.FileHandler(f'rank-{comm.rank}-output.log', mode = 'w'),
                        logging.StreamHandler()
                    ])


if comm.rank == 1:
    logging.error("aborting")
    # sys.exit(-1)
    comm.Abort(1)

logging.info(f"rank {comm.rank}")