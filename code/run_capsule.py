# stdlib imports --------------------------------------------------- #
import os

os.environ["RUST_BACKTRACE"] = "1"
os.environ['POLARS_MAX_THREADS'] = '1'
os.environ["TOKIO_WORKER_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

import json
import logging
import pathlib
import time

import encoding_utils
import matplotlib

# 3rd-party imports necessary for processing ----------------------- #
import pandas as pd
import upath
import utils

# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem
    if __name__.endswith("_main__")
    else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams["pdf.fonttype"] = 42
logging.getLogger("matplotlib.font_manager").setLevel(
    logging.ERROR
)  # suppress matplotlib font warnings on linux


def main():
    t0 = time.time()

    utils.setup_logging()
    params = encoding_utils.Params()  # reads from CLI args
    logger.setLevel(params.logging_level)

    if params.override_params_json:
        logger.info(f"Overriding parameters with {params.override_params_json}")
        params = encoding_utils.Params(**json.loads(params.override_params_json))

    if params.test:
        params = encoding_utils.Params(
            result_prefix=f"test/{params.result_prefix}",
            # TODO make a proper test set of params
        )
        logger.info("Test mode: using modified set of parameters")

    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:
    session_table = pd.read_parquet(utils.get_datacube_dir() / "session_table.parquet")
    session_table["issues"] = session_table["issues"].astype(str)
    session_ids: list[str] = session_table.query(params.session_table_query)[
        "session_id"
    ].values.tolist()
    logger.debug(
        f"Found {len(session_ids)} session_ids available for use after filtering"
    )
    
    # filter session_ids based on those requested via command line arguments:
    if params.single_session_id_to_use and params.unit_ids_to_use:
        raise ValueError(
            "Cannot use both single_session_id_to_use and unit_ids_to_use at the same time"
        )
    if params.unit_ids_to_use:
        requested_session_ids = set(unit_id.rsplit("_")[0] for unit_id in params.unit_ids_to_use)
        logger.info(
            f"Using unit_ids_to_use {params.unit_ids_to_use} to filter session_ids"
        )
        if requested_session_ids - set(session_ids):
            logger.warning(
                f"Some requested unit_ids_to_use do not correspond to known sessions: {requested_session_ids - set(session_ids)}"
            )
        session_ids = set(session_ids) & requested_session_ids
    elif params.single_session_id_to_use is not None:
        if params.single_session_id_to_use not in session_ids:
            logger.warning(
                f"{params.single_session_id_to_use!r} not in filtered session_ids: exiting"
            )
            exit()
        logger.info(
            f"Using single session_id {params.single_session_id_to_use} provided via command line argument"
        )
        session_ids = [params.single_session_id_to_use]
    else:
        session_ids = []
    
    # filter requested sessions based on NWBs available:
    nwb_session_ids = set(p.stem for p in utils.get_nwb_paths())
    if not nwb_session_ids:
        logger.warning("No NWBs found in datacube: exiting")
        exit()
    if set(session_ids) - nwb_session_ids:
        logger.warning(
            f"Some requested session_ids are not available as NWBs: {set(session_ids) - nwb_session_ids}"
        )
    session_ids = set(session_ids) & nwb_session_ids

    logger.info(f"Using list of {len(session_ids)} session_ids after filtering")

    upath.UPath("/results/params.json").write_text(params.model_dump_json(indent=4))
    if params.json_path.exists():
        existing_params = json.loads(params.json_path.read_text())
        if existing_params != params.model_dump():
            raise ValueError(
                f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}"
            )
    else:
        logger.info(f"Writing params file: {params.json_path}")
        params.json_path.write_text(params.model_dump_json(indent=4))

    logger.info(f"starting encoding with {params!r}")
    # encoding_utils.get_fullmodel_data(session_id=session_ids[0], params=encoding_utils.Params())
    encoding_utils.run_encoding(session_ids=[session_ids[0]], params=encoding_utils.Params())

    utils.ensure_nonempty_results_dirs()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()


## TO DO - app panel
# 1. Add features to drop
# 2. Add skip_existing
# 3. Add run_id
# 4. Add file path
# 4. Add file path
