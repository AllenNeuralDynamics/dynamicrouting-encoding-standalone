# stdlib imports --------------------------------------------------- #
import argparse
import dataclasses
import json
import functools
import logging
import pathlib
import time
import types
import typing
import uuid
from typing import Any, Literal
import copy 

# 3rd-party imports necessary for processing ----------------------- #
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pynwb
import upath
import zarr
from dynamic_routing_analysis import io_utils
import utils

# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem if __name__.endswith("_main__") else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


# utility functions ------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--session_id', type=str, default=None)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--update_packages_from_source', type=int, default=1)
    parser.add_argument('--session_table_query', type=str, default="is_ephys & is_task & is_annotated & is_production & project == 'DynamicRouting' & issues=='[]'")
    parser.add_argument('--override_params_json', type=str, default="{}")
    for field in dataclasses.fields(Params):
        if field.name in [getattr(action, 'dest') for action in parser._actions]:
            # already added field above
            continue
        logger.debug(f"adding argparse argument {field}")
        kwargs = {}
        if isinstance(field.type, str):
            kwargs = {'type': eval(field.type)}
        else:
            kwargs = {'type': field.type}
        if kwargs['type'] in (list, tuple):
            logger.debug(f"Cannot correctly parse list-type arguments from App Builder: skipping {field.name}")
        if isinstance(field.type, str) and field.type.startswith('Literal'):
            kwargs['type'] = str
        if isinstance(kwargs['type'], (types.UnionType, typing._UnionGenericAlias)):
            kwargs['type'] = typing.get_args(kwargs['type'])[0]
            logger.debug(f"setting argparse type for union type {field.name!r} ({field.type}) as first component {kwargs['type']!r}")
        parser.add_argument(f'--{field.name}', **kwargs)
    args = parser.parse_args()
    list_args = [k for k,v in vars(args).items() if type(v) in (list, tuple)]
    if list_args:
        raise NotImplementedError(f"Cannot correctly parse list-type arguments from App Builder: remove {list_args} parameter and provide values via `override_params_json` instead")
    logger.info(f"{args=}")
    return args

# processing function ---------------------------------------------- #
# modify the body of this function, but keep the same signature
def process_session(session_id: str, params: "Params", test: int = 0) -> None:
    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    # Get nwb file
    # Currently this can fail for two reasons: 
    # - the file is missing from the datacube, or we have the path to the datacube wrong (raises a FileNotFoundError)
    # - the file is corrupted due to a bad write (raises a RecursionError)
    # Choose how to handle these as appropriate for your capsule
    try:
        session = utils.get_nwb(session_id, raise_on_missing=True, raise_on_bad_file=True) 
    except (FileNotFoundError, RecursionError) as exc:
        logger.info(f"Skipping {session_id}: {exc!r}")
        return
    
    # Process data here, with test mode implemented to break out of the loop early or use reduced param set:
    
    if test:
        logger.info("TEST | Using reduced params set")
        unit_counts_per_areas = session.units[:]['structure'].value_counts()
        filtered_structures = unit_counts_per_areas[(unit_counts_per_areas >= 50) & (~unit_counts_per_areas.index.str.islower())]
        params.areas_to_include = [filtered_structures.index[0]] if not filtered_structures.empty else None
        params.time_of_interest = 'trial'
        params.spike_bin_width = 0.5
        params.run_on_qc_units = True

    logger.info(f"Processing {session_id} with {params.to_json()}")

    # Save data to files in /results
    # If the same name is used across parallel runs of this capsule in a pipeline, a name clash will
    # occur and the pipeline will fail, so use session_id as filename prefix:
    #   /results/<sessionId>.suffix

    # Make output directories
    output_dirs = {'full': pathlib.Path(f'/results/full'), 'reduced': pathlib.Path(f'/results/reduced')}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)


    # fullmodel 
    logger.info(f'Building fullmodel')
    subfolder = 'full'
    io_params = io_utils.RunParams(session_id=session_id)
    io_params.update_multiple_metrics(dataclasses.asdict(params))   
    io_params.update_metric("model_label", "fullmodel")
    io_params.validate_params()
    run_params = io_params.get_params()
    units_table, behavior_info = io_utils.get_session_data(session)
    fit = io_utils.extract_unit_data(run_params, units_table, behavior_info)
    design = io_utils.DesignMatrix(fit)
    design, fit = io_utils.add_kernels(design, run_params, session, fit, behavior_info)
    design_matrix = design.get_X()
    fit['is_good_behavior'] = behavior_info['is_good_behavior']
    fit['dprime'] = behavior_info['dprime']

    # Save results
    model_name = run_params["model_label"]
    output_path = output_dirs[subfolder] / f'{session_id}_{model_name}_inputs.npz'
    logger.info(f"Writing {output_path}")
    np.savez(
        file=output_path,
        design_matrix= {'data': design_matrix.data,
                        'weight_labels': design_matrix.weights.values,
                        'timestamps': design_matrix.timestamps.values}, 
        fit=fit,  # Ensure dict can be saved properly
        run_params=run_params,  # Ensure dict can be saved properly
    )
    
    # dropout models
    features_to_drop = params.features_to_drop or (
        list(run_params['input_variables']) +  
        [run_params['kernels'][key]['function_call'] for key in run_params['input_variables']]
    )
    features_to_drop = list(set(features_to_drop))
    for feature in features_to_drop:
    
        # pipeline will execute different behavior for files in different subfolders:
        
        # Create deep copies of run_params, fit, and design_matrix
        run_params_reduced = copy.deepcopy(run_params)
        fit_reduced = copy.deepcopy(fit)
        design_matrix_reduced = design_matrix.copy()

        logger.info(f'Building reduced model for {feature}')
        subfolder = 'reduced' 
        if feature not in fit['failed_kernels']:
            # Modify the copies
            run_params_reduced["drop_variables"] = [feature]
            run_params_reduced["model_label"] = f'drop_{feature}'
            run_params_reduced["input_variables"].remove(feature)
            run_params_reduced["kernels"].pop(feature)

            # Filter design matrix 
            filtered_weights = [weight for weight in design_matrix_reduced.weights.values if feature not in weight]
            design_matrix_reduced = design_matrix_reduced.sel(weights=filtered_weights)
        else:
            logger.warning(f"Failed kernel {feature}, skipping dropout analyses.")
            continue 


        # Save results
        model_name = run_params_reduced["model_label"]
        output_path = output_dirs[subfolder] / f'{session_id}_{model_name}_inputs.npz'
        logger.info(f"Writing {output_path}")
        np.savez(
            file=output_path,
            design_matrix = {'data': design_mat_reduced.data,
                            'weight_labels': design_mat_reduced.weights,
                            'timestamps': design_mat_reduced.timestamps}, 
            fit=fit_fit_reduced,  # Ensure dict can be saved properly
            run_params=run_params_reduced,  # Ensure dict can be saved properly
        )

# define run params here ------------------------------------------- #

# The `Params` class is used to store parameters for the run, for passing to the processing function.
# @property fields (like `bins` below) are computed from other parameters on-demand as required:
# this way, we can separate the parameters dumped to json from larger arrays etc. required for
# processing.

# - if needed, we can get parameters from the command line (like `nUnitSamples` below) and pass them
#   to the dataclass (see `main()` below)

# this is an example from Sam's processing code, replace with your own parameters as needed:
@dataclasses.dataclass
class Params:
    session_id: str 
    time_of_interest: str = 'quiescent'
    spontaneous_duration: float = 2 * 60 # in seconds
    features_to_drop: list | None = None
    input_variables: list | None = None
    input_offsets: bool = True
    input_window_lengths: dict | None = None
    drop_variables: list = None
    unit_inclusion_criteria: dict[str, float] = dataclasses.field(default_factory=lambda: {'isi_violations': 0.1, 
                                                                                            'presence_ratio': 0.99, 
                                                                                            'amplitude_cutoff': 0.1, 
                                                                                            'firing_rate': 1})
 
    run_on_qc_units: bool = False
    spike_bin_width: float = 0.025
    areas_to_include: list = None
    areas_to_exclude: list = None
    orthogonalize_against_context: list = dataclasses.field(default_factory = lambda:['facial_features'])
    quiescent_start_time: float = -1.5
    quiescent_stop_time: float = 0
    trial_start_time: float = -2
    trial_stop_time: float = 3
    intercept: bool = True

    def to_json(self, **dumps_kwargs) -> str:
        """json string of field name: value pairs, excluding values from property getters (which may be large)"""
        return json.dumps(dataclasses.asdict(self), **dumps_kwargs)

    def write_json(self, path: str | upath.UPath = '/results/params.json') -> None:
        path = upath.UPath(path)
        logger.info(f"Writing params to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))

# ------------------------------------------------------------------ #


def main():
    t0 = time.time()
    
    utils.setup_logging()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:

    params = {}
    for field in dataclasses.fields(Params):
        if (val := getattr(args, field.name, None)) is not None:
            params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            params[k] = v
            
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:    
    session_table = pd.read_parquet(utils.get_datacube_dir() / 'session_table.parquet')
    session_table['issues']=session_table['issues'].astype(str)
    session_ids: list[str] = session_table.query(args.session_table_query)['session_id'].values.tolist()
    logger.debug(f"Found {len(session_ids)} session_ids available for use after filtering")
    
    if args.session_id is not None:
        if args.session_id not in session_ids:
            logger.warning(f"{args.session_id!r} not in filtered session_ids: exiting")
            exit()
        logger.info(f"Using single session_id {args.session_id} provided via command line argument")
        session_ids = [args.session_id]
    elif utils.is_pipeline(): 
        # only one nwb will be available 
        session_ids = set(session_ids) & set(p.stem for p in utils.get_nwb_paths())
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids after filtering")
    
    # run processing function for each session, with test mode implemented:
    for session_id in session_ids:
        try:
            process_session(session_id, params=Params(session_id=session_id, **params), test=args.test)
        except Exception as e:
            logger.exception(f'{session_id} | Failed:')
        else:
            logger.info(f'{session_id} | Completed')

        if args.test:
            logger.info("Test mode: exiting after first session")
            break
    utils.ensure_nonempty_results_dirs(('/results/full', '/results/reduced')) # required for pipeline to work in case this session has no outputs
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")
    
if __name__ == "__main__":
    main()
