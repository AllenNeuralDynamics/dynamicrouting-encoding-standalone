from __future__ import annotations

import datetime
import os
import pathlib
import pickle

import dynamic_routing_analysis
import dynamic_routing_analysis.datacube_utils
import pydantic
import pydantic_settings

os.environ["RUST_BACKTRACE"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["TOKIO_WORKER_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

import concurrent.futures as cf
import logging
import multiprocessing
from typing import Annotated, Any, Iterable, Literal

import lazynwb
import polars as pl
import tqdm
import upath
import utils
from dynamic_routing_analysis import glm_utils, io_utils

logger = logging.getLogger(__name__)

# Required for serializing polars expressions
Expr = Annotated[
    pl.Expr, pydantic.functional_serializers.PlainSerializer(lambda expr: expr.meta.serialize(format='json'), return_type=str)
]

class Params(pydantic_settings.BaseSettings, extra="allow"):
    # ----------------------------------------------------------------------------------
    # Required parameters
    result_prefix: str
    "An identifier for the decoding run, used to name the output files (can have duplicates with different run_id)"
    # ----------------------------------------------------------------------------------

    # Capsule-specific parameters -------------------------------------- #
    single_session_id_to_use: str | None = pydantic.Field(None, exclude=True, repr=True)
    """If provided, only process this session_id. Otherwise, process all sessions that match the filtering criteria"""
    session_table_query: str = (
        'is_ephys & is_task & is_annotated & is_production & project == "DynamicRouting" & issues=="[]"'
    )
    run_id: str = pydantic.Field(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )  # created once at runtime: same for all Params instances
    """A unique string that should be attached to all decoding runs in the same batch"""
    test: bool = pydantic.Field(False, exclude=True)
    logging_level: str | int = pydantic.Field("INFO", exclude=True)
    update_packages_from_source: bool = pydantic.Field(False, exclude=True)
    override_params_json: str | None = pydantic.Field("{}", exclude=True)
    use_process_pool: bool = pydantic.Field(True, exclude=True, repr=True)
    max_workers: int | None = pydantic.Field(None, exclude=True, repr=True)
    """For process pool"""

    # Run parameters that define a unique run (ie will be checked for 'skip_existing')
    datacube_version: str = (
        dynamic_routing_analysis.datacube_utils.get_datacube_version()
    )
    time_of_interest: str = "full_trial"
    input_offsets: bool = True
    input_window_lengths: dict[str, float] = pydantic.Field(default_factory=dict)
    
    # unit inclusion parameters ---------------------------------------- #
    presence_ratio: float | None = 0.7
    decoder_labels_to_exclude: list[str] = pydantic.Field(
        default_factory=lambda: ["noise"]
    )
    
    spike_bin_width: float = 0.1
    """in seconds"""
    areas_to_include: list[str] = pydantic.Field(default_factory=list)
    areas_to_exclude: list[str] = pydantic.Field(default_factory=list)
    orthogonalize_against_context: list[str] = pydantic.Field(
        default_factory=lambda: ["facial_features"]
    )
    trial_start_time: float = -2
    trial_stop_time: float = 3
    intercept: bool = True
    """Whether to include an intercept in the design matrix"""
    method: Literal[
        "ridge_regression",
        "lasso_regression",
        "reduced_rank_regression",
        "elastic_net_regression",
    ] = "ridge_regression"
    no_nested_CV: bool = False
    optimize_on: float = 0.3
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    optimize_penalty_by_cell: bool = False
    optimize_penalty_by_area: bool = False
    optimize_penalty_by_firing_rate: bool = False
    use_fixed_penalty: bool = False
    num_rate_clusters: int = 5

    # RIDGE + ELASTIC NET
    L2_grid_type: Literal["log", "linear"] = "log"
    L2_grid_range: list[float] = pydantic.Field(default_factory=lambda: [1, 2**12])
    L2_grid_num: int = 13
    L2_fixed_lambda: float | None = None

    # LASSO
    L1_grid_type: Literal["log", "linear"] = "log"
    L1_grid_range: list[float] = pydantic.Field(
        default_factory=lambda: [10**-6, 10**-2]
    )
    L1_grid_num: int = 13
    L1_fixed_lambda: float | None = None

    cell_regularization: float | None = None
    cell_regularization_nested: float | None = None

    # ELASTIC NET
    L1_ratio_grid_type: Literal["log", "linear"] = "log"
    L1_ratio_grid_range: list[float] = pydantic.Field(
        default_factory=lambda: [10**-6, 10**-1]
    )
    L1_ratio_grid_num: int = 9
    L1_ratio_fixed: float | None = None
    cell_L1_ratio: float | None = None
    cell_L1_ratio_nested: float | None = None

    # RRR
    rank_grid_num: int = 10
    rank_fixed: int | None = None
    cell_rank: int | None = None
    cell_rank_nested: int | None = None

    reuse_regularization_coefficients: bool = True
    """Whether to re-use regularization coefficients"""

    linear_shift_by: float = 1

    smooth_spikes_half_gaussian: bool = False
    half_gaussian_std_dev: float = 0.05
    features_to_drop: list[str] | None = pydantic.Field(default=None, exclude=True)
    """For modifying via app panel, but needs populating otherwise"""

    linear_shift_variables: list[str] = pydantic.Field(
        default_factory=lambda: ["context"]
    )
    """Will not work with certain feature groups (ie those with offsets)"""

    # Params that will be updated many times during processing (ie for each model) -------------- #
    # project: str | None = pydantic.Field(default=None, exclude=True)
    # drop_variables: list[str] | None = pydantic.Field(default=None, exclude=True)
    # input_variables: list[str] | None = pydantic.Field(default=None, exclude=True)
    # fullmodel_fitted: bool | None = pydantic.Field(default=None, exclude=True)
    # model_label: str | None  = pydantic.Field(default=None, exclude=True)

    @property
    def results_dir(self) -> upath.UPath:
        """Path to general encoding results dir on S3"""
        return upath.UPath("s3://aind-scratch-data/dynamic-routing/encoding/results")

    @property
    def results_folder_name(self) -> str:
        """Name of the results folder on S3, including the run_id"""
        return f"{self.result_prefix}/{self.run_id}"

    @property
    def pkl_data_dir(self) -> upath.UPath:
        """Path to pkl data subfolder within results dir on S3"""
        return self.results_dir / f"fullmodel_pkl_files/{self.results_folder_name}"

    @property
    def json_path(self) -> upath.UPath:
        """Path to params json on S3"""
        return self.results_dir / f"{self.results_folder_name}.json"

    @pydantic.computed_field(repr=False)
    @property
    def unit_inclusion_criteria(self) -> Expr:
        exprs = []
        if self.presence_ratio is not None:
            exprs.append(pl.col("presence_ratio") >= self.presence_ratio)
        if self.decoder_labels_to_exclude:
            exprs.append(pl.col("decoder_labels").is_in(self.decoder_labels_to_exclude).not_())
        if self.areas_to_include:
            exprs.append(pl.col("area").is_in(self.areas_to_include))
        if self.areas_to_exclude:
            exprs.append(pl.col("area").is_in(self.areas_to_exclude).not_())
        return pl.Expr.and_(*exprs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources below is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(
                settings_cls, json_file="parameters.json"
            ),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )


# ------------------------------------------------------------------ #


def get_regularization_coefficients_path(session_id: str) -> pathlib.Path:
    return pathlib.Path(f"/scratch/{session_id}_regularization_coefficients.pkl")


def get_regularization_coefficients(session_id: str) -> dict[str, Any]:
    return pickle.loads(get_regularization_coefficients_path(session_id).read_bytes())


def get_fullmodel_data_path(session_id: str) -> pathlib.Path:
    return pathlib.Path(f"/scratch/{session_id}.pkl")


def get_fullmodel_data(session_id: str, params: Params) -> dict[str, dict]:
    """Equivalent to reading the 'inputs.npz' file in the pipeline.
    If it doesn't exist, it will be created.
    """
    if get_fullmodel_data_path(session_id).exists():
        data = pickle.loads(get_fullmodel_data_path(session_id).read_bytes())
        if params.reuse_regularization_coefficients:
            data["fit"] |= pickle.loads(
                get_regularization_coefficients_path(session_id).read_bytes()
            )
        return data
    else:
        units_table, behavior_info = io_utils.get_session_data_from_datacube(session_id)
        print(units_table.columns)
        units_table = (
            units_table
            # filter first, then get spike times for subset of units
            .filter(params.unit_inclusion_criteria)
            .pipe(
                lazynwb.merge_array_column, "spike_times"
            )
            .pipe(
                lazynwb.merge_array_column, "obs_intervals"
            )
        )
        run_params = params.model_dump()
        run_params |= {
                        "fullmodel_fitted": False,
                        "model_label": "fullmodel",
                        "project":  get_project(session_id)
                    }
        run_params = io_utils.define_kernels(run_params)


        fit: dict[str, Any] = {}
        fit = io_utils.establish_timebins(
            run_params=run_params, fit=fit, behavior_info=behavior_info
        )
        fit = io_utils.process_spikes(
            units_table=units_table, run_params=run_params, fit=fit
        )
        design: io_utils.DesignMatrix = io_utils.DesignMatrix(fit)
        design, fit = io_utils.add_kernels(
            design=design,
            run_params=run_params,
            session=utils.get_nwb(session_id),
            fit=fit,
            behavior_info=behavior_info,
        )
        design_matrix = design.get_X()
        data = {"fit": fit, "design_matrix": design_matrix, "run_params": run_params}
        get_fullmodel_data_path(session_id).write_bytes(pickle.dumps(data))
        if params.test:
            pathlib.Path(
                get_fullmodel_data_path(session_id)
                .as_posix()
                .replace("scratch", "results")
            ).write_bytes(pickle.dumps(data))
        return data


def get_project(session_id: str) -> str:
    return utils.get_session_table().filter(pl.col("session_id") == session_id)[
        "project"
    ][0]


def helper_fullmodel(session_id: str, params: Params) -> None:
    data = get_fullmodel_data(session_id=session_id, params=params)
    run_params = data["run_params"]
    run_params |= {
        "fullmodel_fitted": False,
        "model_label": "fullmodel",
    }
    fit = glm_utils.optimize_model(
        fit=data["fit"], design_mat=data["design_matrix"], run_params=run_params
    )
    fit = glm_utils.evaluate_model(
        fit=fit, design_mat=data["design_matrix"], run_params=run_params
    )
    regularization_coef_dict = {}
    for prefix in ["cell_regularization", "cell_L1_ratio", "cell_rank"]:
        for suffix in ["", "_nested"]:
            key = f"{prefix}{suffix}"
            regularization_coef_dict[key] = fit[key]
    get_regularization_coefficients_path(session_id).write_bytes(
        pickle.dumps(regularization_coef_dict)
    )
    save_results(session_id=session_id, fit=fit, run_params=run_params, params=params)


def get_features_to_drop(session_id: str, params: Params) -> list[str]:
    if params.features_to_drop:
        return params.features_to_drop

    run_params = get_fullmodel_data(session_id=session_id, params=params)["run_params"]
    features_to_drop = list(run_params["input_variables"]) + [
        run_params["kernels"][key]["function_call"]
        for key in run_params["input_variables"]
    ]
    return features_to_drop


def helper_dropout(session_id: str, params: Params, feature_to_drop: str) -> None:
    data = get_fullmodel_data(session_id=session_id, params=params)
    run_params = data["run_params"]
    run_params |= {
        "drop_variables": [feature_to_drop],
        "fullmodel_fitted": params.reuse_regularization_coefficients,
        "model_label": f"drop_{feature_to_drop}",
    }
    fit = glm_utils.dropout(
        fit=data["fit"] | get_regularization_coefficients(session_id),
        design_mat=data["design_matrix"],
        run_params=run_params,
    )
    save_results(session_id=session_id, fit=fit, run_params=run_params, params=params)


def helper_linear_shift(
    session_id: str,
    params: Params,
    shift: int,
    blocks: Iterable[int],
    shift_columns: list[int],
) -> None:
    data = get_fullmodel_data(session_id=session_id, params=params)
    run_params = data["run_params"]
    run_params |= {
        "fullmodel_fitted": params.reuse_regularization_coefficients,
        "model_label": f"shift_{shift}",
    }
    fit = glm_utils.apply_shift_to_design_matrix(
        fit=data["fit"] | get_regularization_coefficients(session_id),
        design_mat=data["design_matrix"],
        run_params=run_params,
        blocks=blocks,
        shift_columns=shift_columns,
        shift=shift,
    )
    save_results(session_id=session_id, fit=fit, run_params=run_params, params=params)


def get_shift_columns(session_id: str, params: Params) -> list[int]:
    return [
        i
        for i, label in enumerate(
            get_fullmodel_data(session_id=session_id, params=params)["design_matrix"] # type: ignore
            .coords["weights"] # type: ignore
            .values
        )
        if any([key in label for key in params.linear_shift_variables])
    ]


def get_linear_shifts(
    session_id: str, params: Params
) -> tuple[Iterable[int], Iterable[int]]:
    data = get_fullmodel_data(session_id=session_id, params=params)
    return glm_utils.get_shift_bins(
        run_params=params.model_dump(), # type: ignore
        fit=data["fit"],
        context=data["design_matrix"].sel(weights="context_0").data,# type: ignore
    )


def save_results(
    session_id: str, fit: dict[str, Any], params: Params, run_params: dict[str, Any]
) -> None:
    fit.pop("timebins_all", None)
    fit.pop("bin_centers_all", None)
    fit.pop("epoch_trace_all", None)
    fit.pop("mask", None)
    fit["spike_count_arr"].pop("spike_counts", None)
    if run_params["model_label"] == "fullmodel":
        pkl_path = params.pkl_data_dir / f"{session_id}.pkl"
        logger.info(f"Writing fullmodel data to {pkl_path}")
        pkl_path.write_bytes(pickle.dumps({"fit": fit, "run_params": run_params}))

    if run_params["model_label"] == "fullmodel":
        dropped_variable = None
    else:
        dropped_variable = "_".join(run_params["model_label"].split("_")[1:])
    if "shift" in run_params["model_label"]:
        shift_index = int(run_params["model_label"].split("_")[-1])
    else:
        shift_index = None

    # save some contents of fit as parquet on S3
    parquet_path = (
        params.results_dir
        / params.results_folder_name
        / f"{session_id}_{run_params['model_label']}.parquet"
    )
    logger.info(f"Writing results to {parquet_path}")
    (
        pl.DataFrame(
            {
                "session_id": session_id,
                "unit_id": fit["spike_count_arr"]["unit_id"].tolist(),
                "project": get_project(session_id),
                "cv_test": fit[run_params["model_label"]]["cv_var_test"].tolist(),
                "cv_train": fit[run_params["model_label"]]["cv_var_train"].tolist(),
                "weights": fit[run_params["model_label"]]["weights"].T.tolist(),
                "dropped_variable": dropped_variable,
                "shift_index": shift_index,
            },
            schema_overrides={
                "shift_index": pl.Int32,
                "drop_variable": pl.String,
            },
        ).write_parquet(
            parquet_path.as_posix(),
            compression_level=18,
            statistics="full",
        )
    )


def run_encoding(
    session_ids: str | Iterable[str],
    params: Params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    # TODO add skip_existing option

    if not params.use_process_pool:
        for session_id in tqdm.tqdm(
            session_ids,
            total=len(tuple(session_ids)),
            unit="session",
            desc="Encoding",
        ):
            logger.info(f"Processing session {session_id}")
            helper_fullmodel(session_id=session_id, params=params)
            for feature_to_drop in get_features_to_drop(
                session_id=session_id, params=params
            ):
                helper_dropout(
                    session_id=session_id,
                    params=params,
                    feature_to_drop=feature_to_drop,
                )
                if params.test:
                    logger.info("Test mode: exiting after first feature dropout")
                    break
            shifts, blocks = get_linear_shifts(session_id=session_id, params=params)
            for shift in shifts:
                helper_linear_shift(
                    session_id=session_id,
                    params=params,
                    shift=shift,
                    blocks=blocks,
                    shift_columns=get_shift_columns(
                        session_id=session_id, params=params
                    ),
                )
                if params.test:
                    logger.info("Test mode: exiting after first shift")
                    break

    else:
        future_to_session: dict[cf.Future, str] = {}
        lock = None  # multiprocessing.Manager().Lock() # or None
        with cf.ProcessPoolExecutor(
            max_workers=params.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for session_id in session_ids:
                future = executor.submit(
                    helper_fullmodel,
                    session_id=session_id,
                    params=params,
                )

                def run_after_full_model(future: cf.Future) -> None:
                    for feature_to_drop in get_features_to_drop(
                        session_id=session_id, params=params
                    ):
                        helper_dropout(
                            session_id=session_id,
                            params=params,
                            feature_to_drop=feature_to_drop,
                        )
                        if params.test:
                            logger.info("Test mode: exiting after first feature dropout")
                            break
                    shifts, blocks = get_linear_shifts(
                        session_id=session_id, params=params
                    )
                    for shift in shifts:
                        helper_linear_shift(
                            session_id=session_id,
                            params=params,
                            shift=shift,
                            blocks=blocks,
                            shift_columns=get_shift_columns(
                                session_id=session_id, params=params
                            ),
                        )
                        if params.test:
                            logger.info("Test mode: exiting after first shift")
                            break

                future.add_done_callback(run_after_full_model)

                future_to_session[future] = session_id
                logger.debug(
                    f"Submitted encoding to process pool for session {session_id}"
                )
                if params.test:
                    logger.info("Test mode: exiting after first session")
                    break
            for future in tqdm.tqdm(
                cf.as_completed(future_to_session),
                total=len(future_to_session),
                unit="session",
                desc="Encoding",
            ):
                session_id = future_to_session[future]
                try:
                    _ = future.result()
                except Exception:
                    logger.exception(f"{session_id} | Failed:")
