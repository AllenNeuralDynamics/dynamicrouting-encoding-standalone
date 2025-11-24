# dynamicrouting-encoding-tests

This capsule runs the GLM for either a single session (if session ID is provided), or loops over all sessions.

Use this script to launch one capsule per session. The master capsule should be set to 2 CPU/16 GB (1 CPU/8 GB works for most sessions, but a few require more memory).

```python
import json
import pathlib
import time

import aind_session
import codeocean.capsule
import codeocean.computation
import codeocean.data_asset
import polars as pl
import tqdm

client = aind_session.get_codeocean_client()

RESULT_PREFIX = 'v268'
RUN_ID = "1"

def run_encoding(session_id: str):
    run_params = codeocean.computation.RunParams(
        capsule_id="63b499b3-cfde-4224-8c38-c5fb3d541e34",  # write cache
        named_parameters=[
            codeocean.computation.NamedRunParam(
                param_name="single_session_id_to_use",
                value=session_id,
            ),
            codeocean.computation.NamedRunParam(
                param_name="result_prefix",
                value=str(RESULT_PREFIX),  # required
            ),
            codeocean.computation.NamedRunParam(
                param_name="run_id",
                value=str(RUN_ID),
            ),
            codeocean.computation.NamedRunParam(
                param_name="test",
                value="0",  # all values must be supplied as strings
            ),
            codeocean.computation.NamedRunParam(
                param_name="use_process_pool",
                value="False",  # all values must be supplied as strings
            ),
            # codeocean.computation.NamedRunParam(
            #     param_name="override_params_json",
            #     value='{"time_of_interest": "quiescent"}',  # all values must be supplied as strings
            # ),
        ],
    )
    computation = client.computations.run_capsule(run_params)
    return computation


session_ids = (
    pl.scan_parquet(
        "s3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet"
    )
    .filter(
        "is_ephys",
        "is_task",
        "is_annotated",
        "is_production",
        pl.col("issues").list.len() == 0,
    )
    .collect()
)["session_id"]

session_ids_16gb = ["713655_2024-08-07", "706401_2024-04-22"]

session_to_computation = {}
for session_id in tqdm.tqdm(session_ids, desc="Sessions", unit="session"):
    # if session_id in session_ids_16gb:
    #     print(f"Skipping {session_id} because it is requires more than a capsule with more than 8GB memory")
    #     continue
    print(session_id)
    session_to_computation[session_id] = run_encoding(session_id).id
    pathlib.Path("computations.json").write_text(
        json.dumps(session_to_computation, indent=4)
    )
    time.sleep(3)  # Sleep to avoid hitting the API rate limit

## To delete all computations from previous run --------------------- #
# for session, id_ in json.loads(pathlib.Path("computations.json").read_text()).items():
#     client.computations.delete_computation(id_)
# quit()

for session, id_ in json.loads(pathlib.Path("computations.json").read_text()).items():
    computation = client.computations.wait_until_completed(client.computations.get_computation(id_))
    
if computation.end_status != codeocean.computation.ComputationEndStatus.Succeeded:
    print(f'at least one computation failed - not writing consolidated results: {computation=}')
    quit()
    
print('writing consolidated results parquet files to S3')
client.computations.run_capsule(
    codeocean.computation.RunParams(
        capsule_id="1003b011-db1f-4c50-b2d3-df7f9dc1dc6e",  
        parameters=[str(RESULT_PREFIX), str(RUN_ID)],
    )
)
```