from amplify import FixstarsClient
from amplify import GurobiClient
from amplify import DWaveSamplerClient
from datetime import timedelta
import os
from dotenv import load_dotenv
from pathlib import Path
from amplify import solve

fixstars_client = None


def CreateFixstarsClient(timeout=1000):
    """
    Create New Fixstars client, make sure FIXSTAR_TOKEN is set in .env file.

    Raises:
        Exception: Throws if FIXSTAR_TOKEN is not defined

    Returns:
        FixstarsClient: Created client
    """

    def load_env():
        # Adjusts to the .env location
        env_path = Path(__file__).resolve().parent / '.env'
        load_dotenv(dotenv_path=env_path)

    client = FixstarsClient()
    load_env()
    FIXSTAR_TOKEN = os.getenv('FIXSTAR_TOKEN')

    if FIXSTAR_TOKEN is None:
        raise Exception("Please set FIXSTAR_TOKEN in .env file")

    client.token = FIXSTAR_TOKEN
    client.parameters.timeout = timeout

    return client


def GetFixstarsClient(timeout=1000):
    """
    Create Fixstars Client if not exists, make sure FIXSTAR_TOKEN is set in .env file.
    Parameter will not have effect if client already exists.

    Args:
        timeout (int, optional): Timeout for Fixstars client. Defaults to 1000 milliseconds.

    Raises:
        Exception: Throws if FIXSTAR_TOKEN is not defined

    Returns:
        FixstarsClient: Fixstars client
    """

    global fixstars_client

    if fixstars_client is not None:
        return fixstars_client

    client = CreateFixstarsClient(timeout)
    fixstars_client = client
    return client


gurobi_client = None


def CreateGurobiClient(
        library_path="/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib",
        timeout_sec=100
):
    """Create New Gurobi Client, Gurobi must be installed with usable license file.

    Args:
        library_path (str, optional): Library path for Gurobi. Defaults to "/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib" (Recent MacOS at the time of writing)

    Returns:
        GurobiClient: Created client
    """

    client = GurobiClient()
    client.library_path = library_path
    client.parameters.time_limit = timedelta(seconds=timeout_sec)
    return client


def GetGurobiClient(
        library_path="/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib",
        timeout_sec=100
):
    """
    Create Gurobi Client if not exists, Gurobi must be installed with usable license file.
    Parameter will not have effect if client already exists.

    Args:
        library_path (str, optional): Library path for Gurobi. Defaults to "/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib" (Recent MacOS at the time of writing)

    Returns:
        GurobiClient: Created client
    """

    global gurobi_client

    if gurobi_client is not None:
        return gurobi_client

    client = CreateGurobiClient(library_path, timeout_sec)
    return client


dwave_client = None


def CreateDWaveClient(solver="Advantage_system4.1", num_reads=1000):
    """Create New DWave Client, make sure DWAVE_TOKEN is set in .env file.

    Raises:
        Exception: Throws if DWAVE_TOKEN is not defined

    Returns:
        DWaveSamplerClient: Created client
    """

    client = DWaveSamplerClient()
    load_dotenv()
    DWAVE_TOKEN = os.getenv('DWAVE_TOKEN')

    if DWAVE_TOKEN is None:
        raise Exception("Please set DWAVE_TOKEN in .env file")

    client.token = DWAVE_TOKEN
    client.solver = solver
    client.parameters.num_reads = num_reads
    return client


def GetDWaveClient(solver="Advantage_system4.1", num_reads=1000):
    """
    Create DWave Client if not exists, make sure DWAVE_TOKEN is set in .env file.
    Parameter will not have effect if client already exists.

    Raises:
        Exception: Throws if DWAVE_TOKEN is not defined

    Returns:
        DWaveSamplerClient: Created client
    """

    global dwave_client

    if dwave_client is not None:
        return dwave_client

    client = CreateDWaveClient(solver, num_reads)
    return client


def RunSimulation(models):
    clientFS = GetFixstarsClient()
    clientG = GetGurobiClient()
    clientDWave = GetDWaveClient()
    for i, tsp in enumerate(models):
        print(f'Run {i+1}')

        resultFS = solve(tsp, clientFS)
        print('Fixstars Run')
        print(f'Best score: {resultFS.best.objective}')
        print(f'Best values: {resultFS.best.values}')
        print(f'Execution time: {resultFS.execution_time}')

        resultG = solve(tsp, clientG)
        print('Gurobi Run')
        print(f'Best score: {resultG.best.objective}')
        print(f'Best values: {resultG.best.values}')
        print(f'Execution time: {resultG.execution_time}')

        resultDWave = solve(tsp, clientDWave)
        print('D-Wave Run')
        print(f'Best score: {resultDWave.best.objective}')
        print(f'Best values: {resultDWave.best.values}')
        print(f'Execution time: {resultDWave.execution_time}')
