from amplify import FixstarsClient
from amplify import GurobiClient
from amplify import DWaveSamplerClient
from datetime import timedelta
import os
from dotenv import load_dotenv


def GetFixstarClient():
    """Create Fixstars client, make sure FIXSTAR_TOKEN is set in .env file

    Raises:
        Exception: Throws if FIXSTAR_TOKEN is not defined

    Returns:
        FixstarsClient: Created client
    """

    client = FixstarsClient()
    load_dotenv()
    FIXSTAR_TOKEN = os.getenv('FIXSTAR_TOKEN')

    if FIXSTAR_TOKEN is None:
        raise Exception("Please set FIXSTAR_TOKEN in .env file")

    client.token = FIXSTAR_TOKEN
    client.parameters.timeout = 1000
    return client


def GetGurobiClient(
        library_path="/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib"
):
    """Create Gurobi Client, Gurobi must be installed with usable license file

    Args:
        library_path (str, optional): Library path for Gurobi. Defaults to "/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib" (Recent MacOS at the time of writing)

    Returns:
        GurobiClient: Created client
    """

    client = GurobiClient()
    client.library_path = library_path
    client.parameters.time_limit = timedelta(seconds=100)
    return client


def GetDWaveClient(solver="Advantage_system4.1"):
    """Create DWave Client, make sure DWAVE_TOKEN is set in .env file

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
    client.parameters.num_reads = 1000
    return client
