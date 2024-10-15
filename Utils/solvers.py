from amplify import FixstarsClient
from amplify import GurobiClient
from amplify import DWaveSamplerClient
from datetime import timedelta
import os
from dotenv import load_dotenv

FIXSTAR_TOKEN = os.getenv('FIXSTAR_TOKEN')
def GetFixstarClient():
    client = FixstarsClient()
    load_dotenv()
    FIXSTAR_TOKEN = os.getenv('FIXSTAR_TOKEN')
    client.token = FIXSTAR_TOKEN
    client.parameters.timeout = 1000 
    return client

def GetGurobiClient():
    client = GurobiClient()
    client.library_path = "/Library/gurobi1103/macos_universal2/lib/libgurobi110.dylib"
    client.parameters.time_limit = timedelta(seconds=100)
    return client

def GetDWaveClient():
    client = DWaveSamplerClient()
    load_dotenv()
    DWAVE_TOKEN = os.getenv('DWAVE_TOKEN')
    client.token = DWAVE_TOKEN
    client.solver = "Advantage_system4.1"
    client.parameters.num_reads = 1000 
    return client