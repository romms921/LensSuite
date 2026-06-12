import os
import re
import glob
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm


# Identify number of systems in the directory
num_systems = len(os.listdir('./')) - 4 # Subtracting 4 to account for the files that are not systems and hidden files
print(f'Number of systems: {num_systems}')


