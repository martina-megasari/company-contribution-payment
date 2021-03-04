import os
import json
# from setuptools import find_packages, setup
#
# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='This project tries to forecast the monthly company contribution',
#     author='martina.megasari@smartpension.co.uk',
#     license='',
# )

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as f:
    CONFIG = json.load(f)
