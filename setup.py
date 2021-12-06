from distutils.core import setup
import numpy as np



setup(name='SessionAnalysis',
      version='1.0',
      description='Basic timeseries organization of neural data.',
      author='Nathan Hall',
      author_email='nathan.hall@duke.edu',
      url='https://',
      packages=['SessionAnalysis'],
      include_dirs=[np.get_include()],
     )
