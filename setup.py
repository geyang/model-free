from setuptools import setup, find_packages

with open('VERSION', 'r') as f:
    __version__ = f.read()

setup(name='dmc_gen',
      packages=find_packages(),
      install_requires=[
          "baselines",
          "pyYaml",
          "gym",
          "jaynes",
          "ml-dash",
          "ml-logger",
          "numpy",
          "seaborn",
          "torch",
          "tqdm",
          "matplotlib",
          "more-itertools",
          "mujoco-py"
      ],
      description='plan2vec',
      author='Ge Yang',
      url='https://github.com/geyang/dmc_gen',
      author_email='ge.ike.yang@gmail.com',
      version=__version__)
