from setuptools import setup

setup(name='DGD',
      version='0.1.0',
      description='python implementation of direct representation learning via MAP estimation',
      author=['Anders Krogh','Viktoria Schuster'],
      license='MIT',
      url='https://github.com/Center-for-Health-Data-Science/DGD_paper',
      packages=['DGD'],
      install_requires=[
            'torch',
            'numpy'
      ]
     )