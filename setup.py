from setuptools import setup, find_packages
setup(
    name='multiagent_mpc',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gym', 'numpy', 'torch', 'cvxpy'
    ],
)