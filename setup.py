from setuptools import setup, find_packages

setup(
    name="whlayout",
    version="0.1.0",
    description="Warehouse block layout optimization models and utilities",
    author="Breno Alves Beirigo",
    author_email="brenobeirigo@gmail.com",
    packages=find_packages(where="src"),
    package_data={"whlayout": ["instances/17_depts/*.csv"]},
    package_dir={"": "src"},
    install_requires=[
        "gurobipy",
        "pandas",
        "matplotlib",
        "plotly"
    ],
)
