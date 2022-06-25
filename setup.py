import setuptools


setuptools.setup(
    name="imps_tools",
    version="0.0.1",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.1",
        "scipy>=1.8.1",
        "pytest"
    ]
)