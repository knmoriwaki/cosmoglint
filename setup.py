from setuptools import setup, find_packages

setup(
    name="cosmoglint",
    version="2.0.0",
    author="Kana Moriwaki",
    url="https://github.com/knmoriwaki/cosmoglint",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "h5py",
        "nflows",
        "astropy",
    ],
    python_requires=">=3.9"
)

