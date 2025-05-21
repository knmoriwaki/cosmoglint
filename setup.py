from setuptools import setup, find_packages

setup(
    name="lim-mock-generator",
    version="0.1.0",
    author="Kana Moriwaki",
    url="https://github.com/knmoriwaki/lim-mock-generator",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "h5py",
    ],
    python_requires=">=3.8"
)

