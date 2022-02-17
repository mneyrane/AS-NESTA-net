from setuptools import setup, find_packages

setup(
    name='NESTA-net',
    version='1.0',
    description='Unrolled NESTA for compressive imaging',
    url='https://github.com/mneyrane/NESTA-net',
    license='MIT',
    python_requires='>=3.9',
    install_requires=['torch', 'numpy', 'Pillow', 'scipy', 'matplotlib', 'seaborn'],
    packages=find_packages(),
)
