from setuptools import setup, find_packages

setup(
    name='AS-NESTA-net',
    version='1.0',
    description='Unrolled NESTA for inverse problems with analysis-sparse models',
    url='https://github.com/mneyrane/AS-NESTA-net',
    license='MIT',
    python_requires='>=3.9',
    install_requires=['torch', 'numpy', 'Pillow', 'scipy', 'matplotlib', 'seaborn'],
    packages=find_packages(),
)
