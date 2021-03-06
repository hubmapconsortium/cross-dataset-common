from setuptools import setup, find_packages

setup(
    name='cross_dataset_common',
    version='0.1.5',
    description='Utilities for preparing processed data to be loaded into database',
    url='https://github.com/hubmapconsortium/cross-dataset-common',
    author='Sean Donahue',
    author_email='seandona@andrew.cmu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    install_requires=[
        'anndata>=0.7.3',
        'requests>=2.22.0',
        'pyyaml>=5.3',
        'scanpy>=1.4.4',
    ],
    python_requires='>=3.6',
)
