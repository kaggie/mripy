from setuptools import setup, find_packages

setup(
    name='hyperpolarized_mri',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'hyperpolarized-mri-cli=hyperpolarized_mri.cli:main'
        ]
    }
)