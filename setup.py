from setuptools import setup, find_packages

setup(
    name='idfloodPy',
    version='0.1.0',
    description='Flood event identification and separation package',
    author='Yuhan Guo',
    author_email='guoyuhan@mail.tsinghua.edu.cn',
    url='https://github.com/yourusername/idfloodPy',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'baseflow',
         "os",
         "warnings"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
