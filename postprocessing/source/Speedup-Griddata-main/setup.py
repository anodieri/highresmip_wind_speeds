from setuptools import setup

setup(
    name='',
    version='1.0',
    py_modules=['speedup_griddata'],
    install_requires=[
        'numpy',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'speedup-griddata = speedup_griddata:main',
        ],
    },
    author='Tianshi Du',
    author_email='dutianshi@hotmail.com',
    description='Speed up the scipy.interpolate.griddata',
    url='https://github.com/DuTianshi/Speedup-Griddata.git',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
