#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://deep_dss.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='deep_dss',
    version='0.1.0',
    description='Investigating the use of graph convolutional neural networks for constraining cosmological parameters.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Sreyas Adiraju',
    author_email='adiraju21s@ncssm.edu',
    url='https://github.com/adiraju21s/deep_dss',
    packages=[
        'deep_dss',
    ],
    package_dir={'deep_dss': 'deep_dss'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='deep_dss',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
)
