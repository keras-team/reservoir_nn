# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installer of reservoir_nn."""

from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name='reservoir_nn',
    version='0.1.0',
    packages=find_namespace_packages(include=['reservoir_nn.*'],),
    install_requires=[
        'absl-py',
        'deprecation',
        'immutabledict',
        'keras',
        'matplotlib',
        'ml_collections',
        'numpy',
        'networkx',
        'scipy',
        'tensorflow',
        'tensorflow_addons',
    ],
)
