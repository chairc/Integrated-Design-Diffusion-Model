#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    @Date   : 2025/8/21 15:49
    @Author : chairc
    @Site   : https://github.com/chairc
"""
from unittest import TestCase
from iddm.utils.check import check_package_is_exist


class Test(TestCase):
    def test_check_package_is_exist(self):
        """
        Test if the package is installed.
        This test checks if the 'flash_attn' package is installed in the environment.
        If the package is installed, the test will pass; otherwise, it will fail.
        1. Import the `check_package_is_exist` function from `iddm.utils.check`.
        2. Call the function with the package name 'flash_attn'.
        3. Assert that the function returns `True`, indicating that the package is installed.
        :return: None
        """
        self.assertTrue(check_package_is_exist(package_name="flash_attn"))
