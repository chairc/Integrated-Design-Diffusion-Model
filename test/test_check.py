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
