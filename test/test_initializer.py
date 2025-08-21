from unittest import TestCase

from iddm.utils import network_initializer


class Test(TestCase):
    def test_network_initializer(self):
        """
        Test network initializer
        This test checks if the network initializer can be imported without errors.
        """
        try:
            network = "unet-flash-self-attn"
            device = "cpu"
            network_model = network_initializer(network=network, device=device)
        except Exception as e:
            self.fail(f"Network initializer failed with error: {e}")
        self.assertIsNotNone(network_model, "Network model should not be None")
        self.assertTrue(hasattr(network_model, "forward"), "Network model should have a forward method")
