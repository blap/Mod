import sys
import os
import unittest
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference_pio.core.engine.backend import Tensor

class TestMoEDispatch(unittest.TestCase):
    def test_moe_primitives(self):
        # Setup
        # Input: [4, 2] (4 tokens, hidden=2)
        data = Tensor([4, 2])
        data.load([
            1.0, 1.0,  # Token 0
            2.0, 2.0,  # Token 1
            3.0, 3.0,  # Token 2
            4.0, 4.0   # Token 3
        ])

        # Indices: [4] -> Expert assignments
        # Token 0 -> Expert 0
        # Token 1 -> Expert 1
        # Token 2 -> Expert 0
        # Token 3 -> Expert 1
        indices = Tensor([4])
        indices.load([0.0, 1.0, 0.0, 1.0])

        # Test 1: Count Value
        count_0 = indices.count_value(0.0)
        self.assertEqual(count_0, 2)
        count_1 = indices.count_value(1.0)
        self.assertEqual(count_1, 2)

        # Test 2: Gather by Value (Expert 0)
        gathered_0, orig_idx_0 = data.gather_by_value(indices, 0.0)
        self.assertEqual(gathered_0.shape, (2, 2))

        # With OpenMP, order is non-deterministic. Sort by original index to verify.
        g_list = gathered_0.to_list()
        idx_list = orig_idx_0.to_list()

        results = []
        for i in range(2):
            idx = idx_list[i]
            vals = g_list[i*2 : (i+1)*2]
            results.append((idx, vals))

        results.sort(key=lambda x: x[0])

        # Expect (0.0, [1.0, 1.0]) and (2.0, [3.0, 3.0])
        self.assertAlmostEqual(results[0][0], 0.0)
        self.assertAlmostEqual(results[0][1][0], 1.0)
        self.assertAlmostEqual(results[1][0], 2.0)
        self.assertAlmostEqual(results[1][1][0], 3.0)

        # Test 3: Scatter Add (Simulate Expert Output)
        # We need to create expert output corresponding to gathered results
        # If order was swapped, we must be careful.
        # But `scatter_add_by_index` uses `orig_idx_0`.
        # So if `gathered_0` has Row A at pos 0, and `orig_idx_0` has Index A at pos 0.
        # Expert processes Row A -> Output A at pos 0.
        # Scatter puts Output A at Index A.
        # So it handles reordering correctly!

        # Simulate expert: add 10 to input
        expert_out_0 = Tensor([2, 2])
        # We process `gathered_0`
        g_vals = gathered_0.to_list()
        out_vals = [v + 10.0 for v in g_vals]
        expert_out_0.load(out_vals)

        # Output buffer [4, 2] init 0
        out = Tensor([4, 2])
        out.fill(0.0)

        out.scatter_add_by_index(expert_out_0, orig_idx_0)

        # Verify
        res_out = out.to_list()
        # [11, 11, 0, 0, 13, 13, 0, 0]
        self.assertAlmostEqual(res_out[0], 11.0) # Token 0
        self.assertAlmostEqual(res_out[2], 0.0)  # Token 1
        self.assertAlmostEqual(res_out[4], 13.0) # Token 2

        print("MoE Primitives Test Passed")

if __name__ == '__main__':
    unittest.main()
