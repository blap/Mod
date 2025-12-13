#!/usr/bin/env python3
"""
Test suite for the Buddy Allocator implementation in advanced_memory_pooling_system.py
"""

import unittest
from advanced_memory_pooling_system import BuddyAllocator, MemoryBlock, TensorType


class TestBuddyAllocator(unittest.TestCase):
    def setUp(self):
        # Initialize with 1024 bytes for testing (must be power of 2)
        self.allocator = BuddyAllocator(total_size=1024, min_block_size=64)

    def test_initialization(self):
        """Test that the buddy allocator initializes correctly"""
        # Check that initially we have one block of total size at the highest level
        self.assertEqual(len(self.allocator.free_blocks[4]), 1)  # Level 4: 64 * 2^4 = 1024
        block = list(self.allocator.free_blocks[4])[0]
        self.assertEqual(block.size, 1024)
        self.assertEqual(block.start_addr, 0)
        self.assertTrue(block.is_free)

    def test_get_buddy_addr(self):
        """Test the _get_buddy_addr method with various inputs"""
        # Test cases: (address, size, expected_buddy_address)
        test_cases = [
            (0, 64, 64),      # Buddy of block at 0 with size 64 is at 64
            (64, 64, 0),      # Buddy of block at 64 with size 64 is at 0
            (0, 128, 128),    # Buddy of block at 0 with size 128 is at 128
            (128, 128, 0),    # Buddy of block at 128 with size 128 is at 0
            (256, 128, 384),  # Buddy of block at 256 with size 128 is at 384
            (384, 128, 256),  # Buddy of block at 384 with size 128 is at 256
        ]

        for addr, size, expected in test_cases:
            with self.subTest(addr=addr, size=size):
                result = self.allocator._get_buddy_addr(addr, size)
                self.assertEqual(result, expected, f"Buddy of block at {addr} with size {size} should be at {expected}, got {result}")

    def test_basic_allocation(self):
        """Test basic allocation functionality"""
        block = self.allocator.allocate(64, TensorType.KV_CACHE, "test_1")
        self.assertIsNotNone(block)
        self.assertEqual(block.size, 64)
        self.assertEqual(block.start_addr, 0)
        self.assertFalse(block.is_free)
        
        # Check that the original block was split
        self.assertEqual(len(self.allocator.free_blocks[4]), 0)  # Level 4 should be empty
        self.assertEqual(len(self.allocator.free_blocks[3]), 1)  # Level 3 should have one block (512-byte)
        self.assertEqual(len(self.allocator.free_blocks[2]), 1)  # Level 2 should have one block (256-byte)
        self.assertEqual(len(self.allocator.free_blocks[1]), 1)  # Level 1 should have one block (128-byte)
        self.assertEqual(len(self.allocator.free_blocks[0]), 1)  # Level 0 should have one block (64-byte) - the remaining half

    def test_allocation_and_deallocation_with_merge(self):
        """Test allocation and deallocation with buddy merging"""
        # Allocate two adjacent blocks
        block1 = self.allocator.allocate(64, TensorType.KV_CACHE, "test_1")
        block2 = self.allocator.allocate(64, TensorType.KV_CACHE, "test_2")
        
        self.assertIsNotNone(block1)
        self.assertIsNotNone(block2)
        
        # Free the first block
        self.allocator.deallocate(block1)
        
        # At this point, we should have one 64-byte block free and one allocated
        # The free block should be in level 0 (64-byte blocks)
        free_blocks_level_0 = list(self.allocator.free_blocks[0])
        self.assertEqual(len(free_blocks_level_0), 1)
        
        # Free the second block - this should trigger a merge
        self.allocator.deallocate(block2)
        
        # Now we should have all blocks merged back to the original 1024-byte block
        # Check if we have the original 1024-byte block back
        levels_with_free_blocks = [level for level, blocks in self.allocator.free_blocks.items() if len(blocks) > 0]
        # After merging, we should have the 1024-byte block back at the highest level
        self.assertIn(4, levels_with_free_blocks)  # Level 4 should have the 1024-byte block
        self.assertEqual(len(self.allocator.free_blocks[4]), 1)
        
    def test_buddy_merge_logic(self):
        """Test that buddy merging works correctly"""
        # Start fresh
        allocator = BuddyAllocator(total_size=256, min_block_size=32)
        
        # Allocate 4 blocks of minimum size (32 bytes each)
        blocks = []
        for i in range(4):
            block = allocator.allocate(32, TensorType.KV_CACHE, f"test_{i}")
            self.assertIsNotNone(block)
            blocks.append(block)
        
        # Free all blocks
        for block in blocks:
            allocator.deallocate(block)
        
        # After freeing and merging, we should have the original 256-byte block back
        total_free_blocks = sum(len(blocks) for blocks in allocator.free_blocks.values())
        self.assertEqual(total_free_blocks, 1)  # Should have only 1 block (the original 256-byte block)
        
        # The block should be at the highest level (level 3: 32 * 2^3 = 256)
        self.assertEqual(len(allocator.free_blocks[3]), 1)
        main_block = list(allocator.free_blocks[3])[0]
        self.assertEqual(main_block.size, 256)
        self.assertEqual(main_block.start_addr, 0)

    def test_alignment_properties(self):
        """Test that addresses are properly aligned to block sizes"""
        allocator = BuddyAllocator(total_size=512, min_block_size=64)
        
        # Allocate different sized blocks and check alignment
        for size in [64, 128, 256]:
            block = allocator.allocate(size, TensorType.KV_CACHE, f"test_{size}")
            self.assertIsNotNone(block)
            # Block addresses should be aligned to the block size
            self.assertEqual(block.start_addr % size, 0, f"Address {block.start_addr} not aligned to size {size}")
            
            # Return the block for next iteration
            allocator.deallocate(block)


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()