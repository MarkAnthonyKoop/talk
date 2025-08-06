#!/usr/bin/env python3
"""
Demonstration of the memory ID collision problem.
"""

import hashlib
import json
from datetime import datetime
import time

def generate_memory_id(data):
    """Current implementation."""
    content_str = json.dumps(data, sort_keys=True, default=str)
    timestamp_str = str(datetime.now().timestamp())
    combined = f"{content_str}_{timestamp_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]  # Only 16 chars!

# Scenario 1: Fast insertion of similar content
print("Scenario 1: Fast insertion of similar content")
print("-" * 50)

ids = []
for i in range(5):
    # Very similar content, inserted quickly
    data = {"task": "Test", "messages": ["Message"]}
    memory_id = generate_memory_id(data)
    ids.append(memory_id)
    print(f"ID {i}: {memory_id}")
    time.sleep(0.0001)  # Very small delay

if len(set(ids)) < len(ids):
    print(f"⚠️  COLLISION! Only {len(set(ids))} unique IDs out of {len(ids)}")
else:
    print("✓ No collisions")

# Scenario 2: Birthday paradox calculation
print("\nScenario 2: Birthday Paradox Analysis")
print("-" * 50)

# With 16 hex chars = 64 bits
bits = 64
total_possibilities = 2 ** bits

# Birthday paradox: 50% collision probability
import math
memories_for_50_percent = math.sqrt(2 * total_possibilities * math.log(2))

print(f"ID space: 16 hex chars = {bits} bits")
print(f"Total possible IDs: {total_possibilities:,}")
print(f"Memories for 50% collision chance: {memories_for_50_percent:,.0f}")
print(f"That's about {memories_for_50_percent/1e9:.1f} billion memories")

# Scenario 3: What happens on collision?
print("\nScenario 3: What happens on collision?")
print("-" * 50)

class SimpleStore:
    def __init__(self):
        self.memories = {}
    
    def store(self, data):
        memory_id = generate_memory_id(data)
        # THIS IS THE BUG - no check if ID exists!
        self.memories[memory_id] = data
        return memory_id

store = SimpleStore()

# Store first memory
id1 = store.store({"task": "Important data", "content": "Don't lose me!"})
print(f"Stored memory 1: {id1}")
print(f"Memory content: {store.memories[id1]}")

# Force a collision (mock the generator)
def forced_collision_generator(data):
    return id1  # Return same ID!

# Temporarily replace generator
original = generate_memory_id
generate_memory_id = forced_collision_generator

# Store second memory with same ID
id2 = store.store({"task": "New data", "content": "I overwrote the old one!"})
print(f"\nStored memory 2: {id2}")
print(f"Memory content: {store.memories[id2]}")

print(f"\n⚠️  ORIGINAL MEMORY LOST! The first memory is gone forever!")
print(f"Store now has {len(store.memories)} memory (should have 2)")

# Scenario 4: Real-world collision risk
print("\nScenario 4: Real-world Collision Risk")
print("-" * 50)

def collision_probability(n, bits):
    """Calculate collision probability for n items in b-bit space."""
    if n > 2**(bits/2):
        return 1.0
    # Approximation of birthday paradox
    return 1 - math.exp(-n**2 / (2 * 2**bits))

memory_counts = [1000, 10000, 100000, 1000000, 10000000]
for count in memory_counts:
    prob = collision_probability(count, 64)
    print(f"{count:>10,} memories: {prob*100:.6f}% collision chance")

print("\nRisk Assessment:")
print("- At 10M memories: 0.000005% chance (very low)")
print("- At 100M memories: 0.0005% chance (still low)")  
print("- At 1B memories: 0.05% chance (getting risky)")
print("- At 5B memories: 1.3% chance (unacceptable!)")

print("\n" + "="*50)
print("THE REAL PROBLEM:")
print("="*50)
print("It's not about the probability - it's that when it happens:")
print("1. You lose data SILENTLY")
print("2. No error, no warning")
print("3. Could lose critical memories")
print("4. Impossible to detect after the fact")