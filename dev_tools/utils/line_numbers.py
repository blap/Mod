with open('test_system_level_optimizations.py', 'r') as f:
    lines = f.readlines()
    for i in range(205, 220):  # Show lines around 212
        if i < len(lines):
            print(f'{i+1}: {lines[i].rstrip()}')