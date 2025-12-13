with open(r'C:\Users\Admin\Documents\GitHub\Mod\src\qwen3_vl\attention\dynamic_sparse_attention.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the function in the first class (DynamicSparseAttention)
function_start = None
for i, line in enumerate(lines):
    if 'def apply_dynamic_sparsity' in line and i > 50:  # Skip the first occurrence
        function_start = i
        break

if function_start is not None:
    # Find the end of the function by looking for the next function or class
    function_end = function_start
    for i in range(function_start + 1, len(lines)):
        if lines[i].strip().startswith('def ') or lines[i].strip().startswith('class ') or (lines[i].strip() and not lines[i].startswith('    ') and not lines[i].startswith('\t')):
            function_end = i
            break
        function_end = i

    print("Function found from line", function_start, "to", function_end)
    for j in range(function_start, function_end + 1):
        print(f"{j}: {lines[j].rstrip()}")