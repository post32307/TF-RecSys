import sys

with open('run_sweep.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip_image_func = False
for i, line in enumerate(lines):
    if line.startswith("def save_table_images"):
        skip_image_func = True
    
    if skip_image_func:
        if line.startswith("def main():"):
            skip_image_func = False
        else:
            continue
            
    if 334 <= i <= 507:
        if line.strip() != "":
            line = "    " + line
    
    if "save_table_images(" in line:
        continue

    new_lines.append(line)

with open('run_sweep.py', 'w') as f:
    f.writelines(new_lines)

