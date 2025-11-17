
# Test the usage example
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Capture output
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

try:
    exec(open('gloria_usage_example.py').read())
    output = buffer.getvalue()
finally:
    sys.stdout = old_stdout

# Print summary of output
lines = output.split('\n')
print("Usage example executed successfully!")
print("\nKey output lines:")
for line in lines:
    if '✓' in line or 'dimensions' in line or 'IOT shape' in line or 'Total' in line:
        print(line)
