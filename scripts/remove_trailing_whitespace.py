import os
import glob

def remove_trailing_whitespace(filepath):
    try:
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into lines and process
        lines = content.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]

        # Join with Unix-style newlines
        cleaned_content = '\n'.join(cleaned_lines)
        if cleaned_content:  # Add final newline if file is not empty
            cleaned_content += '\n'

        # Write back
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cleaned_content)

def main():
    # Get all Python files in src and tests directories
    python_files = []
    for root in ['src', 'tests']:
        for filepath in glob.glob(f'{root}/**/*.py', recursive=True):
            python_files.append(filepath)

    for filepath in python_files:
        print(f'Cleaning {filepath}...')
        remove_trailing_whitespace(filepath)

if __name__ == '__main__':
    main()
