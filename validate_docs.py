#!/usr/bin/env python3
"""
Simple validation script for PyNAS documentation.
"""

import os
import sys
from pathlib import Path

def validate_documentation():
    """Validate the documentation structure and files."""
    docs_dir = Path(__file__).parent / 'docs' / 'source'
    
    # Check if docs directory exists
    if not docs_dir.exists():
        print(f"❌ Documentation directory not found: {docs_dir}")
        return False
    
    print(f"✅ Documentation directory found: {docs_dir}")
    
    # Required files
    required_files = [
        'conf.py',
        'index.rst',
        'installation.rst',
        'quickstart.rst',
        'contributing.rst',
        'changelog.rst'
    ]
    
    # Check required files
    missing_files = []
    for file_name in required_files:
        file_path = docs_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    # Check directories
    required_dirs = ['api', 'tutorials', 'examples']
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = docs_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory")
            # List files in the directory
            for file_path in dir_path.glob('*.rst'):
                print(f"  - {file_path.name}")
        else:
            print(f"❌ {dir_name}/ directory - MISSING")
            missing_dirs.append(dir_name)
    
    # Validate conf.py
    conf_path = docs_dir / 'conf.py'
    if conf_path.exists():
        try:
            with open(conf_path, 'r') as f:
                conf_content = f.read()
            
            # Check for key configurations
            key_configs = [
                'project = ',
                'extensions = ',
                'html_theme = ',
                'autodoc_'
            ]
            
            print("\n📋 Checking conf.py configurations:")
            for config in key_configs:
                if config in conf_content:
                    print(f"✅ {config}found")
                else:
                    print(f"❌ {config}NOT found")
                    
        except Exception as e:
            print(f"❌ Error reading conf.py: {e}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"Missing files: {len(missing_files)}")
    print(f"Missing directories: {len(missing_dirs)}")
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
    
    if missing_dirs:
        print(f"Missing directories: {', '.join(missing_dirs)}")
    
    success = len(missing_files) == 0 and len(missing_dirs) == 0
    if success:
        print("🎉 All documentation files and directories are present!")
    else:
        print("⚠️  Some documentation files or directories are missing.")
    
    return success

def check_rst_syntax():
    """Basic RST syntax validation."""
    docs_dir = Path(__file__).parent / 'docs' / 'source'
    
    print(f"\n🔍 Checking RST syntax...")
    
    rst_files = list(docs_dir.glob('**/*.rst'))
    
    for rst_file in rst_files:
        try:
            with open(rst_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic checks
            lines = content.split('\n')
            issues = []
            
            # Check for common RST issues
            for i, line in enumerate(lines, 1):
                # Check for inconsistent title underlines
                if line and all(c in '=-~^' for c in line):
                    if i > 1:
                        title_line = lines[i-2]
                        if title_line and len(line) != len(title_line):
                            issues.append(f"Line {i}: Title underline length mismatch")
            
            if issues:
                print(f"⚠️  {rst_file.relative_to(docs_dir)}: {len(issues)} issues")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
            else:
                print(f"✅ {rst_file.relative_to(docs_dir)}")
                
        except Exception as e:
            print(f"❌ Error reading {rst_file.relative_to(docs_dir)}: {e}")

if __name__ == '__main__':
    print("🔍 PyNAS Documentation Validation")
    print("=" * 40)
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    # Validate structure
    structure_ok = validate_documentation()
    
    # Check RST syntax
    check_rst_syntax()
    
    # Final status
    if structure_ok:
        print(f"\n🎉 Documentation validation completed successfully!")
        print(f"📖 Ready to build with: sphinx-build docs/source docs/build/html")
    else:
        print(f"\n❌ Documentation validation failed!")
        sys.exit(1)
