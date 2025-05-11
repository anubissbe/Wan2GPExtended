#!/usr/bin/env python3
"""
Simple patch script for Wan2GP to add long video generation support
"""
import os
import sys
import shutil
import re
import subprocess

def install_dependencies():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy", "requests"])
        print("Required dependencies installed.")
        return True
    except:
        print("WARNING: Failed to install dependencies automatically.")
        print("Please run: pip install moviepy requests")
        return False

def patch_wgp():
    # Path to wgp.py
    wgp_file = os.path.join(os.getcwd(), "wgp.py")
    
    if not os.path.exists(wgp_file):
        print(f"ERROR: Could not find {wgp_file}")
        return False
    
    # Create a backup
    backup_file = f"{wgp_file}.bak"
    if not os.path.exists(backup_file):
        shutil.copy2(wgp_file, backup_file)
        print(f"Created backup of wgp.py at {backup_file}")
    
    try:
        # Read the file
        with open(wgp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add import
        if "from long_video_tab import create_long_video_tab" not in content:
            import_position = content.rfind("import") + content[content.rfind("import"):].find("\n") + 1
            import_line = "\n# Long video generation extension\nfrom long_video_tab import create_long_video_tab\n"
            content = content[:import_position] + import_line + content[import_position:]
        
        # Find tabs section
        tabs_match = re.search(r"with gr\.Tabs\(.*?\) as main_tabs:", content, re.DOTALL)
        if not tabs_match:
            print("Could not find the main tabs section in wgp.py")
            return False
        
        tabs_end = tabs_match.end()
        
        # Find a position to add our tab
        about_match = re.search(r"with gr\.Tab\(\"About\"\):", content[tabs_end:])
        if not about_match:
            print("Could not find a suitable position for the new tab")
            return False
        
        insert_pos = tabs_end + about_match.start()
        
        # Add our tab
        if "gr.Tab(\"Long Video Generation\"" not in content:
            tab_code = """
            with gr.Tab("Long Video Generation", id="long_video_gen"):
                long_video_generator = create_long_video_tab(wan_model, state)
"""
            content = content[:insert_pos] + tab_code + content[insert_pos:]
        
        # Write the modified content
        with open(wgp_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("Successfully patched wgp.py!")
        return True
    
    except Exception as e:
        print(f"ERROR: Failed to patch wgp.py: {e}")
        return False

def main():
    print("== Wan2GP Long Video Generation Patch ==")
    
    # Check if module files exist
    if not os.path.exists("long_video_generator.py") or not os.path.exists("long_video_tab.py"):
        print("ERROR: Required module files are missing. Make sure both files are in the current directory.")
        return
    
    # Install dependencies
    install_dependencies()
    
    # Patch wgp.py
    if patch_wgp():
        print("\nPatch completed successfully!")
        print("You can now run Wan2GP with: python wgp.py")
        print("The 'Long Video Generation' tab will be available in the interface.")
    else:
        print("\nPatching failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
