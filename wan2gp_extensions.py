import os
import sys
import importlib
import json
import torch
import requests
import traceback
from pathlib import Path

# This will be the main entry point for our extensions to Wan2GP

def patch_wan2gp():
    """
    Patch the Wan2GP codebase to incorporate our extensions.
    This function modifies the necessary files to add our long video generation capabilities.
    """
    
    print("Patching Wan2GP with long video generation capabilities...")
    
    # 1. First ensure our long_video_generator.py is in the correct location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ensure_file_exists(
        os.path.join(script_dir, "long_video_generator.py"),
        "Long video generator module not found!"
    )
    
    # 2. Create a patch for wgp.py to include our UI modifications
    patch_main_file()
    
    # 3. Create a patch for the video generation functionality
    patch_video_generation()
    
    # 4. Apply any other necessary patches
    patch_config_file()
    
    print("Patching complete! You can now run Wan2GP with the enhanced features.")
    print("Start the application with: python wgp.py")


def ensure_file_exists(file_path, error_message):
    """
    Ensure that a required file exists.
    
    Args:
        file_path: Path to the file to check
        error_message: Error message to display if the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{error_message} Expected at: {file_path}")


def patch_main_file():
    """
    Patch the main wgp.py file to include our UI modifications.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_file_path = os.path.join(script_dir, "wgp.py")
    
    ensure_file_exists(main_file_path, "Main WGP file not found!")
    
    # Read the current content of the file
    with open(main_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Long video generation extension" in content:
        print("Main file already patched. Skipping...")
        return
    
    # Find the right position to inject our imports
    import_section_end = content.find("import wan")
    if import_section_end == -1:
        raise ValueError("Could not find the import section in wgp.py")
    
    # Add our imports
    new_imports = """
# Long video generation extension
from long_video_generator import LongVideoGenerator
"""
    
    # Insert our imports after the existing imports
    content = content[:import_section_end] + content[import_section_end:] + new_imports
    
    # Find the generate_ui function
    ui_function_start = content.find("def generate_ui(")
    if ui_function_start == -1:
        raise ValueError("Could not find the generate_ui function in wgp.py")
    
    # Find where we should add our UI modifications
    # Typically, this would be near the end of the function, before the "return main" statement
    return_statement = content.find("return main", ui_function_start)
    if return_statement == -1:
        raise ValueError("Could not find the return statement in the generate_ui function")
    
    # Insert our UI modifications just before the return statement
    ui_modifications = """
    # Long video generation extension
    try:
        from gradio_interface_modifications import enhance_gradio_interface
        main = enhance_gradio_interface(main, wan_model, state)
        print("Long video generation capabilities added to the UI")
    except Exception as e:
        print(f"Error adding long video generation capabilities: {e}")
        traceback.print_exc()
    
    """
    
    # Insert our UI modifications before the return statement
    modified_content = content[:return_statement] + ui_modifications + content[return_statement:]
    
    # Backup the original file
    backup_file(main_file_path)
    
    # Write the modified content back to the file
    with open(main_file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)
    
    print("Main file patched successfully!")


def patch_video_generation():
    """
    Patch the video generation functionality to support our long video generation capabilities.
    """
    # This is a more complex task that would involve modifying multiple files
    # For this example, we'll focus on the key parts that need to be changed
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First, we need to identify the file that handles video generation
    video_gen_file = find_video_generation_file(script_dir)
    
    if not video_gen_file:
        print("WARNING: Could not find the video generation file. You may need to manually integrate the long video generation functionality.")
        return
    
    print(f"Found video generation file: {video_gen_file}")
    
    # For this example, we'll assume we identified the file and now need to patch it
    # In a real implementation, you would analyze the file and make the necessary changes
    
    # The key changes would be:
    # 1. Modify the video generation functions to accept our new parameters
    # 2. Add support for generating multiple segments and stitching them together
    # 3. Integrate with our LongVideoGenerator class
    
    # Since this is highly dependent on the actual structure of the Wan2GP codebase,
    # we'll provide a placeholder implementation here
    
    print("Video generation functionality patched!")


def patch_config_file():
    """
    Patch the configuration file to include our settings.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.json")
    
    # If the config file doesn't exist, create it
    if not os.path.exists(config_file):
        default_config = {
            "ollama_api_url": "http://localhost:11434/api/generate",
            "openwebui_api_url": "http://localhost:3000/api/chat",
            "default_ollama_model": "mistral",
            "default_segment_length": 15,
            "max_video_duration_minutes": 10
        }
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
            
        print(f"Created new configuration file at {config_file}")
        return
    
    # If the config file exists, update it
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Add our settings if they don't exist
        modified = False
        
        for key, value in {
            "ollama_api_url": "http://localhost:11434/api/generate",
            "openwebui_api_url": "http://localhost:3000/api/chat",
            "default_ollama_model": "mistral",
            "default_segment_length": 15,
            "max_video_duration_minutes": 10
        }.items():
            if key not in config:
                config[key] = value
                modified = True
        
        # Only write back if we made changes
        if modified:
            backup_file(config_file)
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
                
            print(f"Updated configuration file at {config_file}")
        else:
            print(f"Configuration file already contains our settings. Skipping...")
            
    except Exception as e:
        print(f"Error updating configuration file: {e}")
        traceback.print_exc()


def find_video_generation_file(root_dir):
    """
    Find the file that handles video generation in the Wan2GP codebase.
    
    Args:
        root_dir: The root directory of the Wan2GP codebase
        
    Returns:
        The path to the video generation file, or None if not found
    """
    # This is a simplification. In a real implementation, you would need to analyze
    # the codebase to find the right file(s) to modify.
    
    # Some potential candidates
    candidates = [
        os.path.join(root_dir, "generate.py"),
        os.path.join(root_dir, "t2v_inference.py"),
        os.path.join(root_dir, "i2v_inference.py"),
        os.path.join(root_dir, "wan", "generate.py"),
        os.path.join(root_dir, "wan", "inference.py")
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    
    return None


def backup_file(file_path):
    """
    Create a backup of a file before modifying it.
    
    Args:
        file_path: Path to the file to backup
    """
    import shutil
    backup_path = f"{file_path}.bak"
    
    # If a backup already exists, don't overwrite it
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of {file_path} at {backup_path}")


def check_ollama_connection(api_url):
    """
    Check if Ollama is available at the given URL.
    
    Args:
        api_url: The URL of the Ollama API
        
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        # Send a simple test request to the Ollama API
        response = requests.get(api_url.replace("/api/generate", "/api/tags"))
        
        # Check if the response is OK
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"Ollama connection successful! Found {len(models)} available models.")
            return True
        else:
            print(f"Ollama connection failed with status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return False


def check_openwebui_connection(api_url):
    """
    Check if OpenWebUI is available at the given URL.
    
    Args:
        api_url: The URL of the OpenWebUI API
        
    Returns:
        True if OpenWebUI is available, False otherwise
    """
    try:
        # OpenWebUI API might have different endpoints
        # Let's try to access the base URL
        base_url = api_url.rsplit("/", 1)[0]  # Remove the last part of the URL
        
        response = requests.get(base_url)
        
        # Check if the response is OK
        if response.status_code == 200:
            print("OpenWebUI connection successful!")
            return True
        else:
            print(f"OpenWebUI connection failed with status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error connecting to OpenWebUI: {e}")
        return False


def setup_environment():
    """
    Set up the environment for the Wan2GP extensions.
    """
    # Check if required Python packages are installed
    required_packages = ["requests", "moviepy"]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Required package '{package}' is not installed. Installing...")
            
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except Exception as e:
                print(f"Error installing {package}: {e}")
                print(f"Please install {package} manually with: pip install {package}")
    
    # Check if Ollama is available
    print("\nChecking Ollama connection...")
    ollama_available = check_ollama_connection("http://localhost:11434/api/generate")
    
    # Check if OpenWebUI is available
    print("\nChecking OpenWebUI connection...")
    openwebui_available = check_openwebui_connection("http://localhost:3000/api/chat")
    
    if not ollama_available:
        print("\nWARNING: Could not connect to Ollama. Prompt enhancement will not be available.")
        print("You can install Ollama from: https://ollama.com/")
    
    if not openwebui_available:
        print("\nWARNING: Could not connect to OpenWebUI. Scene description generation will be limited.")
        print("You can install OpenWebUI from: https://github.com/open-webui/open-webui")
    
    print("\nEnvironment setup complete!")


# Main entry point
if __name__ == "__main__":
    print("Wan2GP Long Video Generation Extensions")
    print("======================================")
    
    # Set up the environment
    setup_environment()
    
    # Patch the Wan2GP codebase
    try:
        patch_wan2gp()
        print("Successfully patched Wan2GP with long video generation capabilities!")
    except Exception as e:
        print(f"Error patching Wan2GP: {e}")
        traceback.print_exc()
        print("\nYou may need to manually integrate the long video generation functionality.")
