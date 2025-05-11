#!/usr/bin/env python3
"""
Wan2GP Long Video Generation Extension Patch

This script adds long video generation capabilities to Wan2GP by:
1. Creating the necessary module files
2. Patching the main wgp.py file
3. Installing any required dependencies

Usage:
    python patch_wan2gp_long_video.py

Author: Claude
"""

import os
import sys
import importlib.util
import subprocess
import re
import shutil
import traceback

# Directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define module contents
LONG_VIDEO_GENERATOR_MODULE = """import os
import time
import json
import random
import tempfile
import requests
import torch
import gc
import subprocess
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Constants
DEFAULT_SEGMENT_LENGTH = 15  # Default segment length in seconds
MAX_SEGMENT_LENGTH = 15  # Maximum segment length that Wan2GP can handle reliably
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API URL
OPENWEBUI_API_URL = "http://localhost:3000/api/chat"   # Default OpenWebUI API URL

class LongVideoGenerator:
    def __init__(
        self, 
        base_model=None,
        use_ollama=True, 
        ollama_model="mistral", 
        ollama_api_url=OLLAMA_API_URL,
        use_openwebui=True,
        openwebui_api_url=OPENWEBUI_API_URL,
        temp_dir=None
    ):
        """
        Initialize the long video generator.
        
        Args:
            base_model: The Wan2GP model instance to use for video generation
            use_ollama: Whether to use Ollama for prompt enhancement
            ollama_model: The name of the Ollama model to use
            ollama_api_url: The URL of the Ollama API
            use_openwebui: Whether to use OpenWebUI for detailed scene generation
            openwebui_api_url: The URL of the OpenWebUI API
            temp_dir: Directory to store temporary files
        """
        self.base_model = base_model
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_api_url = ollama_api_url
        self.use_openwebui = use_openwebui
        self.openwebui_api_url = openwebui_api_url
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print(f"LongVideoGenerator initialized. Temporary files will be stored in {self.temp_dir}")
    
    def enhance_prompt_with_ollama(self, prompt):
        """
        Enhance the input prompt using Ollama LLM.
        
        Args:
            prompt: The original prompt to enhance
        
        Returns:
            The enhanced prompt
        """
        if not self.use_ollama:
            return prompt
            
        try:
            # Prepare the request to Ollama
            system_prompt = """You are a creative assistant specialized in enhancing video generation prompts.
            Your task is to expand the given prompt with more details, making it more descriptive and suitable for video generation.
            Focus on visual elements, lighting, mood, camera movement, and scene composition.
            Keep your response ONLY to the enhanced prompt without any explanations or additional text."""
            
            payload = {
                "model": self.ollama_model,
                "prompt": f"Enhance this prompt for video generation: {prompt}",
                "system": system_prompt,
                "stream": False
            }
            
            # Send the request to Ollama
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            
            # Extract the enhanced prompt from the response
            enhanced_prompt = response.json().get("response", "").strip()
            
            print(f"Original prompt: {prompt}")
            print(f"Enhanced prompt: {enhanced_prompt}")
            
            return enhanced_prompt
        except Exception as e:
            print(f"Error enhancing prompt with Ollama: {e}")
            return prompt  # Return the original prompt if enhancement fails
    
    def generate_scene_description(self, main_prompt, segment_index, total_segments):
        """
        Generate a detailed scene description for a specific segment using OpenWebUI/Ollama.
        
        Args:
            main_prompt: The main video prompt
            segment_index: Index of the current segment
            total_segments: Total number of segments
        
        Returns:
            A detailed scene description for the segment
        """
        if not self.use_openwebui:
            # If OpenWebUI is not used, we'll just append segment information to the main prompt
            progress = segment_index / total_segments
            return f"{main_prompt} [Scene {segment_index+1}/{total_segments}, timepoint {progress:.2%} through the overall narrative]"
        
        try:
            # Create a system prompt that guides the generation of a specific scene
            # in a coherent sequence based on the main prompt
            system_prompt = f"""You are a creative director for a video sequence. 
            The main concept of the video is: "{main_prompt}"
            
            I need you to describe scene {segment_index+1} of {total_segments} in great detail.
            This scene should naturally flow from the previous scenes and maintain consistency.
            Your description should be highly visual and include details about:
            - What is happening in this specific scene
            - Visual elements, colors, and lighting
            - Camera movements and angles
            - Mood and atmosphere
            - Character positions and actions (if applicable)
            
            Keep the scene description under 200 words and focus only on this specific segment.
            The scene description should follow logically if this is part {segment_index+1} of a {total_segments}-part story.
            
            Only return the scene description without any other text or explanation."""
            
            # Prepare and send the request
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create detailed scene {segment_index+1}/{total_segments} description for: {main_prompt}"}
                ]
            }
            
            response = requests.post(self.openwebui_api_url, json=payload)
            response.raise_for_status()
            
            # Extract the scene description from the response
            scene_description = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not scene_description:
                # Fallback if we can't parse the response
                progress = segment_index / total_segments
                scene_description = f"{main_prompt} [Scene {segment_index+1}/{total_segments}, timepoint {progress:.2%} through the overall narrative]"
            
            print(f"Generated scene {segment_index+1}/{total_segments}: {scene_description[:100]}...")
            
            return scene_description
        except Exception as e:
            print(f"Error generating scene description: {e}")
            # Fallback to a simple scene description
            progress = segment_index / total_segments
            return f"{main_prompt} [Scene {segment_index+1}/{total_segments}, timepoint {progress:.2%} through the overall narrative]"
    
    def generate_video_segment(self, scene_prompt, segment_index, **generation_params):
        """
        Generate a single video segment using the Wan2GP model.
        
        Args:
            scene_prompt: The prompt for this specific scene
            segment_index: Index of the current segment
            generation_params: Additional parameters for the video generation
        
        Returns:
            Path to the generated video file
        """
        if self.base_model is None:
            raise ValueError("Base model is not initialized")
        
        # Set up parameters for this segment
        output_filename = os.path.join(self.temp_dir, f"segment_{segment_index:03d}.mp4")
        
        # Prepare generation parameters
        segment_params = generation_params.copy()
        segment_params["prompt"] = scene_prompt
        
        try:
            # Call the Wan2GP model to generate the video
            # The actual implementation depends on how the Wan2GP model is structured
            t2v = "image2video" not in self.base_model._model_file_name and "Fun_InP" not in self.base_model._model_file_name
            
            # Calculate segment frame count - ensure it's a valid number for the model
            frame_count = (generation_params.get("segment_length", DEFAULT_SEGMENT_LENGTH) * 16) // 4 * 4 + 1
            
            # Extract params from generation_params or use defaults
            height = generation_params.get("height", 512)
            width = generation_params.get("width", 512)
            num_inference_steps = generation_params.get("num_inference_steps", 30)
            guidance_scale = generation_params.get("guidance_scale", 9.0)
            negative_prompt = generation_params.get("negative_prompt", "")
            seed = generation_params.get("seed", None)
            
            # The actual implementation will be different based on whether this is t2v or i2v
            if t2v:
                samples = self.base_model.generate(
                    scene_prompt,
                    frame_num=frame_count,
                    height=height,
                    width=width,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    VAE_tile_size=128,  # Reasonable default for most GPUs
                )
            else:
                # For image-to-video, we'd need an image, but in our workflow we're just doing text-to-video
                # This branch probably won't be used but included for completeness
                image_start = generation_params.get("image_start", None)
                samples = self.base_model.generate(
                    scene_prompt,
                    image_start,
                    None,  # image_end would be None
                    frame_num=frame_count,
                    height=height,
                    width=width,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    VAE_tile_size=128,  # Reasonable default for most GPUs
                )
            
            # Process the generated video
            from wan.utils.utils import cache_video
            cache_video(
                tensor=samples[None], 
                save_file=output_filename, 
                fps=16,  # Standard fps for Wan2GP
                nrow=1, 
                normalize=True, 
                value_range=(-1, 1)
            )
            
            if os.path.exists(output_filename):
                print(f"Generated segment {segment_index+1}: {output_filename}")
                return output_filename
            else:
                raise FileNotFoundError(f"Generated video file not found: {output_filename}")
            
        except Exception as e:
            print(f"Error generating video segment {segment_index+1}: {e}")
            return None
    
    def concatenate_videos(self, video_paths, output_path):
        """
        Concatenate multiple video segments into a single video.
        
        Args:
            video_paths: List of paths to video segments
            output_path: Path to save the concatenated video
        
        Returns:
            Path to the concatenated video
        """
        if not video_paths:
            raise ValueError("No video segments to concatenate")
            
        try:
            # Load video clips
            clips = []
            for path in video_paths:
                if os.path.exists(path):
                    clip = VideoFileClip(path)
                    clips.append(clip)
                else:
                    print(f"Warning: Video file not found: {path}")
            
            if not clips:
                raise ValueError("No valid video clips to concatenate")
                
            # Concatenate clips
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # Write the final video
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                threads=4,
                preset='medium'
            )
            
            # Close clips to free memory
            for clip in clips:
                clip.close()
            final_clip.close()
            
            print(f"Successfully concatenated {len(clips)} video segments to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error concatenating videos: {e}")
            return None
    
    def generate_long_video(
        self, 
        prompt, 
        duration_minutes, 
        output_path, 
        segment_length=DEFAULT_SEGMENT_LENGTH,
        enhance_prompt=True,
        generate_detailed_scenes=True,
        clean_temp_files=True,
        **generation_params
    ):
        """
        Generate a long video by creating multiple segments and concatenating them.
        
        Args:
            prompt: The main prompt for the video
            duration_minutes: Desired duration in minutes
            output_path: Path to save the final video
            segment_length: Length of each segment in seconds
            enhance_prompt: Whether to enhance the main prompt
            generate_detailed_scenes: Whether to generate detailed scene descriptions
            clean_temp_files: Whether to clean up temporary files after generation
            generation_params: Additional parameters for video generation
        
        Returns:
            Path to the generated video, or None if generation failed
        """
        start_time = time.time()
        
        # Calculate number of segments needed
        total_duration_seconds = duration_minutes * 60
        num_segments = max(1, int(total_duration_seconds / segment_length))
        
        print(f"Generating {duration_minutes} minute video ({total_duration_seconds} seconds)")
        print(f"Creating {num_segments} segments of {segment_length} seconds each")
        
        # Enhance the main prompt if enabled
        if enhance_prompt and self.use_ollama:
            enhanced_prompt = self.enhance_prompt_with_ollama(prompt)
        else:
            enhanced_prompt = prompt
        
        # Generate each segment
        video_paths = []
        
        for i in range(num_segments):
            print(f"\\nGenerating segment {i+1}/{num_segments}...")
            
            # Generate detailed scene description if enabled
            if generate_detailed_scenes:
                scene_prompt = self.generate_scene_description(enhanced_prompt, i, num_segments)
            else:
                scene_prompt = enhanced_prompt
            
            # Add segment length to generation parameters
            generation_params["segment_length"] = segment_length
            
            # Generate the video segment
            segment_path = self.generate_video_segment(
                scene_prompt=scene_prompt,
                segment_index=i,
                **generation_params
            )
            
            if segment_path:
                video_paths.append(segment_path)
            else:
                print(f"Failed to generate segment {i+1}/{num_segments}")
        
        # Concatenate the segments if we have any
        final_path = None
        if video_paths:
            print(f"\\nConcatenating {len(video_paths)} video segments...")
            final_path = self.concatenate_videos(video_paths, output_path)
        
        # Clean up temporary files if requested
        if clean_temp_files:
            for path in video_paths:
                try:
                    os.remove(path)
                except:
                    pass
        
        end_time = time.time()
        print(f"\\nVideo generation completed in {end_time - start_time:.2f} seconds")
        return final_path
"""

LONG_VIDEO_TAB_MODULE = """import os
import time
import random
import gradio as gr
import torch
from datetime import datetime
from long_video_generator import LongVideoGenerator

# Constants
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OPENWEBUI_API_URL = "http://localhost:3000/api/chat"
DEFAULT_OLLAMA_MODEL = "mistral"

def create_long_video_tab(wan_model, state):
    """
    Create a new tab dedicated to long video generation
    
    Args:
        wan_model: The Wan2GP model instance
        state: The state dictionary used by Wan2GP
        
    Returns:
        The long video generator instance
    """
    # Initialize the long video generator
    long_video_generator = LongVideoGenerator(
        base_model=wan_model,
        use_ollama=True,
        ollama_model=DEFAULT_OLLAMA_MODEL,
        ollama_api_url=DEFAULT_OLLAMA_API_URL,
        use_openwebui=True, 
        openwebui_api_url=DEFAULT_OPENWEBUI_API_URL
    )
    
    # Function to generate a long video
    def generate_long_video(
        prompt, 
        duration_minutes, 
        segment_length, 
        width,
        height,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        seed,
        enhance_prompt,
        generate_detailed_scenes,
        clean_temp_files,
        use_ollama,
        use_openwebui,
        ollama_model,
        ollama_api_url,
        openwebui_api_url
    ):
        try:
            # Update generator configuration if needed
            long_video_generator.use_ollama = use_ollama
            long_video_generator.use_openwebui = use_openwebui
            long_video_generator.ollama_model = ollama_model
            long_video_generator.ollama_api_url = ollama_api_url
            long_video_generator.openwebui_api_url = openwebui_api_url
            
            # Create output directory if it doesn't exist
            save_path = state.get("gen", {}).get("save_path", "outputs")
            output_dir = os.path.join(save_path, "long_videos")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename for the video
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
            output_filename = f"long_video_{timestamp}_seed{seed}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Set the seed
            if seed <= 0:
                seed = random.randint(0, 999999999)
            
            # Call the long video generator
            status = f"Generating {duration_minutes} minute video..."
            yield status, None
            
            # Prepare generation parameters
            generation_params = {
                "width": int(width),
                "height": int(height),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "negative_prompt": negative_prompt,
                "seed": int(seed)
            }
            
            # Generate the video
            video_path = long_video_generator.generate_long_video(
                prompt=prompt,
                duration_minutes=float(duration_minutes),
                output_path=output_path,
                segment_length=int(segment_length),
                enhance_prompt=enhance_prompt,
                generate_detailed_scenes=generate_detailed_scenes,
                clean_temp_files=clean_temp_files,
                **generation_params
            )
            
            if video_path and os.path.exists(video_path):
                status = f"Video generated successfully: {output_filename}"
                yield status, video_path
            else:
                status = "Failed to generate video"
                yield status, None
        
        except Exception as e:
            import traceback
            error_msg = f"Error generating long video: {str(e)}\\n\\n{traceback.format_exc()}"
            print(error_msg)
            yield error_msg, None
    
    # Create the UI for the long video tab
    with gr.Column():
        gr.Markdown("# Long Video Generation")
        gr.Markdown("Generate longer videos by stitching together multiple segments with coherent narratives.")
        
        with gr.Row():
            with gr.Column(scale=5):
                # Main video prompt
                prompt = gr.TextArea(
                    label="Video Prompt",
                    placeholder="Describe the long video you want to generate...",
                    lines=5,
                    value="A time-lapse of a beautiful coastal city from dawn to dusk, showing the changing colors of the sky, the movement of clouds, and the gradual illumination of city lights. The camera slowly pans across the cityscape, revealing the harmony between nature and urban development."
                )
                
                with gr.Row():
                    # Duration and segment length controls
                    duration_minutes = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                        label="Duration (minutes)"
                    )
                    
                    segment_length = gr.Slider(
                        minimum=5,
                        maximum=15,
                        value=15,
                        step=1,
                        label="Segment Length (seconds)"
                    )
                
                with gr.Accordion("Video Settings", open=True):
                    with gr.Row():
                        width = gr.Dropdown(
                            choices=["320", "384", "448", "512", "576", "640", "704", "768"],
                            value="512",
                            label="Width"
                        )
                        
                        height = gr.Dropdown(
                            choices=["320", "384", "448", "512", "576", "640", "704", "768"],
                            value="512",
                            label="Height"
                        )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=30,
                            step=1,
                            label="Inference Steps"
                        )
                        
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=9.0,
                            step=0.1,
                            label="Guidance Scale"
                        )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Features you want to avoid in the video...",
                        value=""
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1
                    )
                
                with gr.Accordion("Enhancement Settings", open=False):
                    enhance_prompt = gr.Checkbox(
                        label="Enhance Prompt with Ollama",
                        value=True
                    )
                    
                    generate_detailed_scenes = gr.Checkbox(
                        label="Generate Detailed Scene Descriptions",
                        value=True
                    )
                    
                    clean_temp_files = gr.Checkbox(
                        label="Clean Temporary Files",
                        value=True
                    )
                
                with gr.Accordion("API Settings", open=False):
                    use_ollama = gr.Checkbox(
                        label="Use Ollama for Prompt Enhancement",
                        value=True
                    )
                    
                    use_openwebui = gr.Checkbox(
                        label="Use OpenWebUI for Scene Generation",
                        value=True
                    )
                    
                    ollama_model = gr.Textbox(
                        label="Ollama Model",
                        value=DEFAULT_OLLAMA_MODEL
                    )
                    
                    ollama_api_url = gr.Textbox(
                        label="Ollama API URL",
                        value=DEFAULT_OLLAMA_API_URL
                    )
                    
                    openwebui_api_url = gr.Textbox(
                        label="OpenWebUI API URL",
                        value=DEFAULT_OPENWEBUI_API_URL
                    )
            
            with gr.Column(scale=5):
                # Status display
                status_output = gr.Markdown("Ready to generate long video")
                
                # Generated video display
                output_video = gr.Video(label="Generated Long Video")
                
                # Generate button
                generate_btn = gr.Button("Generate Long Video", variant="primary")
        
        # Hook up the event handlers
        generate_btn.click(
            fn=generate_long_video,
            inputs=[
                prompt, 
                duration_minutes, 
                segment_length,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                negative_prompt,
                seed,
                enhance_prompt,
                generate_detailed_scenes,
                clean_temp_files,
                use_ollama,
                use_openwebui,
                ollama_model,
                ollama_api_url,
                openwebui_api_url
            ],
            outputs=[
                status_output,
                output_video
            ]
        )
    
    return long_video_generator
"""

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = ['moviepy', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Required packages installed successfully")
        except Exception as e:
            print(f"Error installing packages: {e}")
            print("Please install the following packages manually:")
            for package in missing_packages:
                print(f"  pip install {package}")
            return False
    
    return True

def create_module_files():
    """Create the module files needed for the long video generator"""
    # Create long_video_generator.py
    with open(os.path.join(SCRIPT_DIR, "long_video_generator.py"), "w", encoding="utf-8") as f:
        f.write(LONG_VIDEO_GENERATOR_MODULE)
    
    # Create long_video_tab.py
    with open(os.path.join(SCRIPT_DIR, "long_video_tab.py"), "w", encoding="utf-8") as f:
        f.write(LONG_VIDEO_TAB_MODULE)
    
    print("Created module files:")
    print(f"  {os.path.join(SCRIPT_DIR, 'long_video_generator.py')}")
    print(f"  {os.path.join(SCRIPT_DIR, 'long_video_tab.py')}")
    
    return True

def patch_main_file():
    """Patch the wgp.py file to add the long video generation tab"""
    wgp_file = os.path.join(SCRIPT_DIR, "wgp.py")
    
    if not os.path.exists(wgp_file):
        print(f"Error: Could not find {wgp_file}")
        return False
    
    # Create a backup of the original file
    backup_file = os.path.join(SCRIPT_DIR, "wgp.py.bak")
    if not os.path.exists(backup_file):
        shutil.copy2(wgp_file, backup_file)
        print(f"Created backup of wgp.py at {backup_file}")
    
    try:
        # Read the original file
        with open(wgp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add import statement
        import_statement = "from long_video_tab import create_long_video_tab"
        if "from long_video_tab import create_long_video_tab" not in content:
            import_position = content.rfind("import") + content[content.rfind("import"):].find("\n") + 1
            if import_position > 0:
                modified_content = content[:import_position] + "\n# Long video generation extension\n" + import_statement + "\n" + content[import_position:]
                content = modified_content
        
        # Find the main tabs section
        tabs_pattern = r"with gr\.Tabs\(.*?\) as main_tabs:"
        tabs_match = re.search(tabs_pattern, content, re.DOTALL)
        
        if not tabs_match:
            print("Could not find the main tabs section in wgp.py")
            return False
        
        tabs_pos = tabs_match.end()
        
        # Find where to insert our new tab
        about_tab_pattern = r"with gr\.Tab\(\"About\"\):"
        about_tab_match = re.search(about_tab_pattern, content[tabs_pos:])
        
        if not about_tab_match:
            print("Could not find a suitable position to insert the new tab")
            return False
        
        insert_pos = tabs_pos + about_tab_match.start()
        
        # Create the long video tab code block
        long_video_tab_code = """
            with gr.Tab("Long Video Generation", id="long_video_gen"):
                long_video_generator = create_long_video_tab(wan_model, state)
"""
        
        # Check if our tab is already there
        if "gr.Tab(\"Long Video Generation\"" in content:
            print("Long video generation tab is already present in wgp.py")
            return True
        
        # Insert our tab code before the About tab
        modified_content = content[:insert_pos] + long_video_tab_code + content[insert_pos:]
        
        # Write the modified content back to the file
        with open(wgp_file, "w", encoding="utf-8") as f:
            f.write(modified_content)
        
        print("Successfully patched wgp.py to add the long video generation tab")
        return True
        
    except Exception as e:
        print(f"Error patching wgp.py: {e}")
        traceback.print_exc()
        return False

def print_manual_instructions():
    """Print manual integration instructions in case automatic patching fails"""
    print("\nIf automatic patching failed, here's how to manually integrate the long video generation feature:")
    print("\n1. Make sure the following files are in your Wan2GP directory:")
    print("   - long_video_generator.py")
    print("   - long_video_tab.py")
    print("\n2. Open wgp.py in a text editor")
    print("\n3. Add the following import at the top of the file, after the other imports:")
    print("   ```python")
    print("   # Long video generation extension")
    print("   from long_video_tab import create_long_video_tab")
    print("   ```")
    print("\n4. Find the tabs section with 'with gr.Tabs(selected=\"video_gen\", ) as main_tabs:'")
    print("\n5. Before the 'with gr.Tab(\"About\"):' line, add the following code:")
    print("   ```python")
    print("   with gr.Tab(\"Long Video Generation\", id=\"long_video_gen\"):")
    print("       long_video_generator = create_long_video_tab(wan_model, state)")
    print("   ```")
    print("\n6. Save the file and run Wan2GP")
    print("\nYou should now see a 'Long Video Generation' tab in the interface")

def main():
    print("="*80)
    print("Wan2GP Long Video Generation Extension Installer")
    print("="*80)
    print("\nThis script will add long video generation capabilities to Wan2GP.")
    print("It will create necessary files and patch the main wgp.py file.")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\nFailed to install required dependencies.")
        print("Please install them manually and run this script again.")
        return
    
    # Create module files
    print("\nCreating module files...")
    if not create_module_files():
        print("\nFailed to create module files.")
        return
    
    # Patch main file
    print("\nPatching wgp.py...")
    if not patch_main_file():
        print("\nFailed to automatically patch wgp.py.")
        print_manual_instructions()
        return
    
    print("\n"+"="*80)
    print("Installation completed successfully!")
    print("="*80)
    print("\nYou can now run Wan2GP and use the 'Long Video Generation' tab.")
    print("The tab will appear after the existing tabs in the interface.")
    
    # Provide usage information
    print("\nUsage:")
    print("1. Start Wan2GP as usual with 'python wgp.py'")
    print("2. Go to the 'Long Video Generation' tab")
    print("3. Enter your prompt, set the desired duration and segment length")
    print("4. Click 'Generate Long Video'")
    
    # Note about Ollama and OpenWebUI
    print("\nNote: For optimal results, install Ollama and Open-WebUI for enhanced prompts")
    print("and better scene descriptions. If they're not available, the generator will")
    print("still work but with simpler scene transitions.")
    print("\nOllama: https://ollama.com/")
    print("Open-WebUI: https://github.com/open-webui/open-webui")

if __name__ == "__main__":
    main()
