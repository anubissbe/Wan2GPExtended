import os
import time
import argparse
import json
import torch
import gradio as gr
import traceback
import gc
import random
from pathlib import Path

# Import our custom long video generator
from long_video_generator import LongVideoGenerator

# Configuration for Ollama and OpenWebUI integration
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OPENWEBUI_API_URL = "http://localhost:3000/api/chat"
DEFAULT_OLLAMA_MODEL = "mistral"  # Change this to your preferred model

# Function to add the long video generation functionality to the Gradio UI
def enhance_gradio_interface(demo_blocks, wan_model, state_obj):
    """
    Enhance the Gradio interface with long video generation capabilities.
    
    Args:
        demo_blocks: The existing gr.Blocks instance from Wan2GP
        wan_model: The Wan2GP model instance
        state_obj: The state object used in the original interface
    
    Returns:
        The enhanced gr.Blocks instance
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
    
    # Find the main tab where we'll add our long video generation tab
    main_tabs = None
    for component in demo_blocks.blocks:
        if isinstance(component, gr.Tabs):
            main_tabs = component
            break
    
    if main_tabs is None:
        print("Could not find main tabs in the interface. Long video functionality will not be added.")
        return demo_blocks
    
    # Add a new tab for long video generation
    with main_tabs:
        with gr.Tab("Long Video Generation", id="long_video_generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Add UI elements for long video generation
                    long_video_prompt = gr.TextArea(
                        label="Video Prompt",
                        placeholder="Describe the long video you want to generate...",
                        lines=5
                    )
                    
                    with gr.Row():
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
                    
                    with gr.Row():
                        enhance_prompt_checkbox = gr.Checkbox(
                            label="Enhance Prompt with Ollama",
                            value=True
                        )
                        
                        detailed_scenes_checkbox = gr.Checkbox(
                            label="Generate Detailed Scenes",
                            value=True
                        )
                        
                        clean_temp_checkbox = gr.Checkbox(
                            label="Clean Temporary Files",
                            value=True
                        )
                    
                    with gr.Row():
                        ollama_model_dropdown = gr.Dropdown(
                            label="Ollama Model",
                            choices=["mistral", "llama3", "mistral-openorca", "wizard-vicuna"],
                            value=DEFAULT_OLLAMA_MODEL
                        )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            ollama_api_url = gr.Textbox(
                                label="Ollama API URL",
                                value=DEFAULT_OLLAMA_API_URL
                            )
                            
                            openwebui_api_url = gr.Textbox(
                                label="OpenWebUI API URL",
                                value=DEFAULT_OPENWEBUI_API_URL
                            )
                        
                        with gr.Row():
                            # Reuse parameters from the main interface
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
                        
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed (-1 for random)",
                                value=-1
                            )
                
                with gr.Column(scale=2):
                    # Output video display
                    output_video = gr.Video(label="Generated Long Video")
                    
                    # Status and progress display
                    status_output = gr.Markdown("Ready to generate long video")
                    
                    # Generation button
                    generate_btn = gr.Button("Generate Long Video", variant="primary")
                    
                    # Stop generation button
                    stop_btn = gr.Button("Stop Generation", variant="stop")
                    
                    # Download button
                    download_btn = gr.Button("Download Video", visible=False)
            
            # Function to update the long video generator configuration
            def update_generator_config(
                ollama_model, 
                ollama_api_url, 
                openwebui_api_url
            ):
                nonlocal long_video_generator
                
                long_video_generator = LongVideoGenerator(
                    base_model=wan_model,
                    use_ollama=True,
                    ollama_model=ollama_model,
                    ollama_api_url=ollama_api_url,
                    use_openwebui=True,
                    openwebui_api_url=openwebui_api_url
                )
                
                return "Configuration updated successfully"
            
            # Function to handle long video generation
            def generate_long_video(
                prompt, 
                duration_mins, 
                segment_len, 
                enhance_prompt, 
                detailed_scenes,
                clean_temp,
                ollama_model, 
                ollama_api_url, 
                openwebui_api_url,
                width, 
                height, 
                num_steps, 
                guidance, 
                seed_val
            ):
                try:
                    # Update generator config if needed
                    if (long_video_generator.ollama_model != ollama_model or
                        long_video_generator.ollama_api_url != ollama_api_url or
                        long_video_generator.openwebui_api_url != openwebui_api_url):
                        
                        update_generator_config(
                            ollama_model, 
                            ollama_api_url, 
                            openwebui_api_url
                        )
                    
                    # Create output directory if it doesn't exist
                    output_dir = os.path.join(os.getcwd(), "outputs", "long_videos")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate a unique filename for the video
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    output_filename = f"long_video_{timestamp}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Set the seed
                    if seed_val <= 0:
                        seed_val = random.randint(0, 2147483647)
                    
                    # Call the long video generator
                    yield "Starting long video generation...", None, gr.update(visible=False)
                    
                    # Prepare generation parameters
                    generation_params = {
                        "width": int(width),
                        "height": int(height),
                        "num_inference_steps": int(num_steps),
                        "guidance_scale": float(guidance),
                        "seed": int(seed_val)
                    }
                    
                    # Generate the video
                    video_path = long_video_generator.generate_long_video(
                        prompt=prompt,
                        duration_minutes=float(duration_mins),
                        output_path=output_path,
                        segment_length=int(segment_len),
                        enhance_prompt=enhance_prompt,
                        generate_detailed_scenes=detailed_scenes,
                        clean_temp_files=clean_temp,
                        **generation_params
                    )
                    
                    if video_path and os.path.exists(video_path):
                        yield f"Video generated successfully: {output_filename}", video_path, gr.update(visible=True)
                    else:
                        yield "Failed to generate video", None, gr.update(visible=False)
                
                except Exception as e:
                    error_msg = f"Error generating long video: {str(e)}\n\n{traceback.format_exc()}"
                    yield error_msg, None, gr.update(visible=False)
            
            # Function to stop generation
            def stop_generation():
                # Implement the stop functionality
                # This would typically set a flag in the generator to stop
                return "Generation stopped"
                
            # Event handler for the generate button
            generate_btn.click(
                fn=generate_long_video,
                inputs=[
                    long_video_prompt, 
                    duration_minutes, 
                    segment_length, 
                    enhance_prompt_checkbox,
                    detailed_scenes_checkbox,
                    clean_temp_checkbox,
                    ollama_model_dropdown,
                    ollama_api_url,
                    openwebui_api_url,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    seed
                ],
                outputs=[
                    status_output,
                    output_video,
                    download_btn
                ]
            )
            
            # Event handler for the stop button
            stop_btn.click(
                fn=stop_generation,
                inputs=[],
                outputs=[status_output]
            )
            
            # Event handler for the download button
            download_btn.click(
                fn=lambda x: x,
                inputs=[output_video],
                outputs=[output_video]
            )
    
    return demo_blocks


# This function will be called from the main wgp.py to add our modifications
def enhance_main_script():
    """
    Function to patch the main script to include our long video generation capabilities.
    This is a simple demonstration of how to modify the code to include our new functionality.
    """
    try:
        # Import the main file - we'll need to integrate this modification in the actual code
        import wgp
        
        # Save a reference to the original generate_ui function
        original_generate_ui = wgp.generate_ui
        
        # Create a new function that wraps the original one and adds our enhancements
        def enhanced_generate_ui(*args, **kwargs):
            demo = original_generate_ui(*args, **kwargs)
            
            # Add our long video generation functionality
            enhanced_demo = enhance_gradio_interface(demo, kwargs.get("wan_model"), kwargs.get("state"))
            
            return enhanced_demo
        
        # Replace the original function with our enhanced version
        wgp.generate_ui = enhanced_generate_ui
        
        print("Successfully enhanced Wan2GP with long video generation capabilities")
        return True
        
    except Exception as e:
        print(f"Failed to enhance Wan2GP: {e}")
        return False


# For testing purposes
if __name__ == "__main__":
    print("This module is intended to be imported by the main Wan2GP script.")
    print("Run 'python wgp.py' to start the enhanced interface.")
