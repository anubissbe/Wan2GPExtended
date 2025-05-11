import os
import time
import random
import gradio as gr
import torch
from datetime import datetime
from long_video_generator import LongVideoGenerator

# Constants
DEFAULT_OLLAMA_API_URL = "http://192.168.1.25:11434/api/generate"
DEFAULT_OPENWEBUI_API_URL = "http://192.168.1.25:3000/api/generate"
DEFAULT_OLLAMA_MODEL = "mistral:7b"

def create_long_video_tab(wan_model, state):
    # We'll use this as a fallback only, but prefer to load models directly
    fallback_model = wan_model
    
    # Get model selection function and load_models function from parent scope
    # These will be accessible since they're in the global scope of wgp.py
    from __main__ import generate_dropdown_model_list, load_models, get_model_type, transformer_quantization, get_model_filename
    
    # Initialize the long video generator with None model - we'll set it at generation time
    long_video_generator = LongVideoGenerator(
        base_model=None,
        use_ollama=True,
        ollama_model=DEFAULT_OLLAMA_MODEL,
        ollama_api_url=DEFAULT_OLLAMA_API_URL,
        use_openwebui=False,
        openwebui_api_url=DEFAULT_OPENWEBUI_API_URL
    )
    
    # Function to abort generation
    def abort_generation():
        if long_video_generator.base_model is not None:
            long_video_generator.base_model._interrupt = True
            return gr.update(value="Aborting generation..."), None, gr.update(visible=True), gr.update(visible=False)
        return gr.update(), None, gr.update(visible=True), gr.update(visible=False)
    
    # Function to generate a long video
    def generate_long_video(
        model_choice,
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
        openwebui_api_url,
        progress=gr.Progress()
    ):
        try:
            # First yield to make abort button visible and generate button invisible
            yield f"Loading model {model_choice}...", None, gr.update(visible=True), gr.update(visible=False)
            
            # Get the model filename based on the model choice
            model_filename = get_model_filename(model_choice, transformer_quantization)
            
            # Load the model (this is using the function from wgp.py)
            current_model, _, _ = load_models(model_filename)
            
            if current_model is None:
                error_msg = f"Error: Failed to load model {model_choice}. Please check console for details."
                print(error_msg)
                yield error_msg, None, gr.update(visible=True), gr.update(visible=False)
                return
                
            # Update the model in the generator
            long_video_generator.base_model = current_model
            # Ensure _interrupt is False at the start
            long_video_generator.base_model._interrupt = False
            
            print(f"Model loaded successfully: {model_choice}")
            print(f"Model type: {type(current_model)}")
            
            # Update other generator configuration
            long_video_generator.use_ollama = use_ollama
            long_video_generator.use_openwebui = use_openwebui
            long_video_generator.ollama_model = ollama_model
            long_video_generator.ollama_api_url = ollama_api_url
            long_video_generator.openwebui_api_url = openwebui_api_url
            
            # Create output directory if it doesn't exist
            if hasattr(state, 'get'):
                # For dictionary-like state
                save_path = state.get("gen", {}).get("save_path", "outputs")
            else:
                # For Gradio State object
                save_path = "outputs"  # Default fallback
                try:
                    # Try to access state value if it's a Gradio State
                    if hasattr(state, 'value') and isinstance(state.value, dict):
                        gen_dict = state.value.get("gen", {})
                        if isinstance(gen_dict, dict):
                            save_path = gen_dict.get("save_path", "outputs")
                except Exception as e:
                    print(f"Warning: Could not access save_path from state: {e}")

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
            status = f"Generating {duration_minutes} minute video using {model_choice} model..."
            yield status, None, gr.update(visible=True), gr.update(visible=False)
            
            # Check Ollama connection if using Ollama
            if use_ollama and enhance_prompt:
                try:
                    import requests
                    test_url = ollama_api_url.replace("/api/generate", "")
                    if test_url.endswith("/api"):
                        test_url = test_url[:-4]
                    response = requests.get(f"{test_url}/api/tags", timeout=5)
                    if response.status_code != 200:
                        status = f"Warning: Could not connect to Ollama at {ollama_api_url}. Will use original prompt."
                        print(status)
                        yield status, None, gr.update(visible=True), gr.update(visible=False)
                except Exception as e:
                    status = f"Warning: Could not connect to Ollama: {str(e)}. Will use original prompt."
                    print(status)
                    yield status, None, gr.update(visible=True), gr.update(visible=False)
            
            # Prepare generation parameters
            generation_params = {
                "width": int(width),
                "height": int(height),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "negative_prompt": negative_prompt,
                "seed": int(seed)
            }
            
            # Set a progress updater function that communicates with Gradio's progress bar
            def progress_callback(current, total, message=""):
                # Check if generation was aborted
                if hasattr(long_video_generator.base_model, '_interrupt') and long_video_generator.base_model._interrupt:
                    return
                # Ensure values are integers to avoid Pydantic validation errors
                progress((int(current), int(total)), desc=message)
            
            # Add progress callback to generation parameters
            generation_params["progress_callback"] = progress_callback
            
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
            
            # Check if generation was aborted
            if hasattr(long_video_generator.base_model, '_interrupt') and long_video_generator.base_model._interrupt:
                status = "Video generation was aborted by user."
                yield status, None, gr.update(visible=True), gr.update(visible=False)
                return
            
            if video_path and os.path.exists(video_path):
                status = f"Video generated successfully using {model_choice} model: {output_filename}"
                yield status, video_path, gr.update(visible=True), gr.update(visible=False)
            else:
                status = f"Failed to generate video using {model_choice} model. Check console for details."
                yield status, None, gr.update(visible=True), gr.update(visible=False)
        
        except Exception as e:
            import traceback
            error_msg = f"Error generating long video: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            yield error_msg, None, gr.update(visible=True), gr.update(visible=False)
    
    # Create the UI for the long video tab
    with gr.Column():
        gr.Markdown("# Long Video Generation")
        gr.Markdown("Generate longer videos by stitching together multiple segments with coherent narratives.")
        
        # Add the model selection dropdown at the top
        with gr.Row():
            gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
            model_choice = generate_dropdown_model_list()
            gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
        
        with gr.Row():
            with gr.Column(scale=5):
                # Main video prompt
                prompt = gr.TextArea(
                    label="Video Prompt (Use SEGMENT 1, SEGMENT 2, etc. to define explicit segments)",
                    placeholder="Describe the long video you want to generate...",
                    lines=5,
                    value="A day in the life of a playful orange tabby kitten.\n\nSEGMENT 1\nDawn: The kitten wakes up in a cozy sunlit living room, stretching on a window sill as golden morning light streams in. The kitten's fur glows in the warm morning light, and its eyes are bright and curious.\n\nSEGMENT 2\nMorning: As time progresses, the kitten becomes increasingly playful, chasing toys across wooden floors, pouncing on a feather toy, and playfully batting at dangling strings. The camera follows the kitten's movements with smooth panning shots, capturing its boundless energy and graceful leaps.\n\nSEGMENT 3\nAfternoon: The room gradually darkens as afternoon turns to evening, with the lighting shifting from bright daylight to warm, orange sunset hues. The kitten plays with a ball of yarn, rolling and tumbling across a carpet.\n\nSEGMENT 4\nEvening: The exhausted kitten yawns widely, finds a comfortable spot on a soft blue cushion, and curls up to sleep as moonlight creates gentle blue highlights on its fur. The video concludes with the peacefully sleeping kitten, its chest rising and falling gently in the dim, moonlit room."
                )
                
                # Negative prompt  
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Features you want to avoid in the video...",
                    value="Low quality, blurry, distorted, pixelated, artificial, unrealistic cat anatomy, human features on cats, multiple cats, dogs, humans, text overlays, watermarks, cartoon style, anime style, unnatural colors, flickering, rapid scene changes, jerky camera movements, unnatural lighting, glitchy effects, oversaturated colors, poor composition"
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
                        value=10,  # Changed from 15 to 10
                        step=1,
                        label="Segment Length (seconds)"
                    )
                
                with gr.Accordion("Video Settings", open=True):
                    with gr.Row():
                        width = gr.Dropdown(
                            choices=["320", "384", "448", "512", "576", "640", "704", "768", "832", "960", "1024", "1280"],
                            value="768",  # Increased from 512 to 768
                            label="Width"
                        )
                        
                        height = gr.Dropdown(
                            choices=["320", "384", "448", "512", "576", "640", "704", "768", "832", "960", "1024", "1280"],
                            value="512",
                            label="Height"
                        )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,  # Increased from 30 to 50
                            step=1,
                            label="Inference Steps"
                        )
                        
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=10.0,  # Increased from 9.0 to 10.0
                            step=0.1,
                            label="Guidance Scale"
                        )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1
                    )
                
                with gr.Accordion("Enhancement Settings", open=True):
                    enhance_prompt = gr.Checkbox(
                        label="Enhance Prompt with Ollama (Creates more detailed prompt)",
                        value=True
                    )
                    
                    generate_detailed_scenes = gr.Checkbox(
                        label="Generate Unique Scene Descriptions for Each Segment (Improves coherence)",
                        value=True,
                        info="Turn this off if you want to use your own SEGMENT markers exactly as written"
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
                        value=False
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
                status_output = gr.Markdown("Ready to generate long video. Select a model and click Generate.")
                
                # Generated video display
                output_video = gr.Video(label="Generated Long Video")
                
                # Generate and Abort buttons
                with gr.Row():
                    generate_btn = gr.Button("Generate Long Video", variant="primary")
                    abort_btn = gr.Button("Abort Generation", variant="stop")
        
        # Hook up the event handlers
        generate_btn.click(
            fn=generate_long_video,
            inputs=[
                model_choice,
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
                output_video,
                generate_btn,
                abort_btn
            ]
        )
        
        # Hook up the abort button
        abort_btn.click(
            fn=abort_generation,
            inputs=[],
            outputs=[status_output, output_video, generate_btn, abort_btn]
        )
    
    return long_video_generator
