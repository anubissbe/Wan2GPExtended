import os
import time
import json
import random
import tempfile
import requests
import torch
import gc
import subprocess
import re
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Constants
DEFAULT_SEGMENT_LENGTH = 15  # Default segment length in seconds
MAX_SEGMENT_LENGTH = 15  # Maximum segment length that Wan2GP can handle reliably
OLLAMA_API_URL = "http://192.168.1.25:11434/api/generate"  # Default Ollama API URL
OPENWEBUI_API_URL = "http://192.168.1.25:3000/api/generate"   # Updated to use same server as Ollama

class LongVideoGenerator:
    def __init__(
        self, 
        base_model=None,
        use_ollama=True, 
        ollama_model="mistral", 
        ollama_api_url=OLLAMA_API_URL,
        use_openwebui=False,
        openwebui_api_url=OPENWEBUI_API_URL,
        temp_dir=None
    ):
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
        if self.base_model is None:
            print("Warning: base_model is None. Video generation will not be possible until a model is loaded.")
    
    def parse_segmented_prompt(self, prompt):
        """Parse a prompt that contains explicit SEGMENT markers and return a list of segment-specific prompts."""
        # Look for SEGMENT pattern
        segments = re.findall(r'SEGMENT\s+\d+.*?(?=SEGMENT\s+\d+|$)', prompt, re.DOTALL)
        
        if not segments:
            # If no segments found, return the whole prompt
            return [prompt]
        
        # Clean up each segment
        clean_segments = []
        for segment in segments:
            # Add the main prompt to each segment
            if prompt.split('SEGMENT')[0].strip():
                segment = prompt.split('SEGMENT')[0].strip() + "\n\n" + segment
            clean_segments.append(segment.strip())
        
        return clean_segments

    def enhance_prompt_with_ollama(self, prompt, progress_callback=None):
        if not self.use_ollama:
            return prompt
            
        try:
            if progress_callback:
                progress_callback(0, 1, "Enhancing prompt with Ollama")
                
            # Prepare the request to Ollama
            system_prompt = """You are a creative assistant specialized in enhancing video generation prompts.
            Your task is to expand the given prompt with more details, making it more descriptive and suitable for video generation.
            Focus on visual elements, lighting, mood, camera movement, and scene composition.
            Include clear progression stages to ensure the video has distinct visual changes over time.
            Divide your response into clear SEGMENT sections if the prompt implies a narrative with progression.
            Keep your response ONLY to the enhanced prompt without any explanations or additional text."""
            
            payload = {
                "model": self.ollama_model,
                "prompt": f"Enhance this prompt for video generation. Make it extremely detailed with strong visual progression over time: {prompt}",
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
            
            if progress_callback:
                progress_callback(1, 1, "Prompt enhancement complete")
                
            return enhanced_prompt
        except Exception as e:
            print(f"Error enhancing prompt with Ollama: {e}")
            if progress_callback:
                progress_callback(1, 1, f"Prompt enhancement failed: {e}")
            return prompt  # Return the original prompt if enhancement fails
    
    def generate_scene_description(self, main_prompt, segment_index, total_segments, progress_callback=None):
        if not self.use_openwebui:
            # Create more variation between segments without external LLM
            progress = segment_index / total_segments
            
            # Create more descriptive segment-specific prompts
            if segment_index == 0:
                time_descriptor = "at the beginning"
                prefix = "Starting scene: "
                additional_context = "Show the opening/establishing shot with clear lighting and setting details."
            elif segment_index == total_segments - 1:
                time_descriptor = "at the end"
                prefix = "Final scene: "
                additional_context = "Show the conclusion with clear visual differences from the beginning."
            else:
                time_descriptor = f"approximately {progress:.0%} of the way through"
                prefix = f"Middle scene {segment_index+1}: "
                additional_context = f"This segment shows clear progression from previous scenes with distinct visual changes."
            
            # Create more detailed scene-specific prompt
            return f"{prefix}{main_prompt} [Scene {segment_index+1}/{total_segments}, {time_descriptor} of the narrative. {additional_context}]"
        
        try:
            if progress_callback:
                progress_callback(0, 1, f"Generating scene description {segment_index+1}/{total_segments}")
                
            # Create a system prompt that guides the generation of a specific scene
            # in a coherent sequence based on the main prompt
            system_prompt = f"""You are a creative director for a video sequence. 
            The main concept of the video is: "{main_prompt}"
            
            I need you to describe scene {segment_index+1} of {total_segments} in great detail.
            This scene should naturally flow from the previous scenes and maintain consistency,
            but should show CLEAR VISUAL PROGRESSION and DIFFERENCES from other scenes.
            
            Your description should be extremely specific about:
            - What is happening in this specific scene (with clear differences from other scenes)
            - Visual elements, colors, and lighting (should evolve throughout the sequence)
            - Camera movements and angles
            - Mood and atmosphere (should have distinct progression)
            - Time of day or temporal progression indicators
            
            Keep the scene description under 250 words and focus only on this specific segment.
            The scene description should follow logically as part {segment_index+1} of a {total_segments}-part story.
            
            Only return the scene description without any other text or explanation."""
            
            # Try different API formats that might be supported by the server
            try_formats = [
                # Standard format for chat completions
                {
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Create detailed scene {segment_index+1}/{total_segments} description for: {main_prompt}"}
                    ]
                },
                # Ollama-style format for /api/generate
                {
                    "model": self.ollama_model,
                    "prompt": f"System: {system_prompt}\n\nUser: Create detailed scene {segment_index+1}/{total_segments} description for: {main_prompt}",
                    "system": system_prompt,
                    "stream": False
                }
            ]
            
            response = None
            error_messages = []
            
            for payload in try_formats:
                try:
                    print(f"Trying API format with URL: {self.openwebui_api_url}")
                    response = requests.post(self.openwebui_api_url, json=payload)
                    response.raise_for_status()
                    break  # If successful, exit the loop
                except Exception as e:
                    error_messages.append(str(e))
            
            if response is None or response.status_code != 200:
                raise Exception(f"All API formats failed: {', '.join(error_messages)}")
                
            # Try to extract the scene description from the response
            try:
                # Format 1: OpenAI-style API response
                scene_description = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except:
                try:
                    # Format 2: Ollama-style API response
                    scene_description = response.json().get("response", "").strip()
                except:
                    scene_description = ""
            
            if not scene_description:
                # Fallback if we can't parse the response
                progress = segment_index / total_segments
                scene_description = f"{main_prompt} [Scene {segment_index+1}/{total_segments}, timepoint {progress:.2%} through the overall narrative]"
            
            print(f"Generated scene {segment_index+1}/{total_segments}: {scene_description[:100]}...")
            
            if progress_callback:
                progress_callback(1, 1, "Scene description generation complete")
                
            return scene_description
        except Exception as e:
            print(f"Error generating scene description: {e}")
            if progress_callback:
                progress_callback(1, 1, f"Scene description generation failed: {e}")
                
            # Fallback to a simple scene description
            progress = segment_index / total_segments
            return f"{main_prompt} [Scene {segment_index+1}/{total_segments}, timepoint {progress:.2%} through the overall narrative]"
    
    def generate_video_segment(self, scene_prompt, segment_index, prev_segment_frames=None, progress_callback=None, **generation_params):
        if self.base_model is None:
            print("Debug: Base model is None. This could be because:")
            print("1. The model wasn't properly passed to the generator")
            print("2. The model was unloaded after being passed")
            print("3. The model initialization failed elsewhere")
            print("Returning None instead of generating video segment")
            return None, None  # Return (None, None) for path and frames
        
        # Set up parameters for this segment
        output_filename = os.path.join(self.temp_dir, f"segment_{segment_index:03d}.mp4")
        
        # Prepare generation parameters
        segment_params = generation_params.copy()
        segment_params["prompt"] = scene_prompt
        
        if progress_callback:
            progress_callback(0, 100, f"Starting generation of segment {segment_index+1}")
        
        try:
            # Import mmgp and offload from Wan2GP
            try:
                from __main__ import offload
            except ImportError:
                try:
                    from mmgp import offload
                except ImportError:
                    print("Warning: Could not import offload from mmgp. Some functionality may be limited.")
                    # Create a dummy shared_state
                    if not hasattr(offload, 'shared_state'):
                        offload.shared_state = {}
            
            # Set up attention mode in shared_state
            if not "_attention" in offload.shared_state:
                print("Setting default attention mode 'sdpa' in shared_state")
                offload.shared_state["_attention"] = "sdpa"  # Use SDPA as default
            
            # Determine if this is a text-to-video or image-to-video model
            t2v = "image2video" not in self.base_model._model_file_name and "Fun_InP" not in self.base_model._model_file_name
            
            # Add _interrupt attribute if it doesn't exist
            if not hasattr(self.base_model, '_interrupt'):
                print(f"Adding _interrupt attribute to {type(self.base_model).__name__}")
                self.base_model._interrupt = False
            
            # Get the underlying model
            if hasattr(self.base_model, 'model'):
                model = self.base_model.model
                
                # Add Tea Cache attributes if they don't exist
                if not hasattr(model, 'enable_teacache'):
                    print(f"Adding enable_teacache attribute to {type(model).__name__}")
                    model.enable_teacache = False
                    model.teacache_multiplier = 0
                    model.rel_l1_thresh = 0
                    model.teacache_start_step = 0
                    model.coefficients = None
                    
                # Add other required attributes from wgp.py
                if not hasattr(model, 'previous_residual'):
                    model.previous_residual = None
                
                # Make sure model has proper attention configuration
                if not hasattr(model, 'attention_mode'):
                    model.attention_mode = offload.shared_state.get("_attention", "sdpa")
            
            # Calculate segment frame count - ensure it's a valid number for the model
            # Use a fixed 30 FPS instead of 24 FPS for smoother motion
            fps = 30
            frame_count = (generation_params.get("segment_length", DEFAULT_SEGMENT_LENGTH) * fps) // 4 * 4 + 1
            
            # Extract params from generation_params or use defaults
            height = generation_params.get("height", 512)
            width = generation_params.get("width", 512)
            num_inference_steps = generation_params.get("num_inference_steps", 50)  # Default increased from 30 to 50
            guidance_scale = generation_params.get("guidance_scale", 8.0)  # Reduced to 8.0 for smoother transitions
            negative_prompt = generation_params.get("negative_prompt", "")
            seed = generation_params.get("seed", None)
            
            # If we're not the first segment and we have previous segment frames, prepare them
            conditioning_frames = None
            if prev_segment_frames is not None and segment_index > 0:
                print(f"Using {len(prev_segment_frames)} frames from previous segment as conditioning")
                conditioning_frames = prev_segment_frames
            
            # Create a callback for the model generation that updates our progress
            def model_callback(step_idx, latent, force_refresh, read_state=False, override_num_inference_steps=-1):
                if step_idx >= 0 and progress_callback:
                    # Check if generation was aborted
                    if hasattr(self.base_model, '_interrupt') and self.base_model._interrupt:
                        return None
                        
                    # Map step progress to 10%-90% of our overall segment progress
                    # Convert to integer to avoid Pydantic validation errors
                    progress_pct = int(10 + (step_idx / num_inference_steps) * 80)
                    progress_callback(progress_pct, 100, f"Generating segment {segment_index+1}: step {step_idx+1}/{num_inference_steps}")
                return None
            
            if progress_callback:
                progress_callback(10, 100, f"Running model inference for segment {segment_index+1}")
            
            # Additional parameters that might be needed - improved for better quality
            additional_params = {
                "VAE_tile_size": 256,  # Increased from 128 for better quality
                "joint_pass": True,  # For speed
                "slg_layers": None,  # No skip layer guidance by default
                "enable_RIFLEx": True  # Enable RIFLEx for better long video coherence
            }
            
            # The actual implementation will be different based on whether this is t2v or i2v
            if t2v:
                # If we have conditioning frames, prepare them
                if conditioning_frames is not None:
                    # Create input_frames parameter for conditioning
                    input_frames = conditioning_frames
                    
                    # Generate with conditioning frames
                    samples = self.base_model.generate(
                        scene_prompt,
                        input_frames=input_frames,  # Use prev segment frames for conditioning
                        frame_num=frame_count,
                        size=(width, height),
                        sampling_steps=num_inference_steps,
                        guide_scale=guidance_scale,
                        n_prompt=negative_prompt,
                        seed=seed,
                        callback=model_callback,
                        **additional_params
                    )
                else:
                    # Normal generation without conditioning
                    samples = self.base_model.generate(
                        scene_prompt,
                        frame_num=frame_count,
                        size=(width, height),
                        sampling_steps=num_inference_steps,
                        guide_scale=guidance_scale,
                        n_prompt=negative_prompt,
                        seed=seed,
                        callback=model_callback,
                        **additional_params
                    )
            else:
                # For image-to-video, we'd need an image, but in our workflow we're just doing text-to-video
                # This branch probably won't be used but included for completeness
                image_start = generation_params.get("image_start", None)
                
                if conditioning_frames is not None:
                    # TODO: For image2video models, we'd need to implement conditioning differently
                    # This is placeholder logic
                    additional_params["prev_frames"] = conditioning_frames
                
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
                    callback=model_callback,
                    **additional_params
                )
                
            # Check if the generation was aborted
            if hasattr(self.base_model, '_interrupt') and self.base_model._interrupt:
                print(f"Generation of segment {segment_index+1} was aborted")
                if progress_callback:
                    progress_callback(100, 100, f"Segment {segment_index+1} aborted")
                return None, None
                
            if progress_callback:
                progress_callback(90, 100, f"Post-processing segment {segment_index+1}")
            
            # Before saving to video, extract the last frames to use for next segment conditioning
            # We'll take the last 4 frames (assuming we generated at least that many)
            last_frames_count = 4  # Number of frames to extract for conditioning
            
            # Clone samples to CPU to avoid CUDA memory issues
            if torch.is_tensor(samples):
                samples_cpu = samples.detach().cpu()
                # Extract the last frames - tensor is [C, F, H, W]
                last_frames = None
                if samples_cpu.shape[1] >= last_frames_count:
                    last_frames = samples_cpu[:, -last_frames_count:].clone()
            
            # Process the generated video - using 30 FPS instead of 24 for smoother motion
            from wan.utils.utils import cache_video
            cache_video(
                tensor=samples[None], 
                save_file=output_filename, 
                fps=30,  # Increased from 24 to 30 for smoother motion
                nrow=1, 
                normalize=True, 
                value_range=(-1, 1)
            )
            
            if os.path.exists(output_filename):
                print(f"Generated segment {segment_index+1}: {output_filename}")
                if progress_callback:
                    progress_callback(100, 100, f"Segment {segment_index+1} complete")
                return output_filename, last_frames
            else:
                raise FileNotFoundError(f"Generated video file not found: {output_filename}")
            
        except Exception as e:
            print(f"Error generating video segment {segment_index+1}: {e}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(100, 100, f"Segment {segment_index+1} failed: {str(e)}")
            return None, None
    
    def concatenate_videos(self, video_paths, output_path, progress_callback=None):
        if not video_paths:
            raise ValueError("No video segments to concatenate")
            
        try:
            if progress_callback:
                progress_callback(0, 100, f"Concatenating {len(video_paths)} video segments")
                
            # Load video clips
            clips = []
            for i, path in enumerate(video_paths):
                if progress_callback:
                    # Update progress for each clip loaded (0%-50%)
                    progress_pct = int((i / len(video_paths)) * 50)  # Convert to integer
                    progress_callback(progress_pct, 100, f"Loading video segment {i+1}/{len(video_paths)}")
                    
                if os.path.exists(path):
                    clip = VideoFileClip(path)
                    clips.append(clip)
                else:
                    print(f"Warning: Video file not found: {path}")
            
            if not clips:
                raise ValueError("No valid video clips to concatenate")
                
            if progress_callback:
                progress_callback(50, 100, "Creating final video with transitions")
                
            # Add crossfade transitions between clips (if there are multiple clips)
            if len(clips) > 1:
                crossfade_duration = 1.5  # Increased from 0.5 to 1.5 seconds for smoother transitions
                final_clips = []
                
                for i in range(len(clips) - 1):
                    # Current clip without the last crossfade_duration seconds
                    current_clip_duration = clips[i].duration
                    if current_clip_duration <= crossfade_duration:
                        final_clips.append(clips[i])
                        continue
                        
                    current_clip = clips[i].subclip(0, current_clip_duration - crossfade_duration)
                    final_clips.append(current_clip)
                    
                    # Add the crossfade segment
                    fade_out_clip = clips[i].subclip(current_clip_duration - crossfade_duration, current_clip_duration)
                    fade_in_clip = clips[i+1].subclip(0, crossfade_duration)
                    
                    # Create a crossfade by overlaying with dissolve
                    crossfade_clip = fade_out_clip.crossfadein(crossfade_duration)
                    final_clips.append(crossfade_clip)
                
                # Add the last clip
                final_clips.append(clips[-1])
                
                # Concatenate all clips
                final_clip = concatenate_videoclips(final_clips, method="compose")
            else:
                # Single clip, no transitions needed
                final_clip = clips[0]
            
            if progress_callback:
                progress_callback(60, 100, "Writing final video file")
                
            # Write the final video with higher bitrate for better quality
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                threads=4,
                preset='slow',  # Changed from 'medium' to 'slow' for better quality
                bitrate="8000k"  # Higher bitrate for better quality
            )
            
            if progress_callback:
                progress_callback(95, 100, "Cleaning up resources")
                
            # Close clips to free memory
            for clip in clips:
                clip.close()
            
            if hasattr(final_clip, 'close'):
                final_clip.close()
            
            print(f"Successfully concatenated {len(clips)} video segments to: {output_path}")
            
            if progress_callback:
                progress_callback(100, 100, "Video concatenation complete")
                
            return output_path
            
        except Exception as e:
            print(f"Error concatenating videos: {e}")
            if progress_callback:
                progress_callback(100, 100, f"Concatenation failed: {str(e)}")
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
        progress_callback=None,  # New parameter for progress reporting
        **generation_params
    ):
        start_time = time.time()
        
        # Add check for model availability at the beginning
        if self.base_model is None:
            print("Error: Cannot generate video - model is not available")
            return None
        
        # Calculate number of segments needed
        total_duration_seconds = duration_minutes * 60
        num_segments = max(1, int(total_duration_seconds / segment_length))
        
        print(f"Generating {duration_minutes} minute video ({total_duration_seconds} seconds)")
        print(f"Creating {num_segments} segments of {segment_length} seconds each")
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(0, num_segments + 1, f"Starting generation of {num_segments} segments")
        
        # Enhance the main prompt if enabled
        if enhance_prompt and self.use_ollama:
            if progress_callback:
                progress_callback(0, num_segments + 1, "Enhancing prompt with Ollama")
            enhanced_prompt = self.enhance_prompt_with_ollama(prompt, progress_callback)
        else:
            enhanced_prompt = prompt
            
        # Check if the prompt contains explicit segments
        parsed_segments = self.parse_segmented_prompt(enhanced_prompt)
        num_defined_segments = len(parsed_segments)

        # Adjust number of segments if explicit segments are provided
        if num_defined_segments > 1:
            print(f"Found {num_defined_segments} explicit segments in the prompt")
            if num_defined_segments != num_segments:
                print(f"Warning: {num_defined_segments} segments defined in prompt, but {num_segments} segments calculated from duration. Using the segments from the prompt.")
                num_segments = num_defined_segments
        
        # Generate each segment
        video_paths = []
        prev_segment_frames = None  # Store frames from previous segment for conditioning
        
        for i in range(num_segments):
            print(f"\nGenerating segment {i+1}/{num_segments}...")
            
            # Check if generation was aborted
            if hasattr(self.base_model, '_interrupt') and self.base_model._interrupt:
                print("Video generation aborted")
                break
                
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, num_segments + 1, f"Generating segment {i+1}/{num_segments}")
            
            # Use the parsed segment if available, otherwise generate a description
            if num_defined_segments > 1 and i < num_defined_segments:
                scene_prompt = parsed_segments[i]
                if progress_callback:
                    progress_callback(i, num_segments + 1, f"Using pre-defined segment {i+1}")
            elif generate_detailed_scenes:
                if progress_callback:
                    sub_progress_msg = f"Creating scene description for segment {i+1}/{num_segments}"
                    progress_callback(i, num_segments + 1, sub_progress_msg)
                scene_prompt = self.generate_scene_description(
                    enhanced_prompt, i, num_segments, progress_callback
                )
            else:
                scene_prompt = enhanced_prompt
            
            # Add segment length to generation parameters
            generation_params["segment_length"] = segment_length
            
            # Generate the video segment - now passing previous segment frames
            if progress_callback:
                sub_progress_msg = f"Generating video for segment {i+1}/{num_segments}"
                progress_callback(i, num_segments + 1, sub_progress_msg)
                
            segment_path, last_frames = self.generate_video_segment(
                scene_prompt=scene_prompt,
                segment_index=i,
                prev_segment_frames=prev_segment_frames,  # Pass frames from previous segment
                progress_callback=progress_callback,
                **generation_params
            )
            
            # Check if generation was aborted during segment generation
            if hasattr(self.base_model, '_interrupt') and self.base_model._interrupt:
                print("Video generation aborted during segment generation")
                break
            
            if segment_path:
                video_paths.append(segment_path)
                # Save the last frames for next segment
                prev_segment_frames = last_frames
            else:
                print(f"Failed to generate segment {i+1}/{num_segments}")
        
        # If no segments were generated or generation was aborted, return None
        if not video_paths or (hasattr(self.base_model, '_interrupt') and self.base_model._interrupt):
            if progress_callback:
                progress_callback(num_segments + 1, num_segments + 1, "Video generation aborted")
            print("No video segments generated or generation was aborted")
            return None
        
        # Update progress for concatenation step
        if progress_callback:
            progress_callback(num_segments, num_segments + 1, "Concatenating video segments")
        
        # Concatenate the segments
        print(f"\nConcatenating {len(video_paths)} video segments...")
        final_path = self.concatenate_videos(
            video_paths, output_path, progress_callback
        )
        
        # Clean up temporary files if requested
        if clean_temp_files:
            for path in video_paths:
                try:
                    os.remove(path)
                except:
                    pass
        
        # Final progress update
        if progress_callback:
            progress_callback(num_segments + 1, num_segments + 1, "Video generation complete")
        
        end_time = time.time()
        print(f"\nVideo generation completed in {end_time - start_time:.2f} seconds")
        return final_path
