#!/usr/bin/env python3
"""TRELLIS.2 Image-to-3D Generator for Sakura DOK"""
import os
import sys
import traceback
from pathlib import Path

artifact_dir = os.environ.get('SAKURA_ARTIFACT_DIR', '/opt/artifact')
os.makedirs(artifact_dir, exist_ok=True)
log_path = os.path.join(artifact_dir, 'run.log')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, 'w')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

def main():
    print("=" * 60)
    print("TRELLIS.2 Image-to-3D Generator for Sakura DOK")
    print("=" * 60)
    print("")
    
    # Check environment
    print("Checking environment...")
    image_url = os.environ.get('IMAGE_URL')
    if not image_url:
        print("ERROR: IMAGE_URL environment variable not set")
        return False
    print(f"IMAGE_URL: {image_url}")
    print("")
    
    # Import dependencies
    print("Importing dependencies...")
    try:
        import requests
        from PIL import Image
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("ERROR: CUDA not available")
            return False
    except Exception as e:
        print(f"ERROR: Failed to import dependencies: {e}")
        traceback.print_exc()
        return False
    print("")
    
    # Environment setup
    print("Setting up environment...")
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("✓ Environment configured")
    print("")
    
    # Download image
    print("Downloading input image...")
    try:
        headers = {'User-Agent': 'TRELLIS2-DOK/1.0'}
        response = requests.get(image_url, timeout=60, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"ERROR: Failed to download image: {e}")
        traceback.print_exc()
        return False
    
    input_path = '/tmp/input.png'
    with open(input_path, 'wb') as f:
        f.write(response.content)
    print(f"✓ Downloaded: {len(response.content)} bytes")
    
    # Load image
    try:
        image = Image.open(input_path).convert('RGBA')
        print(f"✓ Image size: {image.size}")
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        traceback.print_exc()
        return False
    print("")
    
    # Initialize TRELLIS.2
    print("Loading TRELLIS.2 model...")
    try:
        sys.path.insert(0, '/app/trellis2')
        
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel
        print("✓ Imports successful")
        
        # Download model if not cached
        model_path = '/app/models/TRELLIS.2-4B'
        config_path = os.path.join(model_path, 'config.json')
        
        hf_token = os.environ.get('HF_TOKEN')
        
        if not os.path.exists(config_path):
            print(f"Model not cached, downloading from HuggingFace...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='microsoft/TRELLIS.2-4B',
                local_dir=model_path,
                token=hf_token
            )
            print("✓ Model downloaded")
        else:
            print(f"✓ Using cached model at {model_path}")
        
        print("Loading pipeline...")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        print("✓ Pipeline loaded")
        
        print("Moving to CUDA...")
        pipeline.cuda()
        print("✓ Pipeline on GPU")
        
        # Check GPU memory after model load
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"✓ GPU memory allocated: {allocated:.2f} GB")
            print(f"✓ GPU memory reserved: {reserved:.2f} GB")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        traceback.print_exc()
        return False
    print("")
    
    # Generate 3D
    print("Generating 3D asset...")
    print("This may take 10-60 seconds depending on resolution...")
    try:
        with torch.no_grad():
            meshes = pipeline.run(image)
        
        if not meshes or len(meshes) == 0:
            print("ERROR: No mesh generated (empty output)")
            return False
        
        mesh = meshes[0]
        print(f"✓ Generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"ERROR: Generation failed: {e}")
        traceback.print_exc()
        return False
    print("")
    
    # Simplify mesh
    print("Simplifying mesh...")
    try:
        mesh.simplify(16777216)  # nvdiffrast limit
        print(f"✓ Simplified: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"WARNING: Simplification failed: {e}")
        # Continue anyway
    print("")
    
    # Export to GLB
    print("Exporting to GLB...")
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   1000000,
            texture_size        =   4096,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        
        glb_path = os.path.join(artifact_dir, 'output.glb')
        glb.export(glb_path, extension_webp=True)
        
        file_size = os.path.getsize(glb_path)
        print(f"✓ Exported: {glb_path} ({file_size} bytes, {file_size/1e6:.2f} MB)")
    except Exception as e:
        print(f"ERROR: Export failed: {e}")
        traceback.print_exc()
        return False
    print("")
    
    print("=" * 60)
    print("SUCCESS - 3D asset generated")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("\nTask completed successfully (exit 0)")
        else:
            print("\nTask completed with errors (exit 0)")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        print("\nTask completed with exception (exit 0)")
    finally:
        log_file.close()
    
    # Always exit 0 for DOK compatibility
    sys.exit(0)
