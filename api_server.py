# api_server.py

import threading
import queue
import time
import uuid
import os
from flask import Flask, request, jsonify, send_from_directory, Response

# Import the refactored generation function
try:
    from t2v_module import generate_video
except ImportError:
    print("ERROR: Ensure 't2v_module.py' (the refactored t2v script) is in the same directory or Python path.")
    exit(1)

# --- Globals ---
app = Flask(__name__)
job_queue = queue.Queue()
jobs = {} # Dictionary to store job status and results
jobs_lock = threading.Lock() # To protect access to the jobs dictionary

# --- Configuration ---
# Get output directory from environment or use t2v_module's default
VIDEO_OUTPUT_DIR = os.path.abspath(os.environ.get("VIDEO_OUTPUT_DIR", "outputs"))
print(f"API Server: Videos will be saved to and served from: {VIDEO_OUTPUT_DIR}")
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True) # Ensure it exists

# --- Worker Thread ---
def worker():
    print("Worker thread started.")
    while True:
        print("Worker waiting for job...")
        job_id = job_queue.get() # Blocks until a job is available
        print(f"Worker picked up job: {job_id}")

        job_info = None
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['status'] = 'running'
                job_info = jobs[job_id].copy() # Get data needed for generation
            else:
                print(f"Worker: Job ID {job_id} not found in dictionary!")
                job_queue.task_done()
                continue # Skip if job somehow disappeared

        if job_info:
            print(f"Worker starting generation for job {job_id}...")
            video_path = None
            error_message = None
            try:
                # --- Call the refactored generate_video ---
                # Pass parameters from job_info or use defaults
                # Required:
                prompt = job_info.get('prompt')
                # Optional with defaults from t2v_module signature:
                resolution = job_info.get('resolution', "832x480")
                frames = job_info.get('frames', 81)
                steps = job_info.get('steps', 30)
                seed = job_info.get('seed', -1)
                guidance_scale = job_info.get('guidance_scale', 5.0)
                flow_shift = job_info.get('flow_shift', 5.0)
                negative_prompt = job_info.get('negative_prompt', "poor quality, low quality...")
                profile = job_info.get('profile', 4)
                attention = job_info.get('attention', "auto")
                vae_tile_size = job_info.get('vae_tile_size', 0)
                compile_model = job_info.get('compile', False)
                quantize_t5 = job_info.get('quantize', False)
                riflex = job_info.get('riflex', False)
                teacache = job_info.get('teacache', 0.0)
                teacache_start_perc = job_info.get('teacache_start_perc', 0)

                # Execute generation
                video_path = generate_video(
                    input_prompt=prompt,
                    output_dir=VIDEO_OUTPUT_DIR, # Use configured output dir
                    resolution=resolution,
                    frames=frames,
                    steps=steps,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    flow_shift=flow_shift,
                    negative_prompt=negative_prompt,
                    profile=profile,
                    attention=attention,
                    vae_tile_size=vae_tile_size,
                    compile_model=compile_model,
                    quantize_t5=quantize_t5,
                    riflex=riflex,
                    teacache=teacache,
                    teacache_start_perc=teacache_start_perc,
                    verbose=1 # Keep verbose moderate for API logs
                )

                if video_path is None:
                     # Generation might have failed internally but didn't raise Exception
                     raise RuntimeError("generate_video returned None, generation likely failed.")

            except Exception as e:
                print(f"Worker: Error during job {job_id}: {e}")
                error_message = str(e)

            # --- Update job status ---
            with jobs_lock:
                if job_id in jobs: # Check again in case job was somehow removed
                    if error_message:
                        jobs[job_id]['status'] = 'failed'
                        jobs[job_id]['result'] = error_message
                    else:
                        jobs[job_id]['status'] = 'completed'
                        jobs[job_id]['result'] = video_path # Store the full path
                    jobs[job_id]['end_time'] = time.time()
                else:
                    print(f"Worker: Job {job_id} disappeared before completion update!")

        job_queue.task_done()
        print(f"Worker finished job: {job_id}")

# Start the worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# --- API Endpoints ---

@app.route('/generate', methods=['POST'])
def handle_generate():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "'prompt' is required"}), 400

    job_id = uuid.uuid4().hex
    job_details = {
        'status': 'queued',
        'prompt': prompt,
        'result': None,
        'timestamp': time.time(),
        # Store any other parameters passed in the request
        **{k: v for k, v in data.items() if k != 'prompt'}
    }

    with jobs_lock:
        jobs[job_id] = job_details

    job_queue.put(job_id)
    print(f"Queued job {job_id} for prompt: '{prompt[:50]}...'")

    return jsonify({"job_id": job_id}), 202 # Accepted

@app.route('/status/<job_id>', methods=['GET'])
def handle_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job ID not found"}), 404

    response = {
        "job_id": job_id,
        "status": job['status'],
        "prompt": job.get('prompt', '[prompt not stored]'), # Optional: return prompt
    }
    if job['status'] == 'failed':
        response["error"] = job.get('result', 'Unknown error')
    # Optionally add timestamps etc.
    # if 'timestamp' in job: response['submitted_at'] = job['timestamp']
    # if 'end_time' in job: response['finished_at'] = job['end_time']

    return jsonify(response), 200

@app.route('/video/<job_id>', methods=['GET'])
def handle_video(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job ID not found"}), 404

    status = job.get('status')
    result_path = job.get('result')

    if status == 'completed':
        if result_path and os.path.isfile(result_path):
            try:
                # Serve the file
                # Ensure the path is relative to the configured output dir for security
                # and that send_from_directory gets the correct base directory.
                directory = os.path.dirname(result_path)
                filename = os.path.basename(result_path)
                # Basic check to prevent serving files outside the intended directory
                if os.path.abspath(directory) != VIDEO_OUTPUT_DIR:
                     print(f"Attempt to access file outside output directory blocked: {result_path}")
                     return jsonify({"error": "Access denied"}), 403
                print(f"Serving video: {filename} from {directory}")
                return send_from_directory(directory, filename, as_attachment=True)
            except Exception as e:
                print(f"Error serving file {result_path}: {e}")
                return jsonify({"error": "Could not serve video file"}), 500
        else:
            return jsonify({"error": "Video file not found or path missing"}), 404
    elif status == 'failed':
        return jsonify({"error": "Generation failed", "detail": job.get('result', 'Unknown error')}), 500
    elif status == 'running':
        return jsonify({"error": "Video generation in progress"}), 409 # 409 Conflict
    elif status == 'queued':
         return jsonify({"error": "Video generation is queued"}), 409
    else:
        return jsonify({"error": f"Unknown job status: {status}"}), 500


@app.route('/', methods=['GET'])
def handle_root():
    # Simple status page or info
    num_jobs = len(jobs)
    num_queued = job_queue.qsize()
    return jsonify({
        "message": "Wan2GP API Server Running",
        "total_jobs_tracked": num_jobs,
        "jobs_in_queue": num_queued
    })

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Wan2GP API server...")
    # Run Flask server - host='0.0.0.0' makes it accessible on the network
    app.run(host='0.0.0.0', port=3000, debug=False) # Set debug=False for production