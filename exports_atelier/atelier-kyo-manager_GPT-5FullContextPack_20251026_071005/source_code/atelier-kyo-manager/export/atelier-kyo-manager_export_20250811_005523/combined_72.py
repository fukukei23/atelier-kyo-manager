
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\benchmark.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import __init__  # noqa: F401. Walk-around to run this script directly
import coloredlogs

# import torch before onnxruntime so that onnxruntime uses the cuDNN in the torch package.
import torch
from benchmark_helper import measure_memory

SD_MODELS = {
    "1.5": "runwayml/stable-diffusion-v1-5",
    "2.0": "stabilityai/stable-diffusion-2",
    "2.1": "stabilityai/stable-diffusion-2-1",
    "xl-1.0": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "3.0M": "stabilityai/stable-diffusion-3-medium-diffusers",
    "3.5M": "stabilityai/stable-diffusion-3.5-medium",
    "3.5L": "stabilityai/stable-diffusion-3.5-large",
    "Flux.1S": "black-forest-labs/FLUX.1-schnell",
    "Flux.1D": "black-forest-labs/FLUX.1-dev",
}

PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def example_prompts():
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "cute grey cat with blue eyes, wearing a bowtie, acrylic painting",
        "a cute magical flying dog, fantasy art drawn by disney concept artists, highly detailed, digital painting",
        "an illustration of a house with large barn with many cute flower pots and beautiful blue sky scenery",
        "one apple sitting on a table, still life, reflective, full color photograph, centered, close-up product",
        "background texture of stones, masterpiece, artistic, stunning photo, award winner photo",
        "new international organic style house, tropical surroundings, architecture, 8k, hdr",
        "beautiful Renaissance Revival Estate, Hobbit-House, detailed painting, warm colors, 8k, trending on Artstation",
        "blue owl, big green eyes, portrait, intricate metal design, unreal engine, octane render, realistic",
        "delicate elvish moonstone necklace on a velvet background, symmetrical intricate motifs, leaves, flowers, 8k",
    ]

    negative_prompt = "bad composition, ugly, abnormal, malformed"

    return prompts, negative_prompt


def warmup_prompts():
    return "warm up", "bad"


def measure_gpu_memory(monitor_type, func, start_memory=None):
    return measure_memory(is_gpu=True, func=func, monitor_type=monitor_type, start_memory=start_memory)


def get_ort_pipeline(model_name: str, directory: str, provider, disable_safety_checker: bool):
    from diffusers import DDIMScheduler, OnnxStableDiffusionPipeline

    import onnxruntime

    if directory is not None:
        assert os.path.exists(directory)
        session_options = onnxruntime.SessionOptions()
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            directory,
            provider=provider,
            sess_options=session_options,
        )
    else:
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            model_name,
            revision="onnx",
            provider=provider,
            use_auth_token=True,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if disable_safety_checker:
        pipe.safety_checker = None
        pipe.feature_extractor = None

    return pipe


def get_torch_pipeline(model_name: str, disable_safety_checker: bool, enable_torch_compile: bool, use_xformers: bool):
    if "FLUX" in model_name:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        if enable_torch_compile:
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        return pipe

    if "stable-diffusion-3" in model_name:
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        if enable_torch_compile:
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        return pipe

    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from torch import channels_last, float16

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=float16).to("cuda")

    pipe.unet.to(memory_format=channels_last)  # in-place operation

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if enable_torch_compile:
        pipe.unet = torch.compile(pipe.unet)
        pipe.vae = torch.compile(pipe.vae)
        pipe.text_encoder = torch.compile(pipe.text_encoder)
        print("Torch compiled unet, vae and text_encoder")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if disable_safety_checker:
        pipe.safety_checker = None
        pipe.feature_extractor = None

    return pipe


def get_image_filename_prefix(engine: str, model_name: str, batch_size: int, steps: int, disable_safety_checker: bool):
    short_model_name = model_name.split("/")[-1].replace("stable-diffusion-", "sd")
    return f"{engine}_{short_model_name}_b{batch_size}_s{steps}" + ("" if disable_safety_checker else "_safe")


def run_ort_pipeline(
    pipe,
    batch_size: int,
    image_filename_prefix: str,
    height,
    width,
    steps,
    num_prompts,
    batch_count,
    start_memory,
    memory_monitor_type,
    skip_warmup: bool = False,
):
    from diffusers import OnnxStableDiffusionPipeline

    assert isinstance(pipe, OnnxStableDiffusionPipeline)

    prompts, negative_prompt = example_prompts()

    def warmup():
        if skip_warmup:
            return
        prompt, negative = warmup_prompts()
        pipe(
            prompt=[prompt] * batch_size,
            height=height,
            width=width,
            num_inference_steps=steps,
            negative_prompt=[negative] * batch_size,
        )

    # Run warm up, and measure GPU memory of two runs
    # cuDNN/MIOpen The first run has  algo search so it might need more memory)
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    latency_list = []
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        inference_start = time.time()
        images = pipe(
            prompt=[prompt] * batch_size,
            height=height,
            width=width,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt] * batch_size,
        ).images
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"Inference took {latency:.3f} seconds")
        for k, image in enumerate(images):
            image.save(f"{image_filename_prefix}_{i}_{k}.jpg")

    from onnxruntime import __version__ as ort_version

    return {
        "engine": "onnxruntime",
        "version": ort_version,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
    }


def get_negative_prompt_kwargs(negative_prompt, use_num_images_per_prompt, is_flux, batch_size) -> dict:
    # Flux does not support negative prompt
    kwargs = (
        (
            {"negative_prompt": negative_prompt}
            if use_num_images_per_prompt
            else {"negative_prompt": [negative_prompt] * batch_size}
        )
        if not is_flux
        else {}
    )

    # Fix the random seed so that we can inspect the output quality easily.
    if torch.cuda.is_available():
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(123)

    return kwargs


def run_torch_pipeline(
    pipe,
    batch_size: int,
    image_filename_prefix: str,
    height,
    width,
    steps,
    num_prompts,
    batch_count,
    start_memory,
    memory_monitor_type,
    skip_warmup=False,
):
    prompts, negative_prompt = example_prompts()

    import diffusers

    is_flux = isinstance(pipe, diffusers.FluxPipeline)

    def warmup():
        if skip_warmup:
            return
        prompt, negative = warmup_prompts()
        extra_kwargs = get_negative_prompt_kwargs(negative, False, is_flux, batch_size)
        pipe(prompt=[prompt] * batch_size, height=height, width=width, num_inference_steps=steps, **extra_kwargs)

    # Run warm up, and measure GPU memory of two runs (The first run has cuDNN algo search so it might need more memory)
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    torch.set_grad_enabled(False)

    latency_list = []
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        torch.cuda.synchronize()
        inference_start = time.time()
        extra_kwargs = get_negative_prompt_kwargs(negative_prompt, False, is_flux, batch_size)
        images = pipe(
            prompt=[prompt] * batch_size,
            height=height,
            width=width,
            num_inference_steps=steps,
            **extra_kwargs,
        ).images

        torch.cuda.synchronize()
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"Inference took {latency:.3f} seconds")
        for k, image in enumerate(images):
            image.save(f"{image_filename_prefix}_{i}_{k}.jpg")

    return {
        "engine": "torch",
        "version": torch.__version__,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
    }


def run_ort(
    model_name: str,
    directory: str,
    provider: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    tuning: bool,
    skip_warmup: bool = False,
):
    provider_and_options = provider
    if tuning and provider in ["CUDAExecutionProvider", "ROCMExecutionProvider"]:
        provider_and_options = (provider, {"tunable_op_enable": 1, "tunable_op_tuning_enable": 1})

    load_start = time.time()
    pipe = get_ort_pipeline(model_name, directory, provider_and_options, disable_safety_checker)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    image_filename_prefix = get_image_filename_prefix("ort", model_name, batch_size, steps, disable_safety_checker)
    result = run_ort_pipeline(
        pipe,
        batch_size,
        image_filename_prefix,
        height,
        width,
        steps,
        num_prompts,
        batch_count,
        start_memory,
        memory_monitor_type,
        skip_warmup=skip_warmup,
    )

    result.update(
        {
            "model_name": model_name,
            "directory": directory,
            "provider": provider.replace("ExecutionProvider", ""),
            "disable_safety_checker": disable_safety_checker,
            "enable_cuda_graph": False,
        }
    )
    return result


def get_optimum_ort_pipeline(
    model_name: str,
    directory: str,
    provider="CUDAExecutionProvider",
    disable_safety_checker: bool = True,
    use_io_binding: bool = False,
):
    from optimum.onnxruntime import ORTPipelineForText2Image

    if directory is not None and os.path.exists(directory):
        pipeline = ORTPipelineForText2Image.from_pretrained(directory, provider=provider, use_io_binding=use_io_binding)
    else:
        pipeline = ORTPipelineForText2Image.from_pretrained(
            model_name,
            export=True,
            provider=provider,
            use_io_binding=use_io_binding,
        )
        pipeline.save_pretrained(directory)

    if disable_safety_checker:
        pipeline.safety_checker = None
        pipeline.feature_extractor = None

    return pipeline


def run_optimum_ort_pipeline(
    pipe,
    batch_size: int,
    image_filename_prefix: str,
    height,
    width,
    steps,
    num_prompts,
    batch_count,
    start_memory,
    memory_monitor_type,
    use_num_images_per_prompt=False,
    skip_warmup=False,
):
    print("Pipeline type", type(pipe))
    from optimum.onnxruntime.modeling_diffusion import ORTFluxPipeline

    is_flux = isinstance(pipe, ORTFluxPipeline)

    prompts, negative_prompt = example_prompts()

    def warmup():
        if skip_warmup:
            return
        prompt, negative = warmup_prompts()
        extra_kwargs = get_negative_prompt_kwargs(negative, use_num_images_per_prompt, is_flux, batch_size)
        if use_num_images_per_prompt:
            pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                num_images_per_prompt=batch_count,
                **extra_kwargs,
            )
        else:
            pipe(prompt=[prompt] * batch_size, height=height, width=width, num_inference_steps=steps, **extra_kwargs)

    # Run warm up, and measure GPU memory of two runs.
    # The first run has algo search for cuDNN/MIOpen, so it might need more memory.
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    extra_kwargs = get_negative_prompt_kwargs(negative_prompt, use_num_images_per_prompt, is_flux, batch_size)

    latency_list = []
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        inference_start = time.time()
        if use_num_images_per_prompt:
            images = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                num_images_per_prompt=batch_size,
                **extra_kwargs,
            ).images
        else:
            images = pipe(
                prompt=[prompt] * batch_size, height=height, width=width, num_inference_steps=steps, **extra_kwargs
            ).images
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"Inference took {latency:.3f} seconds")
        for k, image in enumerate(images):
            image.save(f"{image_filename_prefix}_{i}_{k}.jpg")

    from onnxruntime import __version__ as ort_version

    return {
        "engine": "optimum_ort",
        "version": ort_version,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
    }


def run_optimum_ort(
    model_name: str,
    directory: str,
    provider: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    use_io_binding: bool = False,
    skip_warmup: bool = False,
):
    load_start = time.time()
    pipe = get_optimum_ort_pipeline(
        model_name, directory, provider, disable_safety_checker, use_io_binding=use_io_binding
    )
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    full_model_name = model_name + "_" + Path(directory).name if directory else model_name
    image_filename_prefix = get_image_filename_prefix(
        "optimum", full_model_name, batch_size, steps, disable_safety_checker
    )
    result = run_optimum_ort_pipeline(
        pipe,
        batch_size,
        image_filename_prefix,
        height,
        width,
        steps,
        num_prompts,
        batch_count,
        start_memory,
        memory_monitor_type,
        skip_warmup=skip_warmup,
    )

    result.update(
        {
            "model_name": model_name,
            "directory": directory,
            "provider": provider.replace("ExecutionProvider", ""),
            "disable_safety_checker": disable_safety_checker,
            "enable_cuda_graph": False,
        }
    )
    return result


def run_ort_trt_static(
    work_dir: str,
    version: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    max_batch_size: int,
    nvtx_profile: bool = False,
    use_cuda_graph: bool = True,
):
    print("[I] Initializing ORT TensorRT EP accelerated StableDiffusionXL txt2img pipeline (static input shape)")

    # Register TensorRT plugins
    from trt_utilities import init_trt_plugins

    init_trt_plugins()

    assert batch_size <= max_batch_size

    from diffusion_models import PipelineInfo

    pipeline_info = PipelineInfo(version)
    short_name = pipeline_info.short_name()

    from engine_builder import EngineType, get_engine_paths
    from pipeline_stable_diffusion import StableDiffusionPipeline

    engine_type = EngineType.ORT_TRT
    onnx_dir, engine_dir, output_dir, framework_model_dir, _ = get_engine_paths(work_dir, pipeline_info, engine_type)

    # Initialize pipeline
    pipeline = StableDiffusionPipeline(
        pipeline_info,
        scheduler="DDIM",
        output_dir=output_dir,
        verbose=False,
        nvtx_profile=nvtx_profile,
        max_batch_size=max_batch_size,
        use_cuda_graph=use_cuda_graph,
        framework_model_dir=framework_model_dir,
        engine_type=engine_type,
    )

    # Load TensorRT engines and pytorch modules
    pipeline.backend.build_engines(
        engine_dir,
        framework_model_dir,
        onnx_dir,
        17,
        opt_image_height=height,
        opt_image_width=width,
        opt_batch_size=batch_size,
        static_batch=True,
        static_image_shape=True,
        max_workspace_size=0,
        device_id=torch.cuda.current_device(),
    )

    # Here we use static batch and image size, so the resource allocation only need done once.
    # For dynamic batch and image size, some cost (like memory allocation) shall be included in latency.
    pipeline.load_resources(height, width, batch_size)

    def warmup():
        prompt, negative = warmup_prompts()
        pipeline.run([prompt] * batch_size, [negative] * batch_size, height, width, denoising_steps=steps)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    image_filename_prefix = get_image_filename_prefix("ort_trt", short_name, batch_size, steps, disable_safety_checker)

    latency_list = []
    prompts, negative_prompt = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        inference_start = time.time()
        # Use warmup mode here since non-warmup mode will save image to disk.
        images, pipeline_time = pipeline.run(
            [prompt] * batch_size,
            [negative_prompt] * batch_size,
            height,
            width,
            denoising_steps=steps,
            guidance=7.5,
            seed=123,
        )
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"End2End took {latency:.3f} seconds. Inference latency: {pipeline_time}")
        for k, image in enumerate(images):
            image.save(f"{image_filename_prefix}_{i}_{k}.jpg")

    pipeline.teardown()

    from tensorrt import __version__ as trt_version

    from onnxruntime import __version__ as ort_version

    return {
        "model_name": pipeline_info.name(),
        "engine": "onnxruntime",
        "version": ort_version,
        "provider": f"tensorrt({trt_version})",
        "directory": engine_dir,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "disable_safety_checker": disable_safety_checker,
        "enable_cuda_graph": use_cuda_graph,
    }


def run_tensorrt_static(
    work_dir: str,
    version: str,
    model_name: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    max_batch_size: int,
    nvtx_profile: bool = False,
    use_cuda_graph: bool = True,
    skip_warmup: bool = False,
):
    print("[I] Initializing TensorRT accelerated StableDiffusionXL txt2img pipeline (static input shape)")

    from cuda import cudart

    # Register TensorRT plugins
    from trt_utilities import init_trt_plugins

    init_trt_plugins()

    assert batch_size <= max_batch_size

    from diffusion_models import PipelineInfo

    pipeline_info = PipelineInfo(version)

    from engine_builder import EngineType, get_engine_paths
    from pipeline_stable_diffusion import StableDiffusionPipeline

    engine_type = EngineType.TRT
    onnx_dir, engine_dir, output_dir, framework_model_dir, timing_cache = get_engine_paths(
        work_dir, pipeline_info, engine_type
    )

    # Initialize pipeline
    pipeline = StableDiffusionPipeline(
        pipeline_info,
        scheduler="DDIM",
        output_dir=output_dir,
        verbose=False,
        nvtx_profile=nvtx_profile,
        max_batch_size=max_batch_size,
        use_cuda_graph=True,
        engine_type=engine_type,
    )

    # Load TensorRT engines and pytorch modules
    pipeline.backend.load_engines(
        engine_dir=engine_dir,
        framework_model_dir=framework_model_dir,
        onnx_dir=onnx_dir,
        onnx_opset=17,
        opt_batch_size=batch_size,
        opt_image_height=height,
        opt_image_width=width,
        static_batch=True,
        static_shape=True,
        enable_all_tactics=False,
        timing_cache=timing_cache,
    )

    # activate engines
    max_device_memory = max(pipeline.backend.max_device_memory(), pipeline.backend.max_device_memory())
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    pipeline.backend.activate_engines(shared_device_memory)

    # Here we use static batch and image size, so the resource allocation only need done once.
    # For dynamic batch and image size, some cost (like memory allocation) shall be included in latency.
    pipeline.load_resources(height, width, batch_size)

    def warmup():
        if skip_warmup:
            return
        prompt, negative = warmup_prompts()
        pipeline.run([prompt] * batch_size, [negative] * batch_size, height, width, denoising_steps=steps)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    image_filename_prefix = get_image_filename_prefix("trt", model_name, batch_size, steps, disable_safety_checker)

    latency_list = []
    prompts, negative_prompt = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        inference_start = time.time()
        # Use warmup mode here since non-warmup mode will save image to disk.
        images, pipeline_time = pipeline.run(
            [prompt] * batch_size,
            [negative_prompt] * batch_size,
            height,
            width,
            denoising_steps=steps,
            seed=123,
        )
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"End2End took {latency:.3f} seconds. Inference latency: {pipeline_time}")
        for k, image in enumerate(images):
            image.save(f"{image_filename_prefix}_{i}_{k}.jpg")

    pipeline.teardown()

    import tensorrt as trt

    return {
        "engine": "tensorrt",
        "version": trt.__version__,
        "provider": "default",
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "enable_cuda_graph": use_cuda_graph,
    }


def run_tensorrt_static_xl(
    work_dir: str,
    version: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    max_batch_size: int,
    nvtx_profile: bool = False,
    use_cuda_graph=True,
    skip_warmup: bool = False,
):
    print("[I] Initializing TensorRT accelerated StableDiffusionXL txt2img pipeline (static input shape)")

    import tensorrt as trt
    from cuda import cudart
    from trt_utilities import init_trt_plugins

    # Validate image dimensions
    image_height = height
    image_width = width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}."
        )

    # Register TensorRT plugins
    init_trt_plugins()

    assert batch_size <= max_batch_size

    from diffusion_models import PipelineInfo
    from engine_builder import EngineType, get_engine_paths

    def init_pipeline(pipeline_class, pipeline_info):
        engine_type = EngineType.TRT

        onnx_dir, engine_dir, output_dir, framework_model_dir, timing_cache = get_engine_paths(
            work_dir, pipeline_info, engine_type
        )

        # Initialize pipeline
        pipeline = pipeline_class(
            pipeline_info,
            scheduler="DDIM",
            output_dir=output_dir,
            verbose=False,
            nvtx_profile=nvtx_profile,
            max_batch_size=max_batch_size,
            use_cuda_graph=use_cuda_graph,
            framework_model_dir=framework_model_dir,
            engine_type=engine_type,
        )

        pipeline.backend.load_engines(
            engine_dir=engine_dir,
            framework_model_dir=framework_model_dir,
            onnx_dir=onnx_dir,
            onnx_opset=17,
            opt_batch_size=batch_size,
            opt_image_height=height,
            opt_image_width=width,
            static_batch=True,
            static_shape=True,
            enable_all_tactics=False,
            timing_cache=timing_cache,
        )
        return pipeline

    from pipeline_stable_diffusion import StableDiffusionPipeline

    pipeline_info = PipelineInfo(version)
    pipeline = init_pipeline(StableDiffusionPipeline, pipeline_info)

    max_device_memory = max(pipeline.backend.max_device_memory(), pipeline.backend.max_device_memory())
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    pipeline.backend.activate_engines(shared_device_memory)

    # Here we use static batch and image size, so the resource allocation only need done once.
    # For dynamic batch and image size, some cost (like memory allocation) shall be included in latency.
    pipeline.load_resources(image_height, image_width, batch_size)

    def run_sd_xl_inference(prompt, negative_prompt, seed=None):
        return pipeline.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            denoising_steps=steps,
            guidance=5.0,
            seed=seed,
        )

    def warmup():
        if skip_warmup:
            return
        prompt, negative = warmup_prompts()
        run_sd_xl_inference([prompt] * batch_size, [negative] * batch_size)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    model_name = pipeline_info.name()
    image_filename_prefix = get_image_filename_prefix("trt", model_name, batch_size, steps, disable_safety_checker)

    latency_list = []
    prompts, negative_prompt = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        inference_start = time.time()
        # Use warmup mode here since non-warmup mode will save image to disk.
        images, pipeline_time = run_sd_xl_inference([prompt] * batch_size, [negative_prompt] * batch_size, seed=123)
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"End2End took {latency:.3f} seconds. Inference latency: {pipeline_time}")
        for k, image in enumerate(images):
            image.save(f"{image_filename_prefix}_{i}_{k}.png")

    pipeline.teardown()

    return {
        "model_name": model_name,
        "engine": "tensorrt",
        "version": trt.__version__,
        "provider": "default",
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "enable_cuda_graph": use_cuda_graph,
    }


def run_ort_trt_xl(
    work_dir: str,
    version: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    max_batch_size: int,
    nvtx_profile: bool = False,
    use_cuda_graph=True,
    skip_warmup: bool = False,
):
    from demo_utils import initialize_pipeline
    from engine_builder import EngineType

    pipeline = initialize_pipeline(
        version=version,
        engine_type=EngineType.ORT_TRT,
        work_dir=work_dir,
        height=height,
        width=width,
        use_cuda_graph=use_cuda_graph,
        max_batch_size=max_batch_size,
        opt_batch_size=batch_size,
    )

    assert batch_size <= max_batch_size

    pipeline.load_resources(height, width, batch_size)

    def run_sd_xl_inference(prompt, negative_prompt, seed=None):
        return pipeline.run(
            prompt,
            negative_prompt,
            height,
            width,
            denoising_steps=steps,
            guidance=5.0,
            seed=seed,
        )

    def warmup():
        if skip_warmup:
            return
        prompt, negative = warmup_prompts()
        run_sd_xl_inference([prompt] * batch_size, [negative] * batch_size)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    model_name = pipeline.pipeline_info.name()
    image_filename_prefix = get_image_filename_prefix("ort_trt", model_name, batch_size, steps, disable_safety_checker)

    latency_list = []
    prompts, negative_prompt = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        inference_start = time.time()
        # Use warmup mode here since non-warmup mode will save image to disk.
        images, pipeline_time = run_sd_xl_inference([prompt] * batch_size, [negative_prompt] * batch_size, seed=123)
        inference_end = time.time()
        latency = inference_end - inference_start
        latency_list.append(latency)
        print(f"End2End took {latency:.3f} seconds. Inference latency: {pipeline_time}")
        for k, image in enumerate(images):
            filename = f"{image_filename_prefix}_{i}_{k}.png"
            image.save(filename)
            print("Image saved to", filename)

    pipeline.teardown()

    from tensorrt import __version__ as trt_version

    from onnxruntime import __version__ as ort_version

    return {
        "model_name": model_name,
        "engine": "onnxruntime",
        "version": ort_version,
        "provider": f"tensorrt{trt_version})",
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "enable_cuda_graph": use_cuda_graph,
    }


def run_torch(
    model_name: str,
    batch_size: int,
    disable_safety_checker: bool,
    enable_torch_compile: bool,
    use_xformers: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    skip_warmup: bool = True,
):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.set_grad_enabled(False)

    load_start = time.time()
    pipe = get_torch_pipeline(model_name, disable_safety_checker, enable_torch_compile, use_xformers)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    image_filename_prefix = get_image_filename_prefix("torch", model_name, batch_size, steps, disable_safety_checker)

    if not enable_torch_compile:
        with torch.inference_mode():
            result = run_torch_pipeline(
                pipe,
                batch_size,
                image_filename_prefix,
                height,
                width,
                steps,
                num_prompts,
                batch_count,
                start_memory,
                memory_monitor_type,
                skip_warmup=skip_warmup,
            )
    else:
        result = run_torch_pipeline(
            pipe,
            batch_size,
            image_filename_prefix,
            height,
            width,
            steps,
            num_prompts,
            batch_count,
            start_memory,
            memory_monitor_type,
            skip_warmup=skip_warmup,
        )

    result.update(
        {
            "model_name": model_name,
            "directory": None,
            "provider": "compile" if enable_torch_compile else "xformers" if use_xformers else "default",
            "disable_safety_checker": disable_safety_checker,
            "enable_cuda_graph": False,
        }
    )
    return result


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--engine",
        required=False,
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "optimum", "torch", "tensorrt"],
        help="Engines to benchmark. Default is onnxruntime.",
    )

    parser.add_argument(
        "-r",
        "--provider",
        required=False,
        type=str,
        default="cuda",
        choices=list(PROVIDERS.keys()),
        help="Provider to benchmark. Default is CUDAExecutionProvider.",
    )

    parser.add_argument(
        "-t",
        "--tuning",
        action="store_true",
        help="Enable TunableOp and tuning. "
        "This will incur longer warmup latency, and is mandatory for some operators of ROCm EP.",
    )

    parser.add_argument(
        "-v",
        "--version",
        required=False,
        type=str,
        choices=list(SD_MODELS.keys()),
        default="1.5",
        help="Stable diffusion version like 1.5, 2.0 or 2.1. Default is 1.5.",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        required=False,
        type=str,
        default=None,
        help="Directory of saved onnx pipeline. It could be the output directory of optimize_pipeline.py.",
    )

    parser.add_argument(
        "-w",
        "--work_dir",
        required=False,
        type=str,
        default=".",
        help="Root directory to save exported onnx models, built engines etc.",
    )

    parser.add_argument(
        "--enable_safety_checker",
        required=False,
        action="store_true",
        help="Enable safety checker",
    )
    parser.set_defaults(enable_safety_checker=False)

    parser.add_argument(
        "--enable_torch_compile",
        required=False,
        action="store_true",
        help="Enable compile unet for PyTorch 2.0",
    )
    parser.set_defaults(enable_torch_compile=False)

    parser.add_argument(
        "--use_xformers",
        required=False,
        action="store_true",
        help="Use xformers for PyTorch",
    )
    parser.set_defaults(use_xformers=False)

    parser.add_argument(
        "--use_io_binding",
        required=False,
        action="store_true",
        help="Use I/O Binding for Optimum.",
    )
    parser.set_defaults(use_io_binding=False)

    parser.add_argument(
        "--skip_warmup",
        required=False,
        action="store_true",
        help="No warmup.",
    )
    parser.set_defaults(skip_warmup=False)

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 8, 10, 16, 32],
        help="Number of images per batch. Default is 1.",
    )

    parser.add_argument(
        "--height",
        required=False,
        type=int,
        default=512,
        help="Output image height. Default is 512.",
    )

    parser.add_argument(
        "--width",
        required=False,
        type=int,
        default=512,
        help="Output image width. Default is 512.",
    )

    parser.add_argument(
        "-s",
        "--steps",
        required=False,
        type=int,
        default=50,
        help="Number of steps. Default is 50.",
    )

    parser.add_argument(
        "-n",
        "--num_prompts",
        required=False,
        type=int,
        default=10,
        help="Number of prompts. Default is 10.",
    )

    parser.add_argument(
        "-c",
        "--batch_count",
        required=False,
        type=int,
        choices=range(1, 11),
        default=5,
        help="Number of batches to test. Default is 5.",
    )

    parser.add_argument(
        "-m",
        "--max_trt_batch_size",
        required=False,
        type=int,
        choices=range(1, 16),
        default=4,
        help="Maximum batch size for TensorRT. Change the value may trigger TensorRT engine rebuild. Default is 4.",
    )

    parser.add_argument(
        "-g",
        "--enable_cuda_graph",
        required=False,
        action="store_true",
        help="Enable Cuda Graph. Requires onnxruntime >= 1.16",
    )
    parser.set_defaults(enable_cuda_graph=False)

    args = parser.parse_args()

    return args


def print_loaded_libraries(cuda_related_only=True):
    import psutil

    p = psutil.Process(os.getpid())
    for lib in p.memory_maps():
        if (not cuda_related_only) or any(x in lib.path for x in ("libcu", "libnv", "tensorrt")):
            print(lib.path)


def main():
    args = parse_arguments()
    print(args)

    if args.engine == "onnxruntime":
        if args.version in ["2.1"]:
            # Set a flag to avoid overflow in attention, which causes black image output in SD 2.1 model.
            # The environment variables shall be set before the first run of Attention or MultiHeadAttention operator.
            os.environ["ORT_DISABLE_TRT_FLASH_ATTENTION"] = "1"

        from packaging import version

        from onnxruntime import __version__ as ort_version

        if version.parse(ort_version) == version.parse("1.16.0"):
            # ORT 1.16 has a bug that might trigger Attention RuntimeError when latest fusion script is applied on clip model.
            # The walkaround is to enable fused causal attention, or disable Attention fusion for clip model.
            os.environ["ORT_ENABLE_FUSED_CAUSAL_ATTENTION"] = "1"

        if args.enable_cuda_graph:
            if not (args.engine == "onnxruntime" and args.provider in ["cuda", "tensorrt"] and args.pipeline is None):
                raise ValueError("The stable diffusion pipeline does not support CUDA graph.")

            if version.parse(ort_version) < version.parse("1.16"):
                raise ValueError("CUDA graph requires ONNX Runtime 1.16 or later")

    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    memory_monitor_type = "rocm" if args.provider == "rocm" else "cuda"

    start_memory = measure_gpu_memory(memory_monitor_type, None)
    print("GPU memory used before loading models:", start_memory)

    sd_model = SD_MODELS[args.version]
    provider = PROVIDERS[args.provider]
    if args.engine == "onnxruntime" and args.provider == "tensorrt":
        if "xl" in args.version:
            print("Testing Txt2ImgXLPipeline with static input shape. Backend is ORT TensorRT EP.")
            result = run_ort_trt_xl(
                work_dir=args.work_dir,
                version=args.version,
                batch_size=args.batch_size,
                disable_safety_checker=True,
                height=args.height,
                width=args.width,
                steps=args.steps,
                num_prompts=args.num_prompts,
                batch_count=args.batch_count,
                start_memory=start_memory,
                memory_monitor_type=memory_monitor_type,
                max_batch_size=args.max_trt_batch_size,
                nvtx_profile=False,
                use_cuda_graph=args.enable_cuda_graph,
                skip_warmup=args.skip_warmup,
            )
        else:
            print("Testing Txt2ImgPipeline with static input shape. Backend is ORT TensorRT EP.")
            result = run_ort_trt_static(
                work_dir=args.work_dir,
                version=args.version,
                batch_size=args.batch_size,
                disable_safety_checker=not args.enable_safety_checker,
                height=args.height,
                width=args.width,
                steps=args.steps,
                num_prompts=args.num_prompts,
                batch_count=args.batch_count,
                start_memory=start_memory,
                memory_monitor_type=memory_monitor_type,
                max_batch_size=args.max_trt_batch_size,
                nvtx_profile=False,
                use_cuda_graph=args.enable_cuda_graph,
                skip_warmup=args.skip_warmup,
            )
    elif args.engine == "optimum" and provider == "CUDAExecutionProvider":
        if "xl" in args.version:
            os.environ["ORT_ENABLE_FUSED_CAUSAL_ATTENTION"] = "1"

        result = run_optimum_ort(
            model_name=sd_model,
            directory=args.pipeline,
            provider=provider,
            batch_size=args.batch_size,
            disable_safety_checker=not args.enable_safety_checker,
            height=args.height,
            width=args.width,
            steps=args.steps,
            num_prompts=args.num_prompts,
            batch_count=args.batch_count,
            start_memory=start_memory,
            memory_monitor_type=memory_monitor_type,
            use_io_binding=args.use_io_binding,
            skip_warmup=args.skip_warmup,
        )
    elif args.engine == "onnxruntime":
        assert args.pipeline and os.path.isdir(args.pipeline), (
            "--pipeline should be specified for the directory of ONNX models"
        )
        print(f"Testing diffusers StableDiffusionPipeline with {provider} provider and tuning={args.tuning}")
        result = run_ort(
            model_name=sd_model,
            directory=args.pipeline,
            provider=provider,
            batch_size=args.batch_size,
            disable_safety_checker=not args.enable_safety_checker,
            height=args.height,
            width=args.width,
            steps=args.steps,
            num_prompts=args.num_prompts,
            batch_count=args.batch_count,
            start_memory=start_memory,
            memory_monitor_type=memory_monitor_type,
            tuning=args.tuning,
            skip_warmup=args.skip_warmup,
        )
    elif args.engine == "tensorrt" and "xl" in args.version:
        print("Testing Txt2ImgXLPipeline with static input shape. Backend is TensorRT.")
        result = run_tensorrt_static_xl(
            work_dir=args.work_dir,
            version=args.version,
            batch_size=args.batch_size,
            disable_safety_checker=True,
            height=args.height,
            width=args.width,
            steps=args.steps,
            num_prompts=args.num_prompts,
            batch_count=args.batch_count,
            start_memory=start_memory,
            memory_monitor_type=memory_monitor_type,
            max_batch_size=args.max_trt_batch_size,
            nvtx_profile=False,
            use_cuda_graph=args.enable_cuda_graph,
            skip_warmup=args.skip_warmup,
        )
    elif args.engine == "tensorrt":
        print("Testing Txt2ImgPipeline with static input shape. Backend is TensorRT.")
        result = run_tensorrt_static(
            work_dir=args.work_dir,
            version=args.version,
            model_name=sd_model,
            batch_size=args.batch_size,
            disable_safety_checker=True,
            height=args.height,
            width=args.width,
            steps=args.steps,
            num_prompts=args.num_prompts,
            batch_count=args.batch_count,
            start_memory=start_memory,
            memory_monitor_type=memory_monitor_type,
            max_batch_size=args.max_trt_batch_size,
            nvtx_profile=False,
            use_cuda_graph=args.enable_cuda_graph,
            skip_warmup=args.skip_warmup,
        )
    else:
        print(
            f"Testing Txt2ImgPipeline with dynamic input shape. Backend is PyTorch: compile={args.enable_torch_compile}, xformers={args.use_xformers}."
        )
        result = run_torch(
            model_name=sd_model,
            batch_size=args.batch_size,
            disable_safety_checker=not args.enable_safety_checker,
            enable_torch_compile=args.enable_torch_compile,
            use_xformers=args.use_xformers,
            height=args.height,
            width=args.width,
            steps=args.steps,
            num_prompts=args.num_prompts,
            batch_count=args.batch_count,
            start_memory=start_memory,
            memory_monitor_type=memory_monitor_type,
            skip_warmup=args.skip_warmup,
        )

    print(result)

    with open("benchmark_result.csv", mode="a", newline="") as csv_file:
        column_names = [
            "model_name",
            "directory",
            "engine",
            "version",
            "provider",
            "disable_safety_checker",
            "height",
            "width",
            "steps",
            "batch_size",
            "batch_count",
            "num_prompts",
            "average_latency",
            "median_latency",
            "first_run_memory_MB",
            "second_run_memory_MB",
            "enable_cuda_graph",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        csv_writer.writerow(result)

    # Show loaded DLLs when steps == 1 for debugging purpose.
    if args.steps == 1:
        print_loaded_libraries(args.provider in ["cuda", "tensorrt"])


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exception(*sys.exc_info())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\fancysets.py ===
from functools import reduce
from itertools import product

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.intfunc import igcd
from sympy.core.numbers import oo, Rational, Integer
from sympy.core.relational import Eq, is_eq
from sympy.core.kind import NumberKind
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.logic.boolalg import And, Or
from .sets import tfn, Set, Interval, Union, FiniteSet, ProductSet, SetKind
from sympy.utilities.misc import filldedent


class Rationals(Set, metaclass=Singleton):
    """
    Represents the rational numbers. This set is also available as
    the singleton ``S.Rationals``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_rational]

    def __iter__(self):
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return S.Reals

    def _kind(self):
        return SetKind(NumberKind)


class Naturals(Set, metaclass=Singleton):
    """
    Represents the natural numbers (or counting numbers) which are all
    positive integers starting from 1. This set is also available as
    the singleton ``S.Naturals``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Naturals)
    >>> next(iterable)
    1
    >>> next(iterable)
    2
    >>> next(iterable)
    3
    >>> pprint(S.Naturals.intersect(Interval(0, 10)))
    {1, 2, ..., 10}

    See Also
    ========

    Naturals0 : non-negative integers (i.e. includes 0, too)
    Integers : also includes negative integers
    """

    is_iterable = True
    _inf: Integer = S.One
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_positive and other.is_integer:
            return S.true
        elif other.is_integer is False or other.is_positive is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(1, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(1, oo).is_superset(other)

    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        return And(Eq(floor(x), x), x >= self.inf, x < oo)

    def _kind(self):
        return SetKind(NumberKind)


class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========

    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_integer and other.is_nonnegative:
            return S.true
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(oo).is_superset(other)


class Integers(Set, metaclass=Singleton):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the singleton ``S.Integers``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """

    is_iterable = True
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_integer]

    def __iter__(self):
        yield S.Zero
        i = S.One
        while True:
            yield i
            yield -i
            i = i + 1

    @property
    def _inf(self):
        return S.NegativeInfinity

    @property
    def _sup(self):
        return S.Infinity

    @property
    def _boundary(self):
        return self

    def _kind(self):
        return SetKind(NumberKind)

    def as_relational(self, x):
        return And(Eq(floor(x), x), -oo < x, x < oo)

    def _eval_is_subset(self, other):
        return Range(-oo, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(-oo, oo).is_superset(other)


class Reals(Interval, metaclass=Singleton):
    """
    Represents all real numbers
    from negative infinity to positive infinity,
    including all integer, rational and irrational numbers.
    This set is also available as the singleton ``S.Reals``.


    Examples
    ========

    >>> from sympy import S, Rational, pi, I
    >>> 5 in S.Reals
    True
    >>> Rational(-1, 2) in S.Reals
    True
    >>> pi in S.Reals
    True
    >>> 3*I in S.Reals
    False
    >>> S.Reals.contains(pi)
    True


    See Also
    ========

    ComplexRegion
    """
    @property
    def start(self):
        return S.NegativeInfinity

    @property
    def end(self):
        return S.Infinity

    @property
    def left_open(self):
        return True

    @property
    def right_open(self):
        return True

    def __eq__(self, other):
        return other == Interval(S.NegativeInfinity, S.Infinity)

    def __hash__(self):
        return hash(Interval(S.NegativeInfinity, S.Infinity))


class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from ``imageset``.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy import FiniteSet, ImageSet, Interval

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    {1, 4, 9}

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in ``base_set`` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    {0}

    See Also
    ========

    sympy.sets.sets.imageset
    """
    def __new__(cls, flambda, *sets):
        if not isinstance(flambda, Lambda):
            raise ValueError('First argument must be a Lambda')

        signature = flambda.signature

        if len(signature) != len(sets):
            raise ValueError('Incompatible signature')

        sets = [_sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Set arguments to ImageSet should of type Set")

        if not all(cls._check_sig(sg, st) for sg, st in zip(signature, sets)):
            raise ValueError("Signature %s does not match sets %s" % (signature, sets))

        if flambda is S.IdentityFunction and len(sets) == 1:
            return sets[0]

        if not set(flambda.variables) & flambda.expr.free_symbols:
            is_empty = fuzzy_or(s.is_empty for s in sets)
            if is_empty == True:
                return S.EmptySet
            elif is_empty == False:
                return FiniteSet(flambda.expr)

        return Basic.__new__(cls, flambda, *sets)

    lamda = property(lambda self: self.args[0])
    base_sets = property(lambda self: self.args[1:])

    @property
    def base_set(self):
        # XXX: Maybe deprecate this? It is poorly defined in handling
        # the multivariate case...
        sets = self.base_sets
        if len(sets) == 1:
            return sets[0]
        else:
            return ProductSet(*sets).flatten()

    @property
    def base_pset(self):
        return ProductSet(*self.base_sets)

    @classmethod
    def _check_sig(cls, sig_i, set_i):
        if sig_i.is_symbol:
            return True
        elif isinstance(set_i, ProductSet):
            sets = set_i.sets
            if len(sig_i) != len(sets):
                return False
            # Recurse through the signature for nested tuples:
            return all(cls._check_sig(ts, ps) for ts, ps in zip(sig_i, sets))
        else:
            # XXX: Need a better way of checking whether a set is a set of
            # Tuples or not. For example a FiniteSet can contain Tuples
            # but so can an ImageSet or a ConditionSet. Others like
            # Integers, Reals etc can not contain Tuples. We could just
            # list the possibilities here... Current code for e.g.
            # _contains probably only works for ProductSet.
            return True # Give the benefit of the doubt

    def __iter__(self):
        already_seen = set()
        for i in self.base_pset:
            val = self.lamda(*i)
            if val in already_seen:
                continue
            else:
                already_seen.add(val)
                yield val

    def _is_multivariate(self):
        return len(self.lamda.variables) > 1

    def _contains(self, other):
        from sympy.solvers.solveset import _solveset_multi

        def get_symsetmap(signature, base_sets):
            '''Attempt to get a map of symbols to base_sets'''
            queue = list(zip(signature, base_sets))
            symsetmap = {}
            for sig, base_set in queue:
                if sig.is_symbol:
                    symsetmap[sig] = base_set
                elif base_set.is_ProductSet:
                    sets = base_set.sets
                    if len(sig) != len(sets):
                        raise ValueError("Incompatible signature")
                    # Recurse
                    queue.extend(zip(sig, sets))
                else:
                    # If we get here then we have something like sig = (x, y) and
                    # base_set = {(1, 2), (3, 4)}. For now we give up.
                    return None

            return symsetmap

        def get_equations(expr, candidate):
            '''Find the equations relating symbols in expr and candidate.'''
            queue = [(expr, candidate)]
            for e, c in queue:
                if not isinstance(e, Tuple):
                    yield Eq(e, c)
                elif not isinstance(c, Tuple) or len(e) != len(c):
                    yield False
                    return
                else:
                    queue.extend(zip(e, c))

        # Get the basic objects together:
        other = _sympify(other)
        expr = self.lamda.expr
        sig = self.lamda.signature
        variables = self.lamda.variables
        base_sets = self.base_sets

        # Use dummy symbols for ImageSet parameters so they don't match
        # anything in other
        rep = {v: Dummy(v.name) for v in variables}
        variables = [v.subs(rep) for v in variables]
        sig = sig.subs(rep)
        expr = expr.subs(rep)

        # Map the parts of other to those in the Lambda expr
        equations = []
        for eq in get_equations(expr, other):
            # Unsatisfiable equation?
            if eq is False:
                return S.false
            equations.append(eq)

        # Map the symbols in the signature to the corresponding domains
        symsetmap = get_symsetmap(sig, base_sets)
        if symsetmap is None:
            # Can't factor the base sets to a ProductSet
            return None

        # Which of the variables in the Lambda signature need to be solved for?
        symss = (eq.free_symbols for eq in equations)
        variables = set(variables) & reduce(set.union, symss, set())

        # Use internal multivariate solveset
        variables = tuple(variables)
        base_sets = [symsetmap[v] for v in variables]
        solnset = _solveset_multi(equations, variables, base_sets)
        if solnset is None:
            return None
        return tfn[fuzzy_not(solnset.is_empty)]

    @property
    def is_iterable(self):
        return all(s.is_iterable for s in self.base_sets)

    def doit(self, **hints):
        from sympy.sets.setexpr import SetExpr
        f = self.lamda
        sig = f.signature
        if len(sig) == 1 and sig[0].is_symbol and isinstance(f.expr, Expr):
            base_set = self.base_sets[0]
            return SetExpr(base_set)._eval_func(f).set
        if all(s.is_FiniteSet for s in self.base_sets):
            return FiniteSet(*(f(*a) for a in product(*self.base_sets)))
        return self

    def _kind(self):
        return SetKind(self.lamda.expr.kind)


class Range(Set):
    """
    Represents a range of integers. Can be called as ``Range(stop)``,
    ``Range(start, stop)``, or ``Range(start, stop, step)``; when ``step`` is
    not given it defaults to 1.

    ``Range(stop)`` is the same as ``Range(0, stop, 1)`` and the stop value
    (just as for Python ranges) is not included in the Range values.

        >>> from sympy import Range
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(0, 10, 3)
        Range(0, 12, 3)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        TypeError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although ``Range`` is a :class:`Set` (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where ``range`` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet
        >>> Range(3).intersect(Range(4, oo))
        EmptySet

    Range will accept symbolic arguments but has very limited support
    for doing anything other than displaying the Range:

        >>> from sympy import Symbol, pprint
        >>> from sympy.abc import i, j, k
        >>> Range(i, j, k).start
        i
        >>> Range(i, j, k).inf
        Traceback (most recent call last):
        ...
        ValueError: invalid method for symbolic range

    Better success will be had when using integer symbols:

        >>> n = Symbol('n', integer=True)
        >>> r = Range(n, n + 20, 3)
        >>> r.inf
        n
        >>> pprint(r)
        {n, n + 3, ..., n + 18}
    """

    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], range):
                raise TypeError(
                    'use sympify(%s) to convert range to Range' % args[0])

        # expand range
        slc = slice(*args)

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        try:
            ok = []
            for w in (start, stop, step):
                w = sympify(w)
                if w in [S.NegativeInfinity, S.Infinity] or (
                        w.has(Symbol) and w.is_integer != False):
                    ok.append(w)
                elif not w.is_Integer:
                    if w.is_infinite:
                        raise ValueError('infinite symbols not allowed')
                    raise ValueError
                else:
                    ok.append(w)
        except ValueError:
            raise ValueError(filldedent('''
    Finite arguments to Range must be integers; `imageset` can define
    other cases, e.g. use `imageset(i, i/10, Range(3))` to give
    [0, 1/10, 1/5].'''))
        start, stop, step = ok

        null = False
        if any(i.has(Symbol) for i in (start, stop, step)):
            dif = stop - start
            n = dif/step
            if n.is_Rational:
                if dif == 0:
                    null = True
                else:  # (x, x + 5, 2) or (x, 3*x, x)
                    n = floor(n)
                    end = start + n*step
                    if dif.is_Rational:  # (x, x + 5, 2)
                        if (end - stop).is_negative:
                            end += step
                    else:  # (x, 3*x, x)
                        if (end/stop - 1).is_negative:
                            end += step
            elif n.is_extended_negative:
                null = True
            else:
                end = stop  # other methods like sup and reversed must fail
        elif start.is_infinite:
            span = step*(stop - start)
            if span is S.NaN or span <= 0:
                null = True
            elif step.is_Integer and stop.is_infinite and abs(step) != 1:
                raise ValueError(filldedent('''
                    Step size must be %s in this case.''' % (1 if step > 0 else -1)))
            else:
                end = stop
        else:
            oostep = step.is_infinite
            if oostep:
                step = S.One if step > 0 else S.NegativeOne
            n = ceiling((stop - start)/step)
            if n <= 0:
                null = True
            elif oostep:
                step = S.One  # make it canonical
                end = start + step
            else:
                end = start + n*step
        if null:
            start = end = S.Zero
            step = S.One
        return Basic.__new__(cls, start, end, step)

    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])

    @property
    def reversed(self):
        """Return an equivalent Range in the opposite order.

        Examples
        ========

        >>> from sympy import Range
        >>> Range(10).reversed
        Range(9, -1, -1)
        """
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                raise ValueError('invalid method for symbolic range')
        if self.start == self.stop:
            return self
        return self.func(
            self.stop - self.step, self.start - self.step, -self.step)

    def _kind(self):
        return SetKind(NumberKind)

    def _contains(self, other):
        if self.start == self.stop:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return tfn[other.is_integer]
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                return
        else:
            n = self.size
        if self.start.is_finite:
            ref = self.start
        elif self.stop.is_finite:
            ref = self.stop
        else:  # both infinite; step is +/- 1 (enforced by __new__)
            return S.true
        if n == 1:
            return Eq(other, self[0])
        res = (ref - other) % self.step
        if res == S.Zero:
            if self.has(Symbol):
                d = Dummy('i')
                return self.as_relational(d).subs(d, other)
            return And(other >= self.inf, other <= self.sup)
        elif res.is_Integer:  # off sequence
            return S.false
        else:  # symbolic/unsimplified residue modulo step
            return None

    def __iter__(self):
        n = self.size  # validate
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            raise TypeError("Cannot iterate over symbolic Range")
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise TypeError("Cannot iterate over Range with infinite start")
        elif self.start != self.stop:
            i = self.start
            if n.is_infinite:
                while True:
                    yield i
                    i += self.step
            else:
                for _ in range(n):
                    yield i
                    i += self.step

    @property
    def is_iterable(self):
        # Check that size can be determined, used by __iter__
        dif = self.stop - self.start
        n = dif/self.step
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            return False
        if self.start in [S.NegativeInfinity, S.Infinity]:
            return False
        if not (n.is_extended_nonnegative and all(i.is_integer for i in self.args)):
            return False
        return True

    def __len__(self):
        rv = self.size
        if rv is S.Infinity:
            raise ValueError('Use .size to get the length of an infinite Range')
        return int(rv)

    @property
    def size(self):
        if self.start == self.stop:
            return S.Zero
        dif = self.stop - self.start
        n = dif/self.step
        if n.is_infinite:
            return S.Infinity
        if  n.is_extended_nonnegative and all(i.is_integer for i in self.args):
            return abs(floor(n))
        raise ValueError('Invalid method for symbolic Range')

    @property
    def is_finite_set(self):
        if self.start.is_integer and self.stop.is_integer:
            return True
        return self.size.is_finite

    @property
    def is_empty(self):
        try:
            return self.size.is_zero
        except ValueError:
            return None

    def __bool__(self):
        # this only distinguishes between definite null range
        # and non-null/unknown null; getting True doesn't mean
        # that it actually is not null
        b = is_eq(self.start, self.stop)
        if b is None:
            raise ValueError('cannot tell if Range is null or not')
        return not bool(b)

    def __getitem__(self, i):
        ooslice = "cannot slice from the end with an infinite value"
        zerostep = "slice step cannot be zero"
        infinite = "slicing not possible on range with infinite start"
        # if we had to take every other element in the following
        # oo, ..., 6, 4, 2, 0
        # we might get oo, ..., 4, 0 or oo, ..., 6, 2
        ambiguous = "cannot unambiguously re-stride from the end " + \
            "with an infinite value"
        if isinstance(i, slice):
            if self.size.is_finite:  # validates, too
                if self.start == self.stop:
                    return Range(0)
                start, stop, step = i.indices(self.size)
                n = ceiling((stop - start)/step)
                if n <= 0:
                    return Range(0)
                canonical_stop = start + n*step
                end = canonical_stop - step
                ss = step*self.step
                return Range(self[start], self[end] + ss, ss)
            else:  # infinite Range
                start = i.start
                stop = i.stop
                if i.step == 0:
                    raise ValueError(zerostep)
                step = i.step or 1
                ss = step*self.step
                #---------------------
                # handle infinite Range
                #   i.e. Range(-oo, oo) or Range(oo, -oo, -1)
                # --------------------
                if self.start.is_infinite and self.stop.is_infinite:
                    raise ValueError(infinite)
                #---------------------
                # handle infinite on right
                #   e.g. Range(0, oo) or Range(0, -oo, -1)
                # --------------------
                if self.stop.is_infinite:
                    # start and stop are not interdependent --
                    # they only depend on step --so we use the
                    # equivalent reversed values
                    return self.reversed[
                        stop if stop is None else -stop + 1:
                        start if start is None else -start:
                        step].reversed
                #---------------------
                # handle infinite on the left
                #   e.g. Range(oo, 0, -1) or Range(-oo, 0)
                # --------------------
                # consider combinations of
                # start/stop {== None, < 0, == 0, > 0} and
                # step {< 0, > 0}
                if start is None:
                    if stop is None:
                        if step < 0:
                            return Range(self[-1], self.start, ss)
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step < 0:
                            return Range(self[-1], self[stop], ss)
                        else:  # > 0
                            return Range(self.start, self[stop], ss)
                    elif stop == 0:
                        if step > 0:
                            return Range(0)
                        else:  # < 0
                            raise ValueError(ooslice)
                    elif stop == 1:
                        if step > 0:
                            raise ValueError(ooslice)  # infinite singleton
                        else:  # < 0
                            raise ValueError(ooslice)
                    else:  # > 1
                        raise ValueError(ooslice)
                elif start < 0:
                    if stop is None:
                        if step < 0:
                            return Range(self[start], self.start, ss)
                        else:  # > 0
                            return Range(self[start], self.stop, ss)
                    elif stop < 0:
                        return Range(self[start], self[stop], ss)
                    elif stop == 0:
                        if step < 0:
                            raise ValueError(ooslice)
                        else:  # > 0
                            return Range(0)
                    elif stop > 0:
                        raise ValueError(ooslice)
                elif start == 0:
                    if stop is None:
                        if step < 0:
                            raise ValueError(ooslice)  # infinite singleton
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step > 1:
                            raise ValueError(ambiguous)
                        elif step == 1:
                            return Range(self.start, self[stop], ss)
                        else:  # < 0
                            return Range(0)
                    else:  # >= 0
                        raise ValueError(ooslice)
                elif start > 0:
                    raise ValueError(ooslice)
        else:
            if self.start == self.stop:
                raise IndexError('Range index out of range')
            if not (all(i.is_integer or i.is_infinite
                    for i in self.args) and ((self.stop - self.start)/
                    self.step).is_extended_positive):
                raise ValueError('Invalid method for symbolic Range')
            if i == 0:
                if self.start.is_infinite:
                    raise ValueError(ooslice)
                return self.start
            if i == -1:
                if self.stop.is_infinite:
                    raise ValueError(ooslice)
                return self.stop - self.step
            n = self.size  # must be known for any other index
            rv = (self.stop if i < 0 else self.start) + i*self.step
            if rv.is_infinite:
                raise ValueError(ooslice)
            val = (rv - self.start)/self.step
            rel = fuzzy_or([val.is_infinite,
                            fuzzy_and([val.is_nonnegative, (n-val).is_nonnegative])])
            if rel:
                return rv
            if rel is None:
                raise ValueError('Invalid method for symbolic Range')
            raise IndexError("Range index out of range")

    @property
    def _inf(self):
        if not self:
            return S.EmptySet.inf
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                if self.step.is_positive and dif.is_positive:
                    return self.start
                elif self.step.is_negative and dif.is_negative:
                    return self.stop - self.step
            raise ValueError('invalid method for symbolic range')
        if self.step > 0:
            return self.start
        else:
            return self.stop - self.step

    @property
    def _sup(self):
        if not self:
            return S.EmptySet.sup
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                if self.step.is_positive and dif.is_positive:
                    return self.stop - self.step
                elif self.step.is_negative and dif.is_negative:
                    return self.start
            raise ValueError('invalid method for symbolic range')
        if self.step > 0:
            return self.stop - self.step
        else:
            return self.start

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        if self.start.is_infinite:
            assert not self.stop.is_infinite  # by instantiation
            a = self.reversed.start
        else:
            a = self.start
        step = self.step
        in_seq = Eq(Mod(x - a, step), 0)
        ints = And(Eq(Mod(a, 1), 0), Eq(Mod(step, 1), 0))
        n = (self.stop - self.start)/self.step
        if n == 0:
            return S.EmptySet.as_relational(x)
        if n == 1:
            return And(Eq(x, a), ints)
        try:
            a, b = self.inf, self.sup
        except ValueError:
            a = None
        if a is not None:
            range_cond = And(
                x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b)
        else:
            a, b = self.start, self.stop - self.step
            range_cond = Or(
                And(self.step >= 1, x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b),
                And(self.step <= -1, x < a if a.is_infinite else x <= a,
                x > b if b.is_infinite else x >= b))
        return And(in_seq, ints, range_cond)


_sympy_converter[range] = lambda r: Range(r.start, r.stop, r.step)

def normalize_theta_set(theta):
    r"""
    Normalize a Real Set `theta` in the interval `[0, 2\pi)`. It returns
    a normalized value of theta in the Set. For Interval, a maximum of
    one cycle $[0, 2\pi]$, is returned i.e. for theta equal to $[0, 10\pi]$,
    returned normalized value would be $[0, 2\pi)$. As of now intervals
    with end points as non-multiples of ``pi`` is not supported.

    Raises
    ======

    NotImplementedError
        The algorithms for Normalizing theta Set are not yet
        implemented.
    ValueError
        The input is not valid, i.e. the input is not a real set.
    RuntimeError
        It is a bug, please report to the github issue tracker.

    Examples
    ========

    >>> from sympy.sets.fancysets import normalize_theta_set
    >>> from sympy import Interval, FiniteSet, pi
    >>> normalize_theta_set(Interval(9*pi/2, 5*pi))
    Interval(pi/2, pi)
    >>> normalize_theta_set(Interval(-3*pi/2, pi/2))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-pi/2, pi/2))
    Union(Interval(0, pi/2), Interval.Ropen(3*pi/2, 2*pi))
    >>> normalize_theta_set(Interval(-4*pi, 3*pi))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
    Interval(pi/2, 3*pi/2)
    >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
    {0, pi}

    """
    from sympy.functions.elementary.trigonometric import _pi_coeff

    if theta.is_Interval:
        interval_len = theta.measure
        # one complete circle
        if interval_len >= 2*S.Pi:
            if interval_len == 2*S.Pi and theta.left_open and theta.right_open:
                k = _pi_coeff(theta.start)
                return Union(Interval(0, k*S.Pi, False, True),
                        Interval(k*S.Pi, 2*S.Pi, True, True))
            return Interval(0, 2*S.Pi, False, True)

        k_start, k_end = _pi_coeff(theta.start), _pi_coeff(theta.end)

        if k_start is None or k_end is None:
            raise NotImplementedError("Normalizing theta without pi as coefficient is "
                                    "not yet implemented")
        new_start = k_start*S.Pi
        new_end = k_end*S.Pi

        if new_start > new_end:
            return Union(Interval(S.Zero, new_end, False, theta.right_open),
                         Interval(new_start, 2*S.Pi, theta.left_open, True))
        else:
            return Interval(new_start, new_end, theta.left_open, theta.right_open)

    elif theta.is_FiniteSet:
        new_theta = []
        for element in theta:
            k = _pi_coeff(element)
            if k is None:
                raise NotImplementedError('Normalizing theta without pi as '
                                          'coefficient, is not Implemented.')
            else:
                new_theta.append(k*S.Pi)
        return FiniteSet(*new_theta)

    elif theta.is_Union:
        return Union(*[normalize_theta_set(interval) for interval in theta.args])

    elif theta.is_subset(S.Reals):
        raise NotImplementedError("Normalizing theta when, it is of type %s is not "
                                  "implemented" % type(theta))
    else:
        raise ValueError(" %s is not a real set" % (theta))


class ComplexRegion(Set):
    r"""
    Represents the Set of all Complex Numbers. It can represent a
    region of Complex Plane in both the standard forms Polar and
    Rectangular coordinates.

    * Polar Form
      Input is in the form of the ProductSet or Union of ProductSets
      of the intervals of ``r`` and ``theta``, and use the flag ``polar=True``.

      .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    * Rectangular Form
      Input is in the form of the ProductSet or Union of ProductSets
      of interval of x and y, the real and imaginary parts of the Complex numbers in a plane.
      Default input type is in rectangular form.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, S, I, Union
    >>> a = Interval(2, 3)
    >>> b = Interval(4, 6)
    >>> c1 = ComplexRegion(a*b)  # Rectangular Form
    >>> c1
    CartesianComplexRegion(ProductSet(Interval(2, 3), Interval(4, 6)))

    * c1 represents the rectangular region in complex plane
      surrounded by the coordinates (2, 4), (3, 4), (3, 6) and
      (2, 6), of the four vertices.

    >>> c = Interval(1, 8)
    >>> c2 = ComplexRegion(Union(a*b, b*c))
    >>> c2
    CartesianComplexRegion(Union(ProductSet(Interval(2, 3), Interval(4, 6)), ProductSet(Interval(4, 6), Interval(1, 8))))

    * c2 represents the Union of two rectangular regions in complex
      plane. One of them surrounded by the coordinates of c1 and
      other surrounded by the coordinates (4, 1), (6, 1), (6, 8) and
      (4, 8).

    >>> 2.5 + 4.5*I in c1
    True
    >>> 2.5 + 6.5*I in c1
    False

    >>> r = Interval(0, 1)
    >>> theta = Interval(0, 2*S.Pi)
    >>> c2 = ComplexRegion(r*theta, polar=True)  # Polar Form
    >>> c2  # unit Disk
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval.Ropen(0, 2*pi)))

    * c2 represents the region in complex plane inside the
      Unit Disk centered at the origin.

    >>> 0.5 + 0.5*I in c2
    True
    >>> 1 + 2*I in c2
    False

    >>> unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    >>> upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    >>> intersection = unit_disk.intersect(upper_half_unit_disk)
    >>> intersection
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval(0, pi)))
    >>> intersection == upper_half_unit_disk
    True

    See Also
    ========

    CartesianComplexRegion
    PolarComplexRegion
    Complexes

    """
    is_ComplexRegion = True

    def __new__(cls, sets, polar=False):
        if polar is False:
            return CartesianComplexRegion(sets)
        elif polar is True:
            return PolarComplexRegion(sets)
        else:
            raise ValueError("polar should be either True or False")

    @property
    def sets(self):
        """
        Return raw input sets to the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.sets
        ProductSet(Interval(2, 3), Interval(4, 5))
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.sets
        Union(ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        return self.args[0]

    @property
    def psets(self):
        """
        Return a tuple of sets (ProductSets) input of the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)),)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        if self.sets.is_ProductSet:
            psets = ()
            psets = psets + (self.sets, )
        else:
            psets = self.sets.args
        return psets

    @property
    def a_interval(self):
        """
        Return the union of intervals of `x` when, self is in
        rectangular form, or the union of intervals of `r` when
        self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.a_interval
        Interval(2, 3)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.a_interval
        Union(Interval(2, 3), Interval(4, 5))

        """
        a_interval = []
        for element in self.psets:
            a_interval.append(element.args[0])

        a_interval = Union(*a_interval)
        return a_interval

    @property
    def b_interval(self):
        """
        Return the union of intervals of `y` when, self is in
        rectangular form, or the union of intervals of `theta`
        when self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.b_interval
        Interval(4, 5)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.b_interval
        Interval(1, 7)

        """
        b_interval = []
        for element in self.psets:
            b_interval.append(element.args[1])

        b_interval = Union(*b_interval)
        return b_interval

    @property
    def _measure(self):
        """
        The measure of self.sets.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, S
        >>> a, b = Interval(2, 5), Interval(4, 8)
        >>> c = Interval(0, 2*S.Pi)
        >>> c1 = ComplexRegion(a*b)
        >>> c1.measure
        12
        >>> c2 = ComplexRegion(a*c, polar=True)
        >>> c2.measure
        6*pi

        """
        return self.sets._measure

    def _kind(self):
        return self.args[0].kind

    @classmethod
    def from_real(cls, sets):
        """
        Converts given subset of real numbers to a complex region.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion
        >>> unit = Interval(0,1)
        >>> ComplexRegion.from_real(unit)
        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))

        """
        if not sets.is_subset(S.Reals):
            raise ValueError("sets must be a subset of the real line")

        return CartesianComplexRegion(sets * FiniteSet(0))

    def _contains(self, other):
        from sympy.functions import arg, Abs

        isTuple = isinstance(other, Tuple)
        if isTuple and len(other) != 2:
            raise ValueError('expecting Tuple of length 2')

        # If the other is not an Expression, and neither a Tuple
        if not isinstance(other, (Expr, Tuple)):
            return S.false

        # self in rectangular form
        if not self.polar:
            re, im = other if isTuple else other.as_real_imag()
            return tfn[fuzzy_or(fuzzy_and([
                pset.args[0]._contains(re),
                pset.args[1]._contains(im)])
                for pset in self.psets)]

        # self in polar form
        elif self.polar:
            if other.is_zero:
                # ignore undefined complex argument
                return tfn[fuzzy_or(pset.args[0]._contains(S.Zero)
                    for pset in self.psets)]
            if isTuple:
                r, theta = other
            else:
                r, theta = Abs(other), arg(other)
            if theta.is_real and theta.is_number:
                # angles in psets are normalized to [0, 2pi)
                theta %= 2*S.Pi
                return tfn[fuzzy_or(fuzzy_and([
                    pset.args[0]._contains(r),
                    pset.args[1]._contains(theta)])
                    for pset in self.psets)]


class CartesianComplexRegion(ComplexRegion):
    r"""
    Set representing a square region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, I, Interval
    >>> region = ComplexRegion(Interval(1, 3) * Interval(4, 6))
    >>> 2 + 5*I in region
    True
    >>> 5*I in region
    False

    See also
    ========

    ComplexRegion
    PolarComplexRegion
    Complexes
    """

    polar = False
    variables = symbols('x, y', cls=Dummy)

    def __new__(cls, sets):

        if sets == S.Reals*S.Reals:
            return S.Complexes

        if all(_a.is_FiniteSet for _a in sets.args) and (len(sets.args) == 2):

            # ** ProductSet of FiniteSets in the Complex Plane. **
            # For Cases like ComplexRegion({2, 4}*{3}), It
            # would return {2 + 3*I, 4 + 3*I}

            # FIXME: This should probably be handled with something like:
            # return ImageSet(Lambda((x, y), x+I*y), sets).rewrite(FiniteSet)
            complex_num = []
            for x in sets.args[0]:
                for y in sets.args[1]:
                    complex_num.append(x + S.ImaginaryUnit*y)
            return FiniteSet(*complex_num)
        else:
            return Set.__new__(cls, sets)

    @property
    def expr(self):
        x, y = self.variables
        return x + S.ImaginaryUnit*y


class PolarComplexRegion(ComplexRegion):
    r"""
    Set representing a polar region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, oo, pi, I
    >>> rset = Interval(0, oo)
    >>> thetaset = Interval(0, pi)
    >>> upper_half_plane = ComplexRegion(rset * thetaset, polar=True)
    >>> 1 + I in upper_half_plane
    True
    >>> 1 - I in upper_half_plane
    False

    See also
    ========

    ComplexRegion
    CartesianComplexRegion
    Complexes

    """

    polar = True
    variables = symbols('r, theta', cls=Dummy)

    def __new__(cls, sets):

        new_sets = []
        # sets is Union of ProductSets
        if not sets.is_ProductSet:
            for k in sets.args:
                new_sets.append(k)
        # sets is ProductSets
        else:
            new_sets.append(sets)
        # Normalize input theta
        for k, v in enumerate(new_sets):
            new_sets[k] = ProductSet(v.args[0],
                                     normalize_theta_set(v.args[1]))
        sets = Union(*new_sets)
        return Set.__new__(cls, sets)

    @property
    def expr(self):
        r, theta = self.variables
        return r*(cos(theta) + S.ImaginaryUnit*sin(theta))


class Complexes(CartesianComplexRegion, metaclass=Singleton):
    """
    The :class:`Set` of all complex numbers

    Examples
    ========

    >>> from sympy import S, I
    >>> S.Complexes
    Complexes
    >>> 1 + I in S.Complexes
    True

    See also
    ========

    Reals
    ComplexRegion

    """

    is_empty = False
    is_finite_set = False

    # Override property from superclass since Complexes has no args
    @property
    def sets(self):
        return ProductSet(S.Reals, S.Reals)

    def __new__(cls):
        return Set.__new__(cls)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\piecewise.py ===
from sympy.core import S, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.function import DefinedFunction
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
    _canonical, _canonical_coeff)
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name

from itertools import product

Undefined = S.NaN  # Piecewise()

class ExprCondPair(Tuple):
    """Represents an expression, condition pair."""

    def __new__(cls, expr, cond):
        expr = as_Basic(expr)
        if cond == True:
            return Tuple.__new__(cls, expr, true)
        elif cond == False:
            return Tuple.__new__(cls, expr, false)
        elif isinstance(cond, Basic) and cond.has(Piecewise):
            cond = piecewise_fold(cond)
            if isinstance(cond, Piecewise):
                cond = cond.rewrite(ITE)

        if not isinstance(cond, Boolean):
            raise TypeError(filldedent('''
                Second argument must be a Boolean,
                not `%s`''' % func_name(cond)))
        return Tuple.__new__(cls, expr, cond)

    @property
    def expr(self):
        """
        Returns the expression of this pair.
        """
        return self.args[0]

    @property
    def cond(self):
        """
        Returns the condition of this pair.
        """
        return self.args[1]

    @property
    def is_commutative(self):
        return self.expr.is_commutative

    def __iter__(self):
        yield self.expr
        yield self.cond

    def _eval_simplify(self, **kwargs):
        return self.func(*[a.simplify(**kwargs) for a in self.args])


class Piecewise(DefinedFunction):
    """
    Represents a piecewise function.

    Usage:

      Piecewise( (expr,cond), (expr,cond), ... )
        - Each argument is a 2-tuple defining an expression and condition
        - The conds are evaluated in turn returning the first that is True.
          If any of the evaluated conds are not explicitly False,
          e.g. ``x < 1``, the function is returned in symbolic form.
        - If the function is evaluated at a place where all conditions are False,
          nan will be returned.
        - Pairs where the cond is explicitly False, will be removed and no pair
          appearing after a True condition will ever be retained. If a single
          pair with a True condition remains, it will be returned, even when
          evaluation is False.

    Examples
    ========

    >>> from sympy import Piecewise, log, piecewise_fold
    >>> from sympy.abc import x, y
    >>> f = x**2
    >>> g = log(x)
    >>> p = Piecewise((0, x < -1), (f, x <= 1), (g, True))
    >>> p.subs(x,1)
    1
    >>> p.subs(x,5)
    log(5)

    Booleans can contain Piecewise elements:

    >>> cond = (x < y).subs(x, Piecewise((2, x < 0), (3, True))); cond
    Piecewise((2, x < 0), (3, True)) < y

    The folded version of this results in a Piecewise whose
    expressions are Booleans:

    >>> folded_cond = piecewise_fold(cond); folded_cond
    Piecewise((2 < y, x < 0), (3 < y, True))

    When a Boolean containing Piecewise (like cond) or a Piecewise
    with Boolean expressions (like folded_cond) is used as a condition,
    it is converted to an equivalent :class:`~.ITE` object:

    >>> Piecewise((1, folded_cond))
    Piecewise((1, ITE(x < 0, y > 2, y > 3)))

    When a condition is an ``ITE``, it will be converted to a simplified
    Boolean expression:

    >>> piecewise_fold(_)
    Piecewise((1, ((x >= 0) | (y > 2)) & ((y > 3) | (x < 0))))

    See Also
    ========

    piecewise_fold
    piecewise_exclusive
    ITE
    """

    nargs = None
    is_Piecewise = True

    def __new__(cls, *args, **options):
        if len(args) == 0:
            raise TypeError("At least one (expr, cond) pair expected.")
        # (Try to) sympify args first
        newargs = []
        for ec in args:
            # ec could be a ExprCondPair or a tuple
            pair = ExprCondPair(*getattr(ec, 'args', ec))
            cond = pair.cond
            if cond is false:
                continue
            newargs.append(pair)
            if cond is true:
                break

        eval = options.pop('evaluate', global_parameters.evaluate)
        if eval:
            r = cls.eval(*newargs)
            if r is not None:
                return r
        elif len(newargs) == 1 and newargs[0].cond == True:
            return newargs[0].expr

        return Basic.__new__(cls, *newargs, **options)

    @classmethod
    def eval(cls, *_args):
        """Either return a modified version of the args or, if no
        modifications were made, return None.

        Modifications that are made here:

        1. relationals are made canonical
        2. any False conditions are dropped
        3. any repeat of a previous condition is ignored
        4. any args past one with a true condition are dropped

        If there are no args left, nan will be returned.
        If there is a single arg with a True condition, its
        corresponding expression will be returned.

        EXAMPLES
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> cond = -x < -1
        >>> args = [(1, cond), (4, cond), (3, False), (2, True), (5, x < 1)]
        >>> Piecewise(*args, evaluate=False)
        Piecewise((1, -x < -1), (4, -x < -1), (2, True))
        >>> Piecewise(*args)
        Piecewise((1, x > 1), (2, True))
        """
        if not _args:
            return Undefined

        if len(_args) == 1 and _args[0][-1] == True:
            return _args[0][0]

        newargs = _piecewise_collapse_arguments(_args)

        # some conditions may have been redundant
        missing = len(newargs) != len(_args)
        # some conditions may have changed
        same = all(a == b for a, b in zip(newargs, _args))
        # if either change happened we return the expr with the
        # updated args
        if not newargs:
            raise ValueError(filldedent('''
                There are no conditions (or none that
                are not trivially false) to define an
                expression.'''))
        if missing or not same:
            return cls(*newargs)

    def doit(self, **hints):
        """
        Evaluate this piecewise function.
        """
        newargs = []
        for e, c in self.args:
            if hints.get('deep', True):
                if isinstance(e, Basic):
                    newe = e.doit(**hints)
                    if newe != self:
                        e = newe
                if isinstance(c, Basic):
                    c = c.doit(**hints)
            newargs.append((e, c))
        return self.func(*newargs)

    def _eval_simplify(self, **kwargs):
        return piecewise_simplify(self, **kwargs)

    def _eval_as_leading_term(self, x, logx, cdir):
        for e, c in self.args:
            if c == True or c.subs(x, 0) == True:
                return e.as_leading_term(x)

    def _eval_adjoint(self):
        return self.func(*[(e.adjoint(), c) for e, c in self.args])

    def _eval_conjugate(self):
        return self.func(*[(e.conjugate(), c) for e, c in self.args])

    def _eval_derivative(self, x):
        return self.func(*[(diff(e, x), c) for e, c in self.args])

    def _eval_evalf(self, prec):
        return self.func(*[(e._evalf(prec), c) for e, c in self.args])

    def _eval_is_meromorphic(self, x, a):
        # Conditions often implicitly assume that the argument is real.
        # Hence, there needs to be some check for as_set.
        if not a.is_real:
            return None

        # Then, scan ExprCondPairs in the given order to find a piece that would contain a,
        # possibly as a boundary point.
        for e, c in self.args:
            cond = c.subs(x, a)

            if cond.is_Relational:
                return None
            if a in c.as_set().boundary:
                return None
            # Apply expression if a is an interior point of the domain of e.
            if cond:
                return e._eval_is_meromorphic(x, a)

    def piecewise_integrate(self, x, **kwargs):
        """Return the Piecewise with each expression being
        replaced with its antiderivative. To obtain a continuous
        antiderivative, use the :func:`~.integrate` function or method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        Note that this does not give a continuous function, e.g.
        at x = 1 the 3rd condition applies and the antiderivative
        there is 2*x so the value of the antiderivative is 2:

        >>> anti = _
        >>> anti.subs(x, 1)
        2

        The continuous derivative accounts for the integral *up to*
        the point of interest, however:

        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> _.subs(x, 1)
        1

        See Also
        ========
        Piecewise._eval_integral
        """
        from sympy.integrals import integrate
        return self.func(*[(integrate(e, x, **kwargs), c) for e, c in self.args])

    def _handle_irel(self, x, handler):
        """Return either None (if the conditions of self depend only on x) else
        a Piecewise expression whose expressions (handled by the handler that
        was passed) are paired with the governing x-independent relationals,
        e.g. Piecewise((A, a(x) & b(y)), (B, c(x) | c(y)) ->
        Piecewise(
            (handler(Piecewise((A, a(x) & True), (B, c(x) | True)), b(y) & c(y)),
            (handler(Piecewise((A, a(x) & True), (B, c(x) | False)), b(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | True)), c(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | False)), True))
        """
        # identify governing relationals
        rel = self.atoms(Relational)
        irel = list(ordered([r for r in rel if x not in r.free_symbols
            and r not in (S.true, S.false)]))
        if irel:
            args = {}
            exprinorder = []
            for truth in product((1, 0), repeat=len(irel)):
                reps = dict(zip(irel, truth))
                # only store the true conditions since the false are implied
                # when they appear lower in the Piecewise args
                if 1 not in truth:
                    cond = None  # flag this one so it doesn't get combined
                else:
                    andargs = Tuple(*[i for i in reps if reps[i]])
                    free = list(andargs.free_symbols)
                    if len(free) == 1:
                        from sympy.solvers.inequalities import (
                            reduce_inequalities, _solve_inequality)
                        try:
                            t = reduce_inequalities(andargs, free[0])
                            # ValueError when there are potentially
                            # nonvanishing imaginary parts
                        except (ValueError, NotImplementedError):
                            # at least isolate free symbol on left
                            t = And(*[_solve_inequality(
                                a, free[0], linear=True)
                                for a in andargs])
                    else:
                        t = And(*andargs)
                    if t is S.false:
                        continue  # an impossible combination
                    cond = t
                expr = handler(self.xreplace(reps))
                if isinstance(expr, self.func) and len(expr.args) == 1:
                    expr, econd = expr.args[0]
                    cond = And(econd, True if cond is None else cond)
                # the ec pairs are being collected since all possibilities
                # are being enumerated, but don't put the last one in since
                # its expr might match a previous expression and it
                # must appear last in the args
                if cond is not None:
                    args.setdefault(expr, []).append(cond)
                    # but since we only store the true conditions we must maintain
                    # the order so that the expression with the most true values
                    # comes first
                    exprinorder.append(expr)
            # convert collected conditions as args of Or
            for k in args:
                args[k] = Or(*args[k])
            # take them in the order obtained
            args = [(e, args[e]) for e in uniq(exprinorder)]
            # add in the last arg
            args.append((expr, True))
            return Piecewise(*args)

    def _eval_integral(self, x, _first=True, **kwargs):
        """Return the indefinite integral of the
        Piecewise such that subsequent substitution of x with a
        value will give the value of the integral (not including
        the constant of integration) up to that point. To only
        integrate the individual parts of Piecewise, use the
        ``piecewise_integrate`` method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        See Also
        ========
        Piecewise.piecewise_integrate
        """
        from sympy.integrals.integrals import integrate

        if _first:
            def handler(ipw):
                if isinstance(ipw, self.func):
                    return ipw._eval_integral(x, _first=False, **kwargs)
                else:
                    return ipw.integrate(x, **kwargs)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv

        # handle a Piecewise from -oo to oo with and no x-independent relationals
        # -----------------------------------------------------------------------
        ok, abei = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            return Integral(self, x)  # unevaluated

        pieces = [(a, b) for a, b, _, _ in abei]
        oo = S.Infinity
        done = [(-oo, oo, -1)]
        for k, p in enumerate(pieces):
            if p == (-oo, oo):
                # all undone intervals will get this key
                for j, (a, b, i) in enumerate(done):
                    if i == -1:
                        done[j] = a, b, k
                break  # nothing else to consider
            N = len(done) - 1
            for j, (a, b, i) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j: j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for a, b, i in done if a != b]

        # append an arg if there is a hole so a reference to
        # argument -1 will give Undefined
        if any(i == -1 for (a, b, i) in done):
            abei.append((-oo, oo, Undefined, -1))

        # return the sum of the intervals
        args = []
        sum = None
        for a, b, i in done:
            anti = integrate(abei[i][-2], x, **kwargs)
            if sum is None:
                sum = anti
            else:
                sum = sum.subs(x, a)
                e = anti._eval_interval(x, a, x)
                if sum.has(*_illegal) or e.has(*_illegal):
                    sum = anti
                else:
                    sum += e
            # see if we know whether b is contained in original
            # condition
            if b is S.Infinity:
                cond = True
            elif self.args[abei[i][-1]].cond.subs(x, b) == False:
                cond = (x < b)
            else:
                cond = (x <= b)
            args.append((sum, cond))
        return Piecewise(*args)

    def _eval_interval(self, sym, a, b, _first=True):
        """Evaluates the function along the sym in a given interval [a, b]"""
        # FIXME: Currently complex intervals are not supported.  A possible
        # replacement algorithm, discussed in issue 5227, can be found in the
        # following papers;
        #     http://portal.acm.org/citation.cfm?id=281649
        #     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.4127&rep=rep1&type=pdf

        if a is None or b is None:
            # In this case, it is just simple substitution
            return super()._eval_interval(sym, a, b)
        else:
            x, lo, hi = map(as_Basic, (sym, a, b))

        if _first:  # get only x-dependent relationals
            def handler(ipw):
                if isinstance(ipw, self.func):
                    return ipw._eval_interval(x, lo, hi, _first=None)
                else:
                    return ipw._eval_interval(x, lo, hi)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv

            if (lo < hi) is S.false or (
                    lo is S.Infinity or hi is S.NegativeInfinity):
                rv = self._eval_interval(x, hi, lo, _first=False)
                if isinstance(rv, Piecewise):
                    rv = Piecewise(*[(-e, c) for e, c in rv.args])
                else:
                    rv = -rv
                return rv

            if (lo < hi) is S.true or (
                    hi is S.Infinity or lo is S.NegativeInfinity):
                pass
            else:
                _a = Dummy('lo')
                _b = Dummy('hi')
                a = lo if lo.is_comparable else _a
                b = hi if hi.is_comparable else _b
                pos = self._eval_interval(x, a, b, _first=False)
                if a == _a and b == _b:
                    # it's purely symbolic so just swap lo and hi and
                    # change the sign to get the value for when lo > hi
                    neg, pos = (-pos.xreplace({_a: hi, _b: lo}),
                        pos.xreplace({_a: lo, _b: hi}))
                else:
                    # at least one of the bounds was comparable, so allow
                    # _eval_interval to use that information when computing
                    # the interval with lo and hi reversed
                    neg, pos = (-self._eval_interval(x, hi, lo, _first=False),
                        pos.xreplace({_a: lo, _b: hi}))

                # allow simplification based on ordering of lo and hi
                p = Dummy('', positive=True)
                if lo.is_Symbol:
                    pos = pos.xreplace({lo: hi - p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({lo: hi + p}).xreplace({p: lo - hi})
                elif hi.is_Symbol:
                    pos = pos.xreplace({hi: lo + p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({hi: lo - p}).xreplace({p: lo - hi})
                # evaluate limits that may have unevaluate Min/Max
                touch = lambda _: _.replace(
                    lambda x: isinstance(x, (Min, Max)),
                    lambda x: x.func(*x.args))
                neg = touch(neg)
                pos = touch(pos)
                # assemble return expression; make the first condition be Lt
                # b/c then the first expression will look the same whether
                # the lo or hi limit is symbolic
                if a == _a:  # the lower limit was symbolic
                    rv = Piecewise(
                        (pos,
                            lo < hi),
                        (neg,
                            True))
                else:
                    rv = Piecewise(
                        (neg,
                            hi < lo),
                        (pos,
                            True))

                if rv == Undefined:
                    raise ValueError("Can't integrate across undefined region.")
                if any(isinstance(i, Piecewise) for i in (pos, neg)):
                    rv = piecewise_fold(rv)
                return rv

        # handle a Piecewise with lo <= hi and no x-independent relationals
        # -----------------------------------------------------------------
        ok, abei = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            # not being able to do the interval of f(x) can
            # be stated as not being able to do the integral
            # of f'(x) over the same range
            return Integral(self.diff(x), (x, lo, hi))  # unevaluated

        pieces = [(a, b) for a, b, _, _ in abei]
        done = [(lo, hi, -1)]
        oo = S.Infinity
        for k, p in enumerate(pieces):
            if p[:2] == (-oo, oo):
                # all undone intervals will get this key
                for j, (a, b, i) in enumerate(done):
                    if i == -1:
                        done[j] = a, b, k
                break  # nothing else to consider
            N = len(done) - 1
            for j, (a, b, i) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j: j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for a, b, i in done if a != b]

        # return the sum of the intervals
        sum = S.Zero
        upto = None
        for a, b, i in done:
            if i == -1:
                if upto is None:
                    return Undefined
                # TODO simplify hi <= upto
                return Piecewise((sum, hi <= upto), (Undefined, True))
            sum += abei[i][-2]._eval_interval(x, a, b)
            upto = b
        return sum

    def _intervals(self, sym, err_on_Eq=False):
        r"""Return a bool and a message (when bool is False), else a
        list of unique tuples, (a, b, e, i), where a and b
        are the lower and upper bounds in which the expression e of
        argument i in self is defined and $a < b$ (when involving
        numbers) or $a \le b$ when involving symbols.

        If there are any relationals not involving sym, or any
        relational cannot be solved for sym, the bool will be False
        a message be given as the second return value. The calling
        routine should have removed such relationals before calling
        this routine.

        The evaluated conditions will be returned as ranges.
        Discontinuous ranges will be returned separately with
        identical expressions. The first condition that evaluates to
        True will be returned as the last tuple with a, b = -oo, oo.
        """
        from sympy.solvers.inequalities import _solve_inequality

        assert isinstance(self, Piecewise)

        def nonsymfail(cond):
            return False, filldedent('''
                A condition not involving
                %s appeared: %s''' % (sym, cond))

        def _solve_relational(r):
            if sym not in r.free_symbols:
                return nonsymfail(r)
            try:
                rv = _solve_inequality(r, sym)
            except NotImplementedError:
                return False, 'Unable to solve relational %s for %s.' % (r, sym)
            if isinstance(rv, Relational):
                free = rv.args[1].free_symbols
                if rv.args[0] != sym or sym in free:
                    return False, 'Unable to solve relational %s for %s.' % (r, sym)
                if rv.rel_op == '==':
                    # this equality has been affirmed to have the form
                    # Eq(sym, rhs) where rhs is sym-free; it represents
                    # a zero-width interval which will be ignored
                    # whether it is an isolated condition or contained
                    # within an And or an Or
                    rv = S.false
                elif rv.rel_op == '!=':
                    try:
                        rv = Or(sym < rv.rhs, sym > rv.rhs)
                    except TypeError:
                        # e.g. x != I ==> all real x satisfy
                        rv = S.true
            elif rv == (S.NegativeInfinity < sym) & (sym < S.Infinity):
                rv = S.true
            return True, rv

        args = list(self.args)
        # make self canonical wrt Relationals
        keys = self.atoms(Relational)
        reps = {}
        for r in keys:
            ok, s = _solve_relational(r)
            if ok != True:
                return False, ok
            reps[r] = s
        # process args individually so if any evaluate, their position
        # in the original Piecewise will be known
        args = [i.xreplace(reps) for i in self.args]

        # precondition args
        expr_cond = []
        default = idefault = None
        for i, (expr, cond) in enumerate(args):
            if cond is S.false:
                continue
            if cond is S.true:
                default = expr
                idefault = i
                break
            if isinstance(cond, Eq):
                # unanticipated condition, but it is here in case a
                # replacement caused an Eq to appear
                if err_on_Eq:
                    return False, 'encountered Eq condition: %s' % cond
                continue  # zero width interval

            cond = to_cnf(cond)
            if isinstance(cond, And):
                cond = distribute_or_over_and(cond)

            if isinstance(cond, Or):
                expr_cond.extend(
                    [(i, expr, o) for o in cond.args
                    if not isinstance(o, Eq)])
            elif cond is not S.false:
                expr_cond.append((i, expr, cond))
            elif cond is S.true:
                default = expr
                idefault = i
                break

        # determine intervals represented by conditions
        int_expr = []
        for iarg, expr, cond in expr_cond:
            if isinstance(cond, And):
                lower = S.NegativeInfinity
                upper = S.Infinity
                exclude = []
                for cond2 in cond.args:
                    if not isinstance(cond2, Relational):
                        return False, 'expecting only Relationals'
                    if isinstance(cond2, Eq):
                        lower = upper  # ignore
                        if err_on_Eq:
                            return False, 'encountered secondary Eq condition'
                        break
                    elif isinstance(cond2, Ne):
                        l, r = cond2.args
                        if l == sym:
                            exclude.append(r)
                        elif r == sym:
                            exclude.append(l)
                        else:
                            return nonsymfail(cond2)
                        continue
                    elif cond2.lts == sym:
                        upper = Min(cond2.gts, upper)
                    elif cond2.gts == sym:
                        lower = Max(cond2.lts, lower)
                    else:
                        return nonsymfail(cond2)  # should never get here
                if exclude:
                    exclude = list(ordered(exclude))
                    newcond = []
                    for i, e in enumerate(exclude):
                        if e < lower == True or e > upper == True:
                            continue
                        if not newcond:
                            newcond.append((None, lower))  # add a primer
                        newcond.append((newcond[-1][1], e))
                    newcond.append((newcond[-1][1], upper))
                    newcond.pop(0)  # remove the primer
                    expr_cond.extend([(iarg, expr, And(i[0] < sym, sym < i[1])) for i in newcond])
                    continue
            elif isinstance(cond, Relational) and cond.rel_op != '!=':
                lower, upper = cond.lts, cond.gts  # part 1: initialize with givens
                if cond.lts == sym:                # part 1a: expand the side ...
                    lower = S.NegativeInfinity   # e.g. x <= 0 ---> -oo <= 0
                elif cond.gts == sym:            # part 1a: ... that can be expanded
                    upper = S.Infinity           # e.g. x >= 0 --->  oo >= 0
                else:
                    return nonsymfail(cond)
            else:
                return False, 'unrecognized condition: %s' % cond

            upper = Max(lower, upper)
            if err_on_Eq and lower == upper:
                return False, 'encountered Eq condition'
            if (lower >= upper) is not S.true:
                int_expr.append((lower, upper, expr, iarg))

        if default is not None:
            int_expr.append(
                (S.NegativeInfinity, S.Infinity, default, idefault))

        return True, list(uniq(int_expr))

    def _eval_nseries(self, x, n, logx, cdir=0):
        args = [(ec.expr._eval_nseries(x, n, logx), ec.cond) for ec in self.args]
        return self.func(*args)

    def _eval_power(self, s):
        return self.func(*[(e**s, c) for e, c in self.args])

    def _eval_subs(self, old, new):
        # this is strictly not necessary, but we can keep track
        # of whether True or False conditions arise and be
        # somewhat more efficient by avoiding other substitutions
        # and avoiding invalid conditions that appear after a
        # True condition
        args = list(self.args)
        args_exist = False
        for i, (e, c) in enumerate(args):
            c = c._subs(old, new)
            if c != False:
                args_exist = True
                e = e._subs(old, new)
            args[i] = (e, c)
            if c == True:
                break
        if not args_exist:
            args = ((Undefined, True),)
        return self.func(*args)

    def _eval_transpose(self):
        return self.func(*[(e.transpose(), c) for e, c in self.args])

    def _eval_template_is_attr(self, is_attr):
        b = None
        for expr, _ in self.args:
            a = getattr(expr, is_attr)
            if a is None:
                return
            if b is None:
                b = a
            elif b is not a:
                return
        return b

    _eval_is_finite = lambda self: self._eval_template_is_attr(
        'is_finite')
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    _eval_is_imaginary = lambda self: self._eval_template_is_attr(
        'is_imaginary')
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    _eval_is_irrational = lambda self: self._eval_template_is_attr(
        'is_irrational')
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr(
        'is_nonnegative')
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr(
        'is_nonpositive')
    _eval_is_nonzero = lambda self: self._eval_template_is_attr(
        'is_nonzero')
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    _eval_is_extended_real = lambda self: self._eval_template_is_attr(
            'is_extended_real')
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr(
            'is_extended_positive')
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr(
            'is_extended_negative')
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr(
            'is_extended_nonzero')
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr(
            'is_extended_nonpositive')
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr(
            'is_extended_nonnegative')
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    _eval_is_zero = lambda self: self._eval_template_is_attr(
        'is_zero')

    @classmethod
    def __eval_cond(cls, cond):
        """Return the truth value of the condition."""
        if cond == True:
            return True
        if isinstance(cond, Eq):
            try:
                diff = cond.lhs - cond.rhs
                if diff.is_commutative:
                    return diff.is_zero
            except TypeError:
                pass

    def as_expr_set_pairs(self, domain=None):
        """Return tuples for each argument of self that give
        the expression and the interval in which it is valid
        which is contained within the given domain.
        If a condition cannot be converted to a set, an error
        will be raised. The variable of the conditions is
        assumed to be real; sets of real values are returned.

        Examples
        ========

        >>> from sympy import Piecewise, Interval
        >>> from sympy.abc import x
        >>> p = Piecewise(
        ...     (1, x < 2),
        ...     (2,(x > 0) & (x < 4)),
        ...     (3, True))
        >>> p.as_expr_set_pairs()
        [(1, Interval.open(-oo, 2)),
         (2, Interval.Ropen(2, 4)),
         (3, Interval(4, oo))]
        >>> p.as_expr_set_pairs(Interval(0, 3))
        [(1, Interval.Ropen(0, 2)),
         (2, Interval(2, 3))]
        """
        if domain is None:
            domain = S.Reals
        exp_sets = []
        U = domain
        complex = not domain.is_subset(S.Reals)
        cond_free = set()
        for expr, cond in self.args:
            cond_free |= cond.free_symbols
            if len(cond_free) > 1:
                raise NotImplementedError(filldedent('''
                    multivariate conditions are not handled.'''))
            if complex:
                for i in cond.atoms(Relational):
                    if not isinstance(i, (Eq, Ne)):
                        raise ValueError(filldedent('''
                            Inequalities in the complex domain are
                            not supported. Try the real domain by
                            setting domain=S.Reals'''))
            cond_int = U.intersect(cond.as_set())
            U = U - cond_int
            if cond_int != S.EmptySet:
                exp_sets.append((expr, cond_int))
        return exp_sets

    def _eval_rewrite_as_ITE(self, *args, **kwargs):
        byfree = {}
        args = list(args)
        default = any(c == True for b, c in args)
        for i, (b, c) in enumerate(args):
            if not isinstance(b, Boolean) and b != True:
                raise TypeError(filldedent('''
                    Expecting Boolean or bool but got `%s`
                    ''' % func_name(b)))
            if c == True:
                break
            # loop over independent conditions for this b
            for c in c.args if isinstance(c, Or) else [c]:
                free = c.free_symbols
                x = free.pop()
                try:
                    byfree[x] = byfree.setdefault(
                        x, S.EmptySet).union(c.as_set())
                except NotImplementedError:
                    if not default:
                        raise NotImplementedError(filldedent('''
                            A method to determine whether a multivariate
                            conditional is consistent with a complete coverage
                            of all variables has not been implemented so the
                            rewrite is being stopped after encountering `%s`.
                            This error would not occur if a default expression
                            like `(foo, True)` were given.
                            ''' % c))
                if byfree[x] in (S.UniversalSet, S.Reals):
                    # collapse the ith condition to True and break
                    args[i] = list(args[i])
                    c = args[i][1] = True
                    break
            if c == True:
                break
        if c != True:
            raise ValueError(filldedent('''
                Conditions must cover all reals or a final default
                condition `(foo, True)` must be given.
                '''))
        last, _ = args[i]  # ignore all past ith arg
        for a, c in reversed(args[:i]):
            last = ITE(c, a, last)
        return _canonical(last)

    def _eval_rewrite_as_KroneckerDelta(self, *args, **kwargs):
        from sympy.functions.special.tensor_functions import KroneckerDelta

        rules = {
            And: [False, False],
            Or: [True, True],
            Not: [True, False],
            Eq: [None, None],
            Ne: [None, None]
        }

        class UnrecognizedCondition(Exception):
            pass

        def rewrite(cond):
            if isinstance(cond, Eq):
                return KroneckerDelta(*cond.args)
            if isinstance(cond, Ne):
                return 1 - KroneckerDelta(*cond.args)

            cls, args = type(cond), cond.args
            if cls not in rules:
                raise UnrecognizedCondition(cls)

            b1, b2 = rules[cls]
            k = Mul(*[1 - rewrite(c) for c in args]) if b1 else Mul(*[rewrite(c) for c in args])

            if b2:
                return 1 - k
            return k

        conditions = []
        true_value = None
        for value, cond in args:
            if type(cond) in rules:
                conditions.append((value, cond))
            elif cond is S.true:
                if true_value is None:
                    true_value = value
            else:
                return

        if true_value is not None:
            result = true_value

            for value, cond in conditions[::-1]:
                try:
                    k = rewrite(cond)
                    result = k * value + (1 - k) * result
                except UnrecognizedCondition:
                    return

            return result


def piecewise_fold(expr, evaluate=True):
    """
    Takes an expression containing a piecewise function and returns the
    expression in piecewise form. In addition, any ITE conditions are
    rewritten in negation normal form and simplified.

    The final Piecewise is evaluated (default) but if the raw form
    is desired, send ``evaluate=False``; if trivial evaluation is
    desired, send ``evaluate=None`` and duplicate conditions and
    processing of True and False will be handled.

    Examples
    ========

    >>> from sympy import Piecewise, piecewise_fold, S
    >>> from sympy.abc import x
    >>> p = Piecewise((x, x < 1), (1, S(1) <= x))
    >>> piecewise_fold(x*p)
    Piecewise((x**2, x < 1), (x, True))

    See Also
    ========

    Piecewise
    piecewise_exclusive
    """
    if not isinstance(expr, Basic) or not expr.has(Piecewise):
        return expr

    new_args = []
    if isinstance(expr, (ExprCondPair, Piecewise)):
        for e, c in expr.args:
            if not isinstance(e, Piecewise):
                e = piecewise_fold(e)
            # we don't keep Piecewise in condition because
            # it has to be checked to see that it's complete
            # and we convert it to ITE at that time
            assert not c.has(Piecewise)  # pragma: no cover
            if isinstance(c, ITE):
                c = c.to_nnf()
                c = simplify_logic(c, form='cnf')
            if isinstance(e, Piecewise):
                new_args.extend([(piecewise_fold(ei), And(ci, c))
                    for ei, ci in e.args])
            else:
                new_args.append((e, c))
    else:
        # Given
        #     P1 = Piecewise((e11, c1), (e12, c2), A)
        #     P2 = Piecewise((e21, c1), (e22, c2), B)
        #     ...
        # the folding of f(P1, P2) is trivially
        # Piecewise(
        #   (f(e11, e21), c1),
        #   (f(e12, e22), c2),
        #   (f(Piecewise(A), Piecewise(B)), True))
        # Certain objects end up rewriting themselves as thus, so
        # we do that grouping before the more generic folding.
        # The following applies this idea when f = Add or f = Mul
        # (and the expression is commutative).
        if expr.is_Add or expr.is_Mul and expr.is_commutative:
            p, args = sift(expr.args, lambda x: x.is_Piecewise, binary=True)
            pc = sift(p, lambda x: tuple([c for e,c in x.args]))
            for c in list(ordered(pc)):
                if len(pc[c]) > 1:
                    pargs = [list(i.args) for i in pc[c]]
                    # the first one is the same; there may be more
                    com = common_prefix(*[
                        [i.cond for i in j] for j in pargs])
                    n = len(com)
                    collected = []
                    for i in range(n):
                        collected.append((
                            expr.func(*[ai[i].expr for ai in pargs]),
                            com[i]))
                    remains = []
                    for a in pargs:
                        if n == len(a):  # no more args
                            continue
                        if a[n].cond == True:  # no longer Piecewise
                            remains.append(a[n].expr)
                        else:  # restore the remaining Piecewise
                            remains.append(
                                Piecewise(*a[n:], evaluate=False))
                    if remains:
                        collected.append((expr.func(*remains), True))
                    args.append(Piecewise(*collected, evaluate=False))
                    continue
                args.extend(pc[c])
        else:
            args = expr.args
        # fold
        folded = list(map(piecewise_fold, args))
        for ec in product(*[
                (i.args if isinstance(i, Piecewise) else
                 [(i, true)]) for i in folded]):
            e, c = zip(*ec)
            new_args.append((expr.func(*e), And(*c)))

    if evaluate is None:
        # don't return duplicate conditions, otherwise don't evaluate
        new_args = list(reversed([(e, c) for c, e in {
            c: e for e, c in reversed(new_args)}.items()]))
    rv = Piecewise(*new_args, evaluate=evaluate)
    if evaluate is None and len(rv.args) == 1 and rv.args[0].cond == True:
        return rv.args[0].expr
    if any(s.expr.has(Piecewise) for p in rv.atoms(Piecewise) for s in p.args):
        return piecewise_fold(rv)
    return rv


def _clip(A, B, k):
    """Return interval B as intervals that are covered by A (keyed
    to k) and all other intervals of B not covered by A keyed to -1.

    The reference point of each interval is the rhs; if the lhs is
    greater than the rhs then an interval of zero width interval will
    result, e.g. (4, 1) is treated like (1, 1).

    Examples
    ========

    >>> from sympy.functions.elementary.piecewise import _clip
    >>> from sympy import Tuple
    >>> A = Tuple(1, 3)
    >>> B = Tuple(2, 4)
    >>> _clip(A, B, 0)
    [(2, 3, 0), (3, 4, -1)]

    Interpretation: interval portion (2, 3) of interval (2, 4) is
    covered by interval (1, 3) and is keyed to 0 as requested;
    interval (3, 4) was not covered by (1, 3) and is keyed to -1.
    """
    a, b = B
    c, d = A
    c, d = Min(Max(c, a), b), Min(Max(d, a), b)
    a = Min(a, b)
    p = []
    if a != c:
        p.append((a, c, -1))
    else:
        pass
    if c != d:
        p.append((c, d, k))
    else:
        pass
    if b != d:
        if d == c and p and p[-1][-1] == -1:
            p[-1] = p[-1][0], b, -1
        else:
            p.append((d, b, -1))
    else:
        pass

    return p


def piecewise_simplify_arguments(expr, **kwargs):
    from sympy.simplify.simplify import simplify

    # simplify conditions
    f1 = expr.args[0].cond.free_symbols
    args = None
    if len(f1) == 1 and not expr.atoms(Eq):
        x = f1.pop()
        # this won't return intervals involving Eq
        # and it won't handle symbols treated as
        # booleans
        ok, abe_ = expr._intervals(x, err_on_Eq=True)
        def include(c, x, a):
            "return True if c.subs(x, a) is True, else False"
            try:
                return c.subs(x, a) == True
            except TypeError:
                return False
        if ok:
            args = []
            covered = S.EmptySet
            from sympy.sets.sets import Interval
            for a, b, e, i in abe_:
                c = expr.args[i].cond
                incl_a = include(c, x, a)
                incl_b = include(c, x, b)
                iv = Interval(a, b, not incl_a, not incl_b)
                cset = iv - covered
                if not cset:
                    continue
                try:
                    a = cset.inf
                except NotImplementedError:
                    pass # continue with the given `a`
                else:
                    incl_a = include(c, x, a)
                if incl_a and incl_b:
                    if a.is_infinite and b.is_infinite:
                        c = S.true
                    elif b.is_infinite:
                        c = (x > a) if a in covered else (x >= a)
                    elif a.is_infinite:
                        c = (x <= b)
                    elif a in covered:
                        c = And(a < x, x <= b)
                    else:
                        c = And(a <= x, x <= b)
                elif incl_a:
                    if a.is_infinite:
                        c = (x < b)
                    elif a in covered:
                        c = And(a < x, x < b)
                    else:
                        c = And(a <= x, x < b)
                elif incl_b:
                    if b.is_infinite:
                        c = (x > a)
                    else:
                        c = And(a < x, x <= b)
                else:
                    if a in covered:
                        c = (x < b)
                    else:
                        c = And(a < x, x < b)
                covered |= iv
                if a is S.NegativeInfinity and incl_a:
                    covered |= {S.NegativeInfinity}
                if b is S.Infinity and incl_b:
                    covered |= {S.Infinity}
                args.append((e, c))
            if not S.Reals.is_subset(covered):
                args.append((Undefined, True))
    if args is None:
        args = list(expr.args)
        for i in range(len(args)):
            e, c  = args[i]
            if isinstance(c, Basic):
                c = simplify(c, **kwargs)
            args[i] = (e, c)

    # simplify expressions
    doit = kwargs.pop('doit', None)
    for i in range(len(args)):
        e, c  = args[i]
        if isinstance(e, Basic):
            # Skip doit to avoid growth at every call for some integrals
            # and sums, see sympy/sympy#17165
            newe = simplify(e, doit=False, **kwargs)
            if newe != e:
                e = newe
        args[i] = (e, c)

    # restore kwargs flag
    if doit is not None:
        kwargs['doit'] = doit

    return Piecewise(*args)


def _piecewise_collapse_arguments(_args):
    newargs = []  # the unevaluated conditions
    current_cond = set()  # the conditions up to a given e, c pair
    for expr, cond in _args:
        cond = cond.replace(
            lambda _: _.is_Relational, _canonical_coeff)
        # Check here if expr is a Piecewise and collapse if one of
        # the conds in expr matches cond. This allows the collapsing
        # of Piecewise((Piecewise((x,x<0)),x<0)) to Piecewise((x,x<0)).
        # This is important when using piecewise_fold to simplify
        # multiple Piecewise instances having the same conds.
        # Eventually, this code should be able to collapse Piecewise's
        # having different intervals, but this will probably require
        # using the new assumptions.
        if isinstance(expr, Piecewise):
            unmatching = []
            for i, (e, c) in enumerate(expr.args):
                if c in current_cond:
                    # this would already have triggered
                    continue
                if c == cond:
                    if c != True:
                        # nothing past this condition will ever
                        # trigger and only those args before this
                        # that didn't match a previous condition
                        # could possibly trigger
                        if unmatching:
                            expr = Piecewise(*(
                                unmatching + [(e, c)]))
                        else:
                            expr = e
                    break
                else:
                    unmatching.append((e, c))

        # check for condition repeats
        got = False
        # -- if an And contains a condition that was
        #    already encountered, then the And will be
        #    False: if the previous condition was False
        #    then the And will be False and if the previous
        #    condition is True then then we wouldn't get to
        #    this point. In either case, we can skip this condition.
        for i in ([cond] +
                  (list(cond.args) if isinstance(cond, And) else
                  [])):
            if i in current_cond:
                got = True
                break
        if got:
            continue

        # -- if not(c) is already in current_cond then c is
        #    a redundant condition in an And. This does not
        #    apply to Or, however: (e1, c), (e2, Or(~c, d))
        #    is not (e1, c), (e2, d) because if c and d are
        #    both False this would give no results when the
        #    true answer should be (e2, True)
        if isinstance(cond, And):
            nonredundant = []
            for c in cond.args:
                if isinstance(c, Relational):
                    if c.negated.canonical in current_cond:
                        continue
                    # if a strict inequality appears after
                    # a non-strict one, then the condition is
                    # redundant
                    if isinstance(c, (Lt, Gt)) and (
                        c.weak in current_cond):
                        cond = False
                        break
                nonredundant.append(c)
            else:
                cond = cond.func(*nonredundant)
        elif isinstance(cond, Relational):
            if cond.negated.canonical in current_cond:
                cond = S.true

        current_cond.add(cond)

        # collect successive e,c pairs when exprs or cond match
        if newargs:
            if newargs[-1].expr == expr:
                orcond = Or(cond, newargs[-1].cond)
                if isinstance(orcond, (And, Or)):
                    orcond = distribute_and_over_or(orcond)
                newargs[-1] = ExprCondPair(expr, orcond)
                continue
            elif newargs[-1].cond == cond:
                continue
        newargs.append(ExprCondPair(expr, cond))
    return newargs


_blessed = lambda e: getattr(e.lhs, '_diff_wrt', False) and (
    getattr(e.rhs, '_diff_wrt', None) or
    isinstance(e.rhs, (Rational, NumberSymbol)))


def piecewise_simplify(expr, **kwargs):
    expr = piecewise_simplify_arguments(expr, **kwargs)
    if not isinstance(expr, Piecewise):
        return expr
    args = list(expr.args)

    args = _piecewise_simplify_eq_and(args)
    args = _piecewise_simplify_equal_to_next_segment(args)
    return Piecewise(*args)


def _piecewise_simplify_equal_to_next_segment(args):
    """
    See if expressions valid for an Equal expression happens to evaluate
    to the same function as in the next piecewise segment, see:
    https://github.com/sympy/sympy/issues/8458
    """
    prevexpr = None
    for i, (expr, cond) in reversed(list(enumerate(args))):
        if prevexpr is not None:
            if isinstance(cond, And):
                eqs, other = sift(cond.args,
                                  lambda i: isinstance(i, Eq), binary=True)
            elif isinstance(cond, Eq):
                eqs, other = [cond], []
            else:
                eqs = other = []
            _prevexpr = prevexpr
            _expr = expr
            if eqs and not other:
                eqs = list(ordered(eqs))
                for e in eqs:
                    # allow 2 args to collapse into 1 for any e
                    # otherwise limit simplification to only simple-arg
                    # Eq instances
                    if len(args) == 2 or _blessed(e):
                        _prevexpr = _prevexpr.subs(*e.args)
                        _expr = _expr.subs(*e.args)
            # Did it evaluate to the same?
            if _prevexpr == _expr:
                # Set the expression for the Not equal section to the same
                # as the next. These will be merged when creating the new
                # Piecewise
                args[i] = args[i].func(args[i + 1][0], cond)
            else:
                # Update the expression that we compare against
                prevexpr = expr
        else:
            prevexpr = expr
    return args


def _piecewise_simplify_eq_and(args):
    """
    Try to simplify conditions and the expression for
    equalities that are part of the condition, e.g.
    Piecewise((n, And(Eq(n,0), Eq(n + m, 0))), (1, True))
    -> Piecewise((0, And(Eq(n, 0), Eq(m, 0))), (1, True))
    """
    for i, (expr, cond) in enumerate(args):
        if isinstance(cond, And):
            eqs, other = sift(cond.args,
                              lambda i: isinstance(i, Eq), binary=True)
        elif isinstance(cond, Eq):
            eqs, other = [cond], []
        else:
            eqs = other = []
        if eqs:
            eqs = list(ordered(eqs))
            for j, e in enumerate(eqs):
                # these blessed lhs objects behave like Symbols
                # and the rhs are simple replacements for the "symbols"
                if _blessed(e):
                    expr = expr.subs(*e.args)
                    eqs[j + 1:] = [ei.subs(*e.args) for ei in eqs[j + 1:]]
                    other = [ei.subs(*e.args) for ei in other]
            cond = And(*(eqs + other))
            args[i] = args[i].func(expr, cond)
    return args


def piecewise_exclusive(expr, *, skip_nan=False, deep=True):
    """
    Rewrite :class:`Piecewise` with mutually exclusive conditions.

    Explanation
    ===========

    SymPy represents the conditions of a :class:`Piecewise` in an
    "if-elif"-fashion, allowing more than one condition to be simultaneously
    True. The interpretation is that the first condition that is True is the
    case that holds. While this is a useful representation computationally it
    is not how a piecewise formula is typically shown in a mathematical text.
    The :func:`piecewise_exclusive` function can be used to rewrite any
    :class:`Piecewise` with more typical mutually exclusive conditions.

    Note that further manipulation of the resulting :class:`Piecewise`, e.g.
    simplifying it, will most likely make it non-exclusive. Hence, this is
    primarily a function to be used in conjunction with printing the Piecewise
    or if one would like to reorder the expression-condition pairs.

    If it is not possible to determine that all possibilities are covered by
    the different cases of the :class:`Piecewise` then a final
    :class:`~sympy.core.numbers.NaN` case will be included explicitly. This
    can be prevented by passing ``skip_nan=True``.

    Examples
    ========

    >>> from sympy import piecewise_exclusive, Symbol, Piecewise, S
    >>> x = Symbol('x', real=True)
    >>> p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))
    >>> piecewise_exclusive(p)
    Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, x > 0))
    >>> piecewise_exclusive(Piecewise((2, x > 1)))
    Piecewise((2, x > 1), (nan, x <= 1))
    >>> piecewise_exclusive(Piecewise((2, x > 1)), skip_nan=True)
    Piecewise((2, x > 1))

    Parameters
    ==========

    expr: a SymPy expression.
        Any :class:`Piecewise` in the expression will be rewritten.
    skip_nan: ``bool`` (default ``False``)
        If ``skip_nan`` is set to ``True`` then a final
        :class:`~sympy.core.numbers.NaN` case will not be included.
    deep:  ``bool`` (default ``True``)
        If ``deep`` is ``True`` then :func:`piecewise_exclusive` will rewrite
        any :class:`Piecewise` subexpressions in ``expr`` rather than just
        rewriting ``expr`` itself.

    Returns
    =======

    An expression equivalent to ``expr`` but where all :class:`Piecewise` have
    been rewritten with mutually exclusive conditions.

    See Also
    ========

    Piecewise
    piecewise_fold
    """

    def make_exclusive(*pwargs):

        cumcond = false
        newargs = []

        # Handle the first n-1 cases
        for expr_i, cond_i in pwargs[:-1]:
            cancond = And(cond_i, Not(cumcond)).simplify()
            cumcond = Or(cond_i, cumcond).simplify()
            newargs.append((expr_i, cancond))

        # For the nth case defer simplification of cumcond
        expr_n, cond_n = pwargs[-1]
        cancond_n = And(cond_n, Not(cumcond)).simplify()
        newargs.append((expr_n, cancond_n))

        if not skip_nan:
            cumcond = Or(cond_n, cumcond).simplify()
            if cumcond is not true:
                newargs.append((Undefined, Not(cumcond).simplify()))

        return Piecewise(*newargs, evaluate=False)

    if deep:
        return expr.replace(Piecewise, make_exclusive)
    elif isinstance(expr, Piecewise):
        return make_exclusive(*expr.args)
    else:
        return expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\symbolic.py ===
"""Fortran/C symbolic expressions

References:
- J3/21-007: Draft Fortran 202x. https://j3-fortran.org/doc/year/21/21-007.pdf

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

# To analyze Fortran expressions to solve dimensions specifications,
# for instances, we implement a minimal symbolic engine for parsing
# expressions into a tree of expression instances. As a first
# instance, we care only about arithmetic expressions involving
# integers and operations like addition (+), subtraction (-),
# multiplication (*), division (Fortran / is Python //, Fortran // is
# concatenate), and exponentiation (**).  In addition, .pyf files may
# contain C expressions that support here is implemented as well.
#
# TODO: support logical constants (Op.BOOLEAN)
# TODO: support logical operators (.AND., ...)
# TODO: support defined operators (.MYOP., ...)
#
__all__ = ['Expr']


import re
import warnings
from enum import Enum
from math import gcd


class Language(Enum):
    """
    Used as Expr.tostring language argument.
    """
    Python = 0
    Fortran = 1
    C = 2


class Op(Enum):
    """
    Used as Expr op attribute.
    """
    INTEGER = 10
    REAL = 12
    COMPLEX = 15
    STRING = 20
    ARRAY = 30
    SYMBOL = 40
    TERNARY = 100
    APPLY = 200
    INDEXING = 210
    CONCAT = 220
    RELATIONAL = 300
    TERMS = 1000
    FACTORS = 2000
    REF = 3000
    DEREF = 3001


class RelOp(Enum):
    """
    Used in Op.RELATIONAL expression to specify the function part.
    """
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    @classmethod
    def fromstring(cls, s, language=Language.C):
        if language is Language.Fortran:
            return {'.eq.': RelOp.EQ, '.ne.': RelOp.NE,
                    '.lt.': RelOp.LT, '.le.': RelOp.LE,
                    '.gt.': RelOp.GT, '.ge.': RelOp.GE}[s.lower()]
        return {'==': RelOp.EQ, '!=': RelOp.NE, '<': RelOp.LT,
                '<=': RelOp.LE, '>': RelOp.GT, '>=': RelOp.GE}[s]

    def tostring(self, language=Language.C):
        if language is Language.Fortran:
            return {RelOp.EQ: '.eq.', RelOp.NE: '.ne.',
                    RelOp.LT: '.lt.', RelOp.LE: '.le.',
                    RelOp.GT: '.gt.', RelOp.GE: '.ge.'}[self]
        return {RelOp.EQ: '==', RelOp.NE: '!=',
                RelOp.LT: '<', RelOp.LE: '<=',
                RelOp.GT: '>', RelOp.GE: '>='}[self]


class ArithOp(Enum):
    """
    Used in Op.APPLY expression to specify the function part.
    """
    POS = 1
    NEG = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    POW = 7


class OpError(Exception):
    pass


class Precedence(Enum):
    """
    Used as Expr.tostring precedence argument.
    """
    ATOM = 0
    POWER = 1
    UNARY = 2
    PRODUCT = 3
    SUM = 4
    LT = 6
    EQ = 7
    LAND = 11
    LOR = 12
    TERNARY = 13
    ASSIGN = 14
    TUPLE = 15
    NONE = 100


integer_types = (int,)
number_types = (int, float)


def _pairs_add(d, k, v):
    # Internal utility method for updating terms and factors data.
    c = d.get(k)
    if c is None:
        d[k] = v
    else:
        c = c + v
        if c:
            d[k] = c
        else:
            del d[k]


class ExprWarning(UserWarning):
    pass


def ewarn(message):
    warnings.warn(message, ExprWarning, stacklevel=2)


class Expr:
    """Represents a Fortran expression as a op-data pair.

    Expr instances are hashable and sortable.
    """

    @staticmethod
    def parse(s, language=Language.C):
        """Parse a Fortran expression to a Expr.
        """
        return fromstring(s, language=language)

    def __init__(self, op, data):
        assert isinstance(op, Op)

        # sanity checks
        if op is Op.INTEGER:
            # data is a 2-tuple of numeric object and a kind value
            # (default is 4)
            assert isinstance(data, tuple) and len(data) == 2
            assert isinstance(data[0], int)
            assert isinstance(data[1], (int, str)), data
        elif op is Op.REAL:
            # data is a 2-tuple of numeric object and a kind value
            # (default is 4)
            assert isinstance(data, tuple) and len(data) == 2
            assert isinstance(data[0], float)
            assert isinstance(data[1], (int, str)), data
        elif op is Op.COMPLEX:
            # data is a 2-tuple of constant expressions
            assert isinstance(data, tuple) and len(data) == 2
        elif op is Op.STRING:
            # data is a 2-tuple of quoted string and a kind value
            # (default is 1)
            assert isinstance(data, tuple) and len(data) == 2
            assert (isinstance(data[0], str)
                    and data[0][::len(data[0]) - 1] in ('""', "''", '@@'))
            assert isinstance(data[1], (int, str)), data
        elif op is Op.SYMBOL:
            # data is any hashable object
            assert hash(data) is not None
        elif op in (Op.ARRAY, Op.CONCAT):
            # data is a tuple of expressions
            assert isinstance(data, tuple)
            assert all(isinstance(item, Expr) for item in data), data
        elif op in (Op.TERMS, Op.FACTORS):
            # data is {<term|base>:<coeff|exponent>} where dict values
            # are nonzero Python integers
            assert isinstance(data, dict)
        elif op is Op.APPLY:
            # data is (<function>, <operands>, <kwoperands>) where
            # operands are Expr instances
            assert isinstance(data, tuple) and len(data) == 3
            # function is any hashable object
            assert hash(data[0]) is not None
            assert isinstance(data[1], tuple)
            assert isinstance(data[2], dict)
        elif op is Op.INDEXING:
            # data is (<object>, <indices>)
            assert isinstance(data, tuple) and len(data) == 2
            # function is any hashable object
            assert hash(data[0]) is not None
        elif op is Op.TERNARY:
            # data is (<cond>, <expr1>, <expr2>)
            assert isinstance(data, tuple) and len(data) == 3
        elif op in (Op.REF, Op.DEREF):
            # data is Expr instance
            assert isinstance(data, Expr)
        elif op is Op.RELATIONAL:
            # data is (<relop>, <left>, <right>)
            assert isinstance(data, tuple) and len(data) == 3
        else:
            raise NotImplementedError(
                f'unknown op or missing sanity check: {op}')

        self.op = op
        self.data = data

    def __eq__(self, other):
        return (isinstance(other, Expr)
                and self.op is other.op
                and self.data == other.data)

    def __hash__(self):
        if self.op in (Op.TERMS, Op.FACTORS):
            data = tuple(sorted(self.data.items()))
        elif self.op is Op.APPLY:
            data = self.data[:2] + tuple(sorted(self.data[2].items()))
        else:
            data = self.data
        return hash((self.op, data))

    def __lt__(self, other):
        if isinstance(other, Expr):
            if self.op is not other.op:
                return self.op.value < other.op.value
            if self.op in (Op.TERMS, Op.FACTORS):
                return (tuple(sorted(self.data.items()))
                        < tuple(sorted(other.data.items())))
            if self.op is Op.APPLY:
                if self.data[:2] != other.data[:2]:
                    return self.data[:2] < other.data[:2]
                return tuple(sorted(self.data[2].items())) < tuple(
                    sorted(other.data[2].items()))
            return self.data < other.data
        return NotImplemented

    def __le__(self, other): return self == other or self < other

    def __gt__(self, other): return not (self <= other)

    def __ge__(self, other): return not (self < other)

    def __repr__(self):
        return f'{type(self).__name__}({self.op}, {self.data!r})'

    def __str__(self):
        return self.tostring()

    def tostring(self, parent_precedence=Precedence.NONE,
                 language=Language.Fortran):
        """Return a string representation of Expr.
        """
        if self.op in (Op.INTEGER, Op.REAL):
            precedence = (Precedence.SUM if self.data[0] < 0
                          else Precedence.ATOM)
            r = str(self.data[0]) + (f'_{self.data[1]}'
                                     if self.data[1] != 4 else '')
        elif self.op is Op.COMPLEX:
            r = ', '.join(item.tostring(Precedence.TUPLE, language=language)
                          for item in self.data)
            r = '(' + r + ')'
            precedence = Precedence.ATOM
        elif self.op is Op.SYMBOL:
            precedence = Precedence.ATOM
            r = str(self.data)
        elif self.op is Op.STRING:
            r = self.data[0]
            if self.data[1] != 1:
                r = self.data[1] + '_' + r
            precedence = Precedence.ATOM
        elif self.op is Op.ARRAY:
            r = ', '.join(item.tostring(Precedence.TUPLE, language=language)
                          for item in self.data)
            r = '[' + r + ']'
            precedence = Precedence.ATOM
        elif self.op is Op.TERMS:
            terms = []
            for term, coeff in sorted(self.data.items()):
                if coeff < 0:
                    op = ' - '
                    coeff = -coeff
                else:
                    op = ' + '
                if coeff == 1:
                    term = term.tostring(Precedence.SUM, language=language)
                elif term == as_number(1):
                    term = str(coeff)
                else:
                    term = f'{coeff} * ' + term.tostring(
                        Precedence.PRODUCT, language=language)
                if terms:
                    terms.append(op)
                elif op == ' - ':
                    terms.append('-')
                terms.append(term)
            r = ''.join(terms) or '0'
            precedence = Precedence.SUM if terms else Precedence.ATOM
        elif self.op is Op.FACTORS:
            factors = []
            tail = []
            for base, exp in sorted(self.data.items()):
                op = ' * '
                if exp == 1:
                    factor = base.tostring(Precedence.PRODUCT,
                                           language=language)
                elif language is Language.C:
                    if exp in range(2, 10):
                        factor = base.tostring(Precedence.PRODUCT,
                                               language=language)
                        factor = ' * '.join([factor] * exp)
                    elif exp in range(-10, 0):
                        factor = base.tostring(Precedence.PRODUCT,
                                               language=language)
                        tail += [factor] * -exp
                        continue
                    else:
                        factor = base.tostring(Precedence.TUPLE,
                                               language=language)
                        factor = f'pow({factor}, {exp})'
                else:
                    factor = base.tostring(Precedence.POWER,
                                           language=language) + f' ** {exp}'
                if factors:
                    factors.append(op)
                factors.append(factor)
            if tail:
                if not factors:
                    factors += ['1']
                factors += ['/', '(', ' * '.join(tail), ')']
            r = ''.join(factors) or '1'
            precedence = Precedence.PRODUCT if factors else Precedence.ATOM
        elif self.op is Op.APPLY:
            name, args, kwargs = self.data
            if name is ArithOp.DIV and language is Language.C:
                numer, denom = [arg.tostring(Precedence.PRODUCT,
                                             language=language)
                                for arg in args]
                r = f'{numer} / {denom}'
                precedence = Precedence.PRODUCT
            else:
                args = [arg.tostring(Precedence.TUPLE, language=language)
                        for arg in args]
                args += [k + '=' + v.tostring(Precedence.NONE)
                         for k, v in kwargs.items()]
                r = f'{name}({", ".join(args)})'
                precedence = Precedence.ATOM
        elif self.op is Op.INDEXING:
            name = self.data[0]
            args = [arg.tostring(Precedence.TUPLE, language=language)
                    for arg in self.data[1:]]
            r = f'{name}[{", ".join(args)}]'
            precedence = Precedence.ATOM
        elif self.op is Op.CONCAT:
            args = [arg.tostring(Precedence.PRODUCT, language=language)
                    for arg in self.data]
            r = " // ".join(args)
            precedence = Precedence.PRODUCT
        elif self.op is Op.TERNARY:
            cond, expr1, expr2 = [a.tostring(Precedence.TUPLE,
                                             language=language)
                                  for a in self.data]
            if language is Language.C:
                r = f'({cond}?{expr1}:{expr2})'
            elif language is Language.Python:
                r = f'({expr1} if {cond} else {expr2})'
            elif language is Language.Fortran:
                r = f'merge({expr1}, {expr2}, {cond})'
            else:
                raise NotImplementedError(
                    f'tostring for {self.op} and {language}')
            precedence = Precedence.ATOM
        elif self.op is Op.REF:
            r = '&' + self.data.tostring(Precedence.UNARY, language=language)
            precedence = Precedence.UNARY
        elif self.op is Op.DEREF:
            r = '*' + self.data.tostring(Precedence.UNARY, language=language)
            precedence = Precedence.UNARY
        elif self.op is Op.RELATIONAL:
            rop, left, right = self.data
            precedence = (Precedence.EQ if rop in (RelOp.EQ, RelOp.NE)
                          else Precedence.LT)
            left = left.tostring(precedence, language=language)
            right = right.tostring(precedence, language=language)
            rop = rop.tostring(language=language)
            r = f'{left} {rop} {right}'
        else:
            raise NotImplementedError(f'tostring for op {self.op}')
        if parent_precedence.value < precedence.value:
            # If parent precedence is higher than operand precedence,
            # operand will be enclosed in parenthesis.
            return '(' + r + ')'
        return r

    def __pos__(self):
        return self

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            if self.op is other.op:
                if self.op in (Op.INTEGER, Op.REAL):
                    return as_number(
                        self.data[0] + other.data[0],
                        max(self.data[1], other.data[1]))
                if self.op is Op.COMPLEX:
                    r1, i1 = self.data
                    r2, i2 = other.data
                    return as_complex(r1 + r2, i1 + i2)
                if self.op is Op.TERMS:
                    r = Expr(self.op, dict(self.data))
                    for k, v in other.data.items():
                        _pairs_add(r.data, k, v)
                    return normalize(r)
            if self.op is Op.COMPLEX and other.op in (Op.INTEGER, Op.REAL):
                return self + as_complex(other)
            elif self.op in (Op.INTEGER, Op.REAL) and other.op is Op.COMPLEX:
                return as_complex(self) + other
            elif self.op is Op.REAL and other.op is Op.INTEGER:
                return self + as_real(other, kind=self.data[1])
            elif self.op is Op.INTEGER and other.op is Op.REAL:
                return as_real(self, kind=other.data[1]) + other
            return as_terms(self) + as_terms(other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, number_types):
            return as_number(other) + self
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, number_types):
            return as_number(other) - self
        return NotImplemented

    def __mul__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            if self.op is other.op:
                if self.op in (Op.INTEGER, Op.REAL):
                    return as_number(self.data[0] * other.data[0],
                                     max(self.data[1], other.data[1]))
                elif self.op is Op.COMPLEX:
                    r1, i1 = self.data
                    r2, i2 = other.data
                    return as_complex(r1 * r2 - i1 * i2, r1 * i2 + r2 * i1)

                if self.op is Op.FACTORS:
                    r = Expr(self.op, dict(self.data))
                    for k, v in other.data.items():
                        _pairs_add(r.data, k, v)
                    return normalize(r)
                elif self.op is Op.TERMS:
                    r = Expr(self.op, {})
                    for t1, c1 in self.data.items():
                        for t2, c2 in other.data.items():
                            _pairs_add(r.data, t1 * t2, c1 * c2)
                    return normalize(r)

            if self.op is Op.COMPLEX and other.op in (Op.INTEGER, Op.REAL):
                return self * as_complex(other)
            elif other.op is Op.COMPLEX and self.op in (Op.INTEGER, Op.REAL):
                return as_complex(self) * other
            elif self.op is Op.REAL and other.op is Op.INTEGER:
                return self * as_real(other, kind=self.data[1])
            elif self.op is Op.INTEGER and other.op is Op.REAL:
                return as_real(self, kind=other.data[1]) * other

            if self.op is Op.TERMS:
                return self * as_terms(other)
            elif other.op is Op.TERMS:
                return as_terms(self) * other

            return as_factors(self) * as_factors(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, number_types):
            return as_number(other) * self
        return NotImplemented

    def __pow__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            if other.op is Op.INTEGER:
                exponent = other.data[0]
                # TODO: other kind not used
                if exponent == 0:
                    return as_number(1)
                if exponent == 1:
                    return self
                if exponent > 0:
                    if self.op is Op.FACTORS:
                        r = Expr(self.op, {})
                        for k, v in self.data.items():
                            r.data[k] = v * exponent
                        return normalize(r)
                    return self * (self ** (exponent - 1))
                elif exponent != -1:
                    return (self ** (-exponent)) ** -1
                return Expr(Op.FACTORS, {self: exponent})
            return as_apply(ArithOp.POW, self, other)
        return NotImplemented

    def __truediv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            # Fortran / is different from Python /:
            # - `/` is a truncate operation for integer operands
            return normalize(as_apply(ArithOp.DIV, self, other))
        return NotImplemented

    def __rtruediv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            return other / self
        return NotImplemented

    def __floordiv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            # Fortran // is different from Python //:
            # - `//` is a concatenate operation for string operands
            return normalize(Expr(Op.CONCAT, (self, other)))
        return NotImplemented

    def __rfloordiv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            return other // self
        return NotImplemented

    def __call__(self, *args, **kwargs):
        # In Fortran, parenthesis () are use for both function call as
        # well as indexing operations.
        #
        # TODO: implement a method for deciding when __call__ should
        # return an INDEXING expression.
        return as_apply(self, *map(as_expr, args),
                        **{k: as_expr(v) for k, v in kwargs.items()})

    def __getitem__(self, index):
        # Provided to support C indexing operations that .pyf files
        # may contain.
        index = as_expr(index)
        if not isinstance(index, tuple):
            index = index,
        if len(index) > 1:
            ewarn(f'C-index should be a single expression but got `{index}`')
        return Expr(Op.INDEXING, (self,) + index)

    def substitute(self, symbols_map):
        """Recursively substitute symbols with values in symbols map.

        Symbols map is a dictionary of symbol-expression pairs.
        """
        if self.op is Op.SYMBOL:
            value = symbols_map.get(self)
            if value is None:
                return self
            m = re.match(r'\A(@__f2py_PARENTHESIS_(\w+)_\d+@)\Z', self.data)
            if m:
                # complement to fromstring method
                items, paren = m.groups()
                if paren in ['ROUNDDIV', 'SQUARE']:
                    return as_array(value)
                assert paren == 'ROUND', (paren, value)
            return value
        if self.op in (Op.INTEGER, Op.REAL, Op.STRING):
            return self
        if self.op in (Op.ARRAY, Op.COMPLEX):
            return Expr(self.op, tuple(item.substitute(symbols_map)
                                       for item in self.data))
        if self.op is Op.CONCAT:
            return normalize(Expr(self.op, tuple(item.substitute(symbols_map)
                                                 for item in self.data)))
        if self.op is Op.TERMS:
            r = None
            for term, coeff in self.data.items():
                if r is None:
                    r = term.substitute(symbols_map) * coeff
                else:
                    r += term.substitute(symbols_map) * coeff
            if r is None:
                ewarn('substitute: empty TERMS expression interpreted as'
                      ' int-literal 0')
                return as_number(0)
            return r
        if self.op is Op.FACTORS:
            r = None
            for base, exponent in self.data.items():
                if r is None:
                    r = base.substitute(symbols_map) ** exponent
                else:
                    r *= base.substitute(symbols_map) ** exponent
            if r is None:
                ewarn('substitute: empty FACTORS expression interpreted'
                      ' as int-literal 1')
                return as_number(1)
            return r
        if self.op is Op.APPLY:
            target, args, kwargs = self.data
            if isinstance(target, Expr):
                target = target.substitute(symbols_map)
            args = tuple(a.substitute(symbols_map) for a in args)
            kwargs = {k: v.substitute(symbols_map)
                          for k, v in kwargs.items()}
            return normalize(Expr(self.op, (target, args, kwargs)))
        if self.op is Op.INDEXING:
            func = self.data[0]
            if isinstance(func, Expr):
                func = func.substitute(symbols_map)
            args = tuple(a.substitute(symbols_map) for a in self.data[1:])
            return normalize(Expr(self.op, (func,) + args))
        if self.op is Op.TERNARY:
            operands = tuple(a.substitute(symbols_map) for a in self.data)
            return normalize(Expr(self.op, operands))
        if self.op in (Op.REF, Op.DEREF):
            return normalize(Expr(self.op, self.data.substitute(symbols_map)))
        if self.op is Op.RELATIONAL:
            rop, left, right = self.data
            left = left.substitute(symbols_map)
            right = right.substitute(symbols_map)
            return normalize(Expr(self.op, (rop, left, right)))
        raise NotImplementedError(f'substitute method for {self.op}: {self!r}')

    def traverse(self, visit, *args, **kwargs):
        """Traverse expression tree with visit function.

        The visit function is applied to an expression with given args
        and kwargs.

        Traverse call returns an expression returned by visit when not
        None, otherwise return a new normalized expression with
        traverse-visit sub-expressions.
        """
        result = visit(self, *args, **kwargs)
        if result is not None:
            return result

        if self.op in (Op.INTEGER, Op.REAL, Op.STRING, Op.SYMBOL):
            return self
        elif self.op in (Op.COMPLEX, Op.ARRAY, Op.CONCAT, Op.TERNARY):
            return normalize(Expr(self.op, tuple(
                item.traverse(visit, *args, **kwargs)
                for item in self.data)))
        elif self.op in (Op.TERMS, Op.FACTORS):
            data = {}
            for k, v in self.data.items():
                k = k.traverse(visit, *args, **kwargs)
                v = (v.traverse(visit, *args, **kwargs)
                     if isinstance(v, Expr) else v)
                if k in data:
                    v = data[k] + v
                data[k] = v
            return normalize(Expr(self.op, data))
        elif self.op is Op.APPLY:
            obj = self.data[0]
            func = (obj.traverse(visit, *args, **kwargs)
                    if isinstance(obj, Expr) else obj)
            operands = tuple(operand.traverse(visit, *args, **kwargs)
                             for operand in self.data[1])
            kwoperands = {k: v.traverse(visit, *args, **kwargs)
                              for k, v in self.data[2].items()}
            return normalize(Expr(self.op, (func, operands, kwoperands)))
        elif self.op is Op.INDEXING:
            obj = self.data[0]
            obj = (obj.traverse(visit, *args, **kwargs)
                   if isinstance(obj, Expr) else obj)
            indices = tuple(index.traverse(visit, *args, **kwargs)
                            for index in self.data[1:])
            return normalize(Expr(self.op, (obj,) + indices))
        elif self.op in (Op.REF, Op.DEREF):
            return normalize(Expr(self.op,
                                  self.data.traverse(visit, *args, **kwargs)))
        elif self.op is Op.RELATIONAL:
            rop, left, right = self.data
            left = left.traverse(visit, *args, **kwargs)
            right = right.traverse(visit, *args, **kwargs)
            return normalize(Expr(self.op, (rop, left, right)))
        raise NotImplementedError(f'traverse method for {self.op}')

    def contains(self, other):
        """Check if self contains other.
        """
        found = []

        def visit(expr, found=found):
            if found:
                return expr
            elif expr == other:
                found.append(1)
                return expr

        self.traverse(visit)

        return len(found) != 0

    def symbols(self):
        """Return a set of symbols contained in self.
        """
        found = set()

        def visit(expr, found=found):
            if expr.op is Op.SYMBOL:
                found.add(expr)

        self.traverse(visit)

        return found

    def polynomial_atoms(self):
        """Return a set of expressions used as atoms in polynomial self.
        """
        found = set()

        def visit(expr, found=found):
            if expr.op is Op.FACTORS:
                for b in expr.data:
                    b.traverse(visit)
                return expr
            if expr.op in (Op.TERMS, Op.COMPLEX):
                return
            if expr.op is Op.APPLY and isinstance(expr.data[0], ArithOp):
                if expr.data[0] is ArithOp.POW:
                    expr.data[1][0].traverse(visit)
                    return expr
                return
            if expr.op in (Op.INTEGER, Op.REAL):
                return expr

            found.add(expr)

            if expr.op in (Op.INDEXING, Op.APPLY):
                return expr

        self.traverse(visit)

        return found

    def linear_solve(self, symbol):
        """Return a, b such that a * symbol + b == self.

        If self is not linear with respect to symbol, raise RuntimeError.
        """
        b = self.substitute({symbol: as_number(0)})
        ax = self - b
        a = ax.substitute({symbol: as_number(1)})

        zero, _ = as_numer_denom(a * symbol - ax)

        if zero != as_number(0):
            raise RuntimeError(f'not a {symbol}-linear equation:'
                               f' {a} * {symbol} + {b} == {self}')
        return a, b


def normalize(obj):
    """Normalize Expr and apply basic evaluation methods.
    """
    if not isinstance(obj, Expr):
        return obj

    if obj.op is Op.TERMS:
        d = {}
        for t, c in obj.data.items():
            if c == 0:
                continue
            if t.op is Op.COMPLEX and c != 1:
                t = t * c
                c = 1
            if t.op is Op.TERMS:
                for t1, c1 in t.data.items():
                    _pairs_add(d, t1, c1 * c)
            else:
                _pairs_add(d, t, c)
        if len(d) == 0:
            # TODO: determine correct kind
            return as_number(0)
        elif len(d) == 1:
            (t, c), = d.items()
            if c == 1:
                return t
        return Expr(Op.TERMS, d)

    if obj.op is Op.FACTORS:
        coeff = 1
        d = {}
        for b, e in obj.data.items():
            if e == 0:
                continue
            if b.op is Op.TERMS and isinstance(e, integer_types) and e > 1:
                # expand integer powers of sums
                b = b * (b ** (e - 1))
                e = 1

            if b.op in (Op.INTEGER, Op.REAL):
                if e == 1:
                    coeff *= b.data[0]
                elif e > 0:
                    coeff *= b.data[0] ** e
                else:
                    _pairs_add(d, b, e)
            elif b.op is Op.FACTORS:
                if e > 0 and isinstance(e, integer_types):
                    for b1, e1 in b.data.items():
                        _pairs_add(d, b1, e1 * e)
                else:
                    _pairs_add(d, b, e)
            else:
                _pairs_add(d, b, e)
        if len(d) == 0 or coeff == 0:
            # TODO: determine correct kind
            assert isinstance(coeff, number_types)
            return as_number(coeff)
        elif len(d) == 1:
            (b, e), = d.items()
            if e == 1:
                t = b
            else:
                t = Expr(Op.FACTORS, d)
            if coeff == 1:
                return t
            return Expr(Op.TERMS, {t: coeff})
        elif coeff == 1:
            return Expr(Op.FACTORS, d)
        else:
            return Expr(Op.TERMS, {Expr(Op.FACTORS, d): coeff})

    if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
        dividend, divisor = obj.data[1]
        t1, c1 = as_term_coeff(dividend)
        t2, c2 = as_term_coeff(divisor)
        if isinstance(c1, integer_types) and isinstance(c2, integer_types):
            g = gcd(c1, c2)
            c1, c2 = c1 // g, c2 // g
        else:
            c1, c2 = c1 / c2, 1

        if t1.op is Op.APPLY and t1.data[0] is ArithOp.DIV:
            numer = t1.data[1][0] * c1
            denom = t1.data[1][1] * t2 * c2
            return as_apply(ArithOp.DIV, numer, denom)

        if t2.op is Op.APPLY and t2.data[0] is ArithOp.DIV:
            numer = t2.data[1][1] * t1 * c1
            denom = t2.data[1][0] * c2
            return as_apply(ArithOp.DIV, numer, denom)

        d = dict(as_factors(t1).data)
        for b, e in as_factors(t2).data.items():
            _pairs_add(d, b, -e)
        numer, denom = {}, {}
        for b, e in d.items():
            if e > 0:
                numer[b] = e
            else:
                denom[b] = -e
        numer = normalize(Expr(Op.FACTORS, numer)) * c1
        denom = normalize(Expr(Op.FACTORS, denom)) * c2

        if denom.op in (Op.INTEGER, Op.REAL) and denom.data[0] == 1:
            # TODO: denom kind not used
            return numer
        return as_apply(ArithOp.DIV, numer, denom)

    if obj.op is Op.CONCAT:
        lst = [obj.data[0]]
        for s in obj.data[1:]:
            last = lst[-1]
            if (
                    last.op is Op.STRING
                    and s.op is Op.STRING
                    and last.data[0][0] in '"\''
                    and s.data[0][0] == last.data[0][-1]
            ):
                new_last = as_string(last.data[0][:-1] + s.data[0][1:],
                                     max(last.data[1], s.data[1]))
                lst[-1] = new_last
            else:
                lst.append(s)
        if len(lst) == 1:
            return lst[0]
        return Expr(Op.CONCAT, tuple(lst))

    if obj.op is Op.TERNARY:
        cond, expr1, expr2 = map(normalize, obj.data)
        if cond.op is Op.INTEGER:
            return expr1 if cond.data[0] else expr2
        return Expr(Op.TERNARY, (cond, expr1, expr2))

    return obj


def as_expr(obj):
    """Convert non-Expr objects to Expr objects.
    """
    if isinstance(obj, complex):
        return as_complex(obj.real, obj.imag)
    if isinstance(obj, number_types):
        return as_number(obj)
    if isinstance(obj, str):
        # STRING expression holds string with boundary quotes, hence
        # applying repr:
        return as_string(repr(obj))
    if isinstance(obj, tuple):
        return tuple(map(as_expr, obj))
    return obj


def as_symbol(obj):
    """Return object as SYMBOL expression (variable or unparsed expression).
    """
    return Expr(Op.SYMBOL, obj)


def as_number(obj, kind=4):
    """Return object as INTEGER or REAL constant.
    """
    if isinstance(obj, int):
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, float):
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op in (Op.INTEGER, Op.REAL):
            return obj
    raise OpError(f'cannot convert {obj} to INTEGER or REAL constant')


def as_integer(obj, kind=4):
    """Return object as INTEGER constant.
    """
    if isinstance(obj, int):
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op is Op.INTEGER:
            return obj
    raise OpError(f'cannot convert {obj} to INTEGER constant')


def as_real(obj, kind=4):
    """Return object as REAL constant.
    """
    if isinstance(obj, int):
        return Expr(Op.REAL, (float(obj), kind))
    if isinstance(obj, float):
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op is Op.REAL:
            return obj
        elif obj.op is Op.INTEGER:
            return Expr(Op.REAL, (float(obj.data[0]), kind))
    raise OpError(f'cannot convert {obj} to REAL constant')


def as_string(obj, kind=1):
    """Return object as STRING expression (string literal constant).
    """
    return Expr(Op.STRING, (obj, kind))


def as_array(obj):
    """Return object as ARRAY expression (array constant).
    """
    if isinstance(obj, Expr):
        obj = obj,
    return Expr(Op.ARRAY, obj)


def as_complex(real, imag=0):
    """Return object as COMPLEX expression (complex literal constant).
    """
    return Expr(Op.COMPLEX, (as_expr(real), as_expr(imag)))


def as_apply(func, *args, **kwargs):
    """Return object as APPLY expression (function call, constructor, etc.)
    """
    return Expr(Op.APPLY,
                (func, tuple(map(as_expr, args)),
                 {k: as_expr(v) for k, v in kwargs.items()}))


def as_ternary(cond, expr1, expr2):
    """Return object as TERNARY expression (cond?expr1:expr2).
    """
    return Expr(Op.TERNARY, (cond, expr1, expr2))


def as_ref(expr):
    """Return object as referencing expression.
    """
    return Expr(Op.REF, expr)


def as_deref(expr):
    """Return object as dereferencing expression.
    """
    return Expr(Op.DEREF, expr)


def as_eq(left, right):
    return Expr(Op.RELATIONAL, (RelOp.EQ, left, right))


def as_ne(left, right):
    return Expr(Op.RELATIONAL, (RelOp.NE, left, right))


def as_lt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LT, left, right))


def as_le(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LE, left, right))


def as_gt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.GT, left, right))


def as_ge(left, right):
    return Expr(Op.RELATIONAL, (RelOp.GE, left, right))


def as_terms(obj):
    """Return expression as TERMS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.TERMS:
            return obj
        if obj.op is Op.INTEGER:
            return Expr(Op.TERMS, {as_integer(1, obj.data[1]): obj.data[0]})
        if obj.op is Op.REAL:
            return Expr(Op.TERMS, {as_real(1, obj.data[1]): obj.data[0]})
        return Expr(Op.TERMS, {obj: 1})
    raise OpError(f'cannot convert {type(obj)} to terms Expr')


def as_factors(obj):
    """Return expression as FACTORS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.FACTORS:
            return obj
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                if coeff == 1:
                    return Expr(Op.FACTORS, {term: 1})
                return Expr(Op.FACTORS, {term: 1, Expr.number(coeff): 1})
        if (obj.op is Op.APPLY
             and obj.data[0] is ArithOp.DIV
             and not obj.data[2]):
            return Expr(Op.FACTORS, {obj.data[1][0]: 1, obj.data[1][1]: -1})
        return Expr(Op.FACTORS, {obj: 1})
    raise OpError(f'cannot convert {type(obj)} to terms Expr')


def as_term_coeff(obj):
    """Return expression as term-coefficient pair.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.INTEGER:
            return as_integer(1, obj.data[1]), obj.data[0]
        if obj.op is Op.REAL:
            return as_real(1, obj.data[1]), obj.data[0]
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                return term, coeff
            # TODO: find common divisor of coefficients
        if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
            t, c = as_term_coeff(obj.data[1][0])
            return as_apply(ArithOp.DIV, t, obj.data[1][1]), c
        return obj, 1
    raise OpError(f'cannot convert {type(obj)} to term and coeff')


def as_numer_denom(obj):
    """Return expression as numer-denom pair.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op in (Op.INTEGER, Op.REAL, Op.COMPLEX, Op.SYMBOL,
                      Op.INDEXING, Op.TERNARY):
            return obj, as_number(1)
        elif obj.op is Op.APPLY:
            if obj.data[0] is ArithOp.DIV and not obj.data[2]:
                numers, denoms = map(as_numer_denom, obj.data[1])
                return numers[0] * denoms[1], numers[1] * denoms[0]
            return obj, as_number(1)
        elif obj.op is Op.TERMS:
            numers, denoms = [], []
            for term, coeff in obj.data.items():
                n, d = as_numer_denom(term)
                n = n * coeff
                numers.append(n)
                denoms.append(d)
            numer, denom = as_number(0), as_number(1)
            for i in range(len(numers)):
                n = numers[i]
                for j in range(len(numers)):
                    if i != j:
                        n *= denoms[j]
                numer += n
                denom *= denoms[i]
            if denom.op in (Op.INTEGER, Op.REAL) and denom.data[0] < 0:
                numer, denom = -numer, -denom
            return numer, denom
        elif obj.op is Op.FACTORS:
            numer, denom = as_number(1), as_number(1)
            for b, e in obj.data.items():
                bnumer, bdenom = as_numer_denom(b)
                if e > 0:
                    numer *= bnumer ** e
                    denom *= bdenom ** e
                elif e < 0:
                    numer *= bdenom ** (-e)
                    denom *= bnumer ** (-e)
            return numer, denom
    raise OpError(f'cannot convert {type(obj)} to numer and denom')


def _counter():
    # Used internally to generate unique dummy symbols
    counter = 0
    while True:
        counter += 1
        yield counter


COUNTER = _counter()


def eliminate_quotes(s):
    """Replace quoted substrings of input string.

    Return a new string and a mapping of replacements.
    """
    d = {}

    def repl(m):
        kind, value = m.groups()[:2]
        if kind:
            # remove trailing underscore
            kind = kind[:-1]
        p = {"'": "SINGLE", '"': "DOUBLE"}[value[0]]
        k = f'{kind}@__f2py_QUOTES_{p}_{COUNTER.__next__()}@'
        d[k] = value
        return k

    new_s = re.sub(r'({kind}_|)({single_quoted}|{double_quoted})'.format(
        kind=r'\w[\w\d_]*',
        single_quoted=r"('([^'\\]|(\\.))*')",
        double_quoted=r'("([^"\\]|(\\.))*")'),
        repl, s)

    assert '"' not in new_s
    assert "'" not in new_s

    return new_s, d


def insert_quotes(s, d):
    """Inverse of eliminate_quotes.
    """
    for k, v in d.items():
        kind = k[:k.find('@')]
        if kind:
            kind += '_'
        s = s.replace(k, kind + v)
    return s


def replace_parenthesis(s):
    """Replace substrings of input that are enclosed in parenthesis.

    Return a new string and a mapping of replacements.
    """
    # Find a parenthesis pair that appears first.

    # Fortran deliminator are `(`, `)`, `[`, `]`, `(/', '/)`, `/`.
    # We don't handle `/` deliminator because it is not a part of an
    # expression.
    left, right = None, None
    mn_i = len(s)
    for left_, right_ in (('(/', '/)'),
                          '()',
                          '{}',  # to support C literal structs
                          '[]'):
        i = s.find(left_)
        if i == -1:
            continue
        if i < mn_i:
            mn_i = i
            left, right = left_, right_

    if left is None:
        return s, {}

    i = mn_i
    j = s.find(right, i)

    while s.count(left, i + 1, j) != s.count(right, i + 1, j):
        j = s.find(right, j + 1)
        if j == -1:
            raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

    p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]

    k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
    v = s[i + len(left):j]
    r, d = replace_parenthesis(s[j + len(right):])
    d[k] = v
    return s[:i] + k + r, d


def _get_parenthesis_kind(s):
    assert s.startswith('@__f2py_PARENTHESIS_'), s
    return s.split('_')[4]


def unreplace_parenthesis(s, d):
    """Inverse of replace_parenthesis.
    """
    for k, v in d.items():
        p = _get_parenthesis_kind(k)
        left = {'ROUND': '(', 'SQUARE': '[', 'CURLY': '{', 'ROUNDDIV': '(/'}[p]
        right = {'ROUND': ')', 'SQUARE': ']', 'CURLY': '}', 'ROUNDDIV': '/)'}[p]
        s = s.replace(k, left + v + right)
    return s


def fromstring(s, language=Language.C):
    """Create an expression from a string.

    This is a "lazy" parser, that is, only arithmetic operations are
    resolved, non-arithmetic operations are treated as symbols.
    """
    r = _FromStringWorker(language=language).parse(s)
    if isinstance(r, Expr):
        return r
    raise ValueError(f'failed to parse `{s}` to Expr instance: got `{r}`')


class _Pair:
    # Internal class to represent a pair of expressions

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def substitute(self, symbols_map):
        left, right = self.left, self.right
        if isinstance(left, Expr):
            left = left.substitute(symbols_map)
        if isinstance(right, Expr):
            right = right.substitute(symbols_map)
        return _Pair(left, right)

    def __repr__(self):
        return f'{type(self).__name__}({self.left}, {self.right})'


class _FromStringWorker:

    def __init__(self, language=Language.C):
        self.original = None
        self.quotes_map = None
        self.language = language

    def finalize_string(self, s):
        return insert_quotes(s, self.quotes_map)

    def parse(self, inp):
        self.original = inp
        unquoted, self.quotes_map = eliminate_quotes(inp)
        return self.process(unquoted)

    def process(self, s, context='expr'):
        """Parse string within the given context.

        The context may define the result in case of ambiguous
        expressions. For instance, consider expressions `f(x, y)` and
        `(x, y) + (a, b)` where `f` is a function and pair `(x, y)`
        denotes complex number. Specifying context as "args" or
        "expr", the subexpression `(x, y)` will be parse to an
        argument list or to a complex number, respectively.
        """
        if isinstance(s, (list, tuple)):
            return type(s)(self.process(s_, context) for s_ in s)

        assert isinstance(s, str), (type(s), s)

        # replace subexpressions in parenthesis with f2py @-names
        r, raw_symbols_map = replace_parenthesis(s)
        r = r.strip()

        def restore(r):
            # restores subexpressions marked with f2py @-names
            if isinstance(r, (list, tuple)):
                return type(r)(map(restore, r))
            return unreplace_parenthesis(r, raw_symbols_map)

        # comma-separated tuple
        if ',' in r:
            operands = restore(r.split(','))
            if context == 'args':
                return tuple(self.process(operands))
            if context == 'expr':
                if len(operands) == 2:
                    # complex number literal
                    return as_complex(*self.process(operands))
            raise NotImplementedError(
                f'parsing comma-separated list (context={context}): {r}')

        # ternary operation
        m = re.match(r'\A([^?]+)[?]([^:]+)[:](.+)\Z', r)
        if m:
            assert context == 'expr', context
            oper, expr1, expr2 = restore(m.groups())
            oper = self.process(oper)
            expr1 = self.process(expr1)
            expr2 = self.process(expr2)
            return as_ternary(oper, expr1, expr2)

        # relational expression
        if self.language is Language.Fortran:
            m = re.match(
                r'\A(.+)\s*[.](eq|ne|lt|le|gt|ge)[.]\s*(.+)\Z', r, re.I)
        else:
            m = re.match(
                r'\A(.+)\s*([=][=]|[!][=]|[<][=]|[<]|[>][=]|[>])\s*(.+)\Z', r)
        if m:
            left, rop, right = m.groups()
            if self.language is Language.Fortran:
                rop = '.' + rop + '.'
            left, right = self.process(restore((left, right)))
            rop = RelOp.fromstring(rop, language=self.language)
            return Expr(Op.RELATIONAL, (rop, left, right))

        # keyword argument
        m = re.match(r'\A(\w[\w\d_]*)\s*[=](.*)\Z', r)
        if m:
            keyname, value = m.groups()
            value = restore(value)
            return _Pair(keyname, self.process(value))

        # addition/subtraction operations
        operands = re.split(r'((?<!\d[edED])[+-])', r)
        if len(operands) > 1:
            result = self.process(restore(operands[0] or '0'))
            for op, operand in zip(operands[1::2], operands[2::2]):
                operand = self.process(restore(operand))
                op = op.strip()
                if op == '+':
                    result += operand
                else:
                    assert op == '-'
                    result -= operand
            return result

        # string concatenate operation
        if self.language is Language.Fortran and '//' in r:
            operands = restore(r.split('//'))
            return Expr(Op.CONCAT,
                        tuple(self.process(operands)))

        # multiplication/division operations
        operands = re.split(r'(?<=[@\w\d_])\s*([*]|/)',
                            (r if self.language is Language.C
                             else r.replace('**', '@__f2py_DOUBLE_STAR@')))
        if len(operands) > 1:
            operands = restore(operands)
            if self.language is not Language.C:
                operands = [operand.replace('@__f2py_DOUBLE_STAR@', '**')
                            for operand in operands]
            # Expression is an arithmetic product
            result = self.process(operands[0])
            for op, operand in zip(operands[1::2], operands[2::2]):
                operand = self.process(operand)
                op = op.strip()
                if op == '*':
                    result *= operand
                else:
                    assert op == '/'
                    result /= operand
            return result

        # referencing/dereferencing
        if r.startswith(('*', '&')):
            op = {'*': Op.DEREF, '&': Op.REF}[r[0]]
            operand = self.process(restore(r[1:]))
            return Expr(op, operand)

        # exponentiation operations
        if self.language is not Language.C and '**' in r:
            operands = list(reversed(restore(r.split('**'))))
            result = self.process(operands[0])
            for operand in operands[1:]:
                operand = self.process(operand)
                result = operand ** result
            return result

        # int-literal-constant
        m = re.match(r'\A({digit_string})({kind}|)\Z'.format(
            digit_string=r'\d+',
            kind=r'_(\d+|\w[\w\d_]*)'), r)
        if m:
            value, _, kind = m.groups()
            if kind and kind.isdigit():
                kind = int(kind)
            return as_integer(int(value), kind or 4)

        # real-literal-constant
        m = re.match(r'\A({significant}({exponent}|)|\d+{exponent})({kind}|)\Z'
                     .format(
                         significant=r'[.]\d+|\d+[.]\d*',
                         exponent=r'[edED][+-]?\d+',
                         kind=r'_(\d+|\w[\w\d_]*)'), r)
        if m:
            value, _, _, kind = m.groups()
            if kind and kind.isdigit():
                kind = int(kind)
            value = value.lower()
            if 'd' in value:
                return as_real(float(value.replace('d', 'e')), kind or 8)
            return as_real(float(value), kind or 4)

        # string-literal-constant with kind parameter specification
        if r in self.quotes_map:
            kind = r[:r.find('@')]
            return as_string(self.quotes_map[r], kind or 1)

        # array constructor or literal complex constant or
        # parenthesized expression
        if r in raw_symbols_map:
            paren = _get_parenthesis_kind(r)
            items = self.process(restore(raw_symbols_map[r]),
                                 'expr' if paren == 'ROUND' else 'args')
            if paren == 'ROUND':
                if isinstance(items, Expr):
                    return items
            if paren in ['ROUNDDIV', 'SQUARE']:
                # Expression is a array constructor
                if isinstance(items, Expr):
                    items = (items,)
                return as_array(items)

        # function call/indexing
        m = re.match(r'\A(.+)\s*(@__f2py_PARENTHESIS_(ROUND|SQUARE)_\d+@)\Z',
                     r)
        if m:
            target, args, paren = m.groups()
            target = self.process(restore(target))
            args = self.process(restore(args)[1:-1], 'args')
            if not isinstance(args, tuple):
                args = args,
            if paren == 'ROUND':
                kwargs = {a.left: a.right for a in args
                              if isinstance(a, _Pair)}
                args = tuple(a for a in args if not isinstance(a, _Pair))
                # Warning: this could also be Fortran indexing operation..
                return as_apply(target, *args, **kwargs)
            else:
                # Expression is a C/Python indexing operation
                # (e.g. used in .pyf files)
                assert paren == 'SQUARE'
                return target[args]

        # Fortran standard conforming identifier
        m = re.match(r'\A\w[\w\d_]*\Z', r)
        if m:
            return as_symbol(r)

        # fall-back to symbol
        r = self.finalize_string(restore(r))
        ewarn(
            f'fromstring: treating {r!r} as symbol (original={self.original})')
        return as_symbol(r)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\pool\base.py ===
# pool/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


"""Base constructs for connection pools.

"""

from __future__ import annotations

from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref

from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol

if TYPE_CHECKING:
    from ..engine.interfaces import DBAPIConnection
    from ..engine.interfaces import DBAPICursor
    from ..engine.interfaces import Dialect
    from ..event import _DispatchCommon
    from ..event import _ListenerFnType
    from ..event import dispatcher
    from ..sql._typing import _InfoType


@dataclasses.dataclass(frozen=True)
class PoolResetState:
    """describes the state of a DBAPI connection as it is being passed to
    the :meth:`.PoolEvents.reset` connection pool event.

    .. versionadded:: 2.0.0b3

    """

    __slots__ = ("transaction_was_reset", "terminate_only", "asyncio_safe")

    transaction_was_reset: bool
    """Indicates if the transaction on the DBAPI connection was already
    essentially "reset" back by the :class:`.Connection` object.

    This boolean is True if the :class:`.Connection` had transactional
    state present upon it, which was then not closed using the
    :meth:`.Connection.rollback` or :meth:`.Connection.commit` method;
    instead, the transaction was closed inline within the
    :meth:`.Connection.close` method so is guaranteed to remain non-present
    when this event is reached.

    """

    terminate_only: bool
    """indicates if the connection is to be immediately terminated and
    not checked in to the pool.

    This occurs for connections that were invalidated, as well as asyncio
    connections that were not cleanly handled by the calling code that
    are instead being garbage collected.   In the latter case,
    operations can't be safely run on asyncio connections within garbage
    collection as there is not necessarily an event loop present.

    """

    asyncio_safe: bool
    """Indicates if the reset operation is occurring within a scope where
    an enclosing event loop is expected to be present for asyncio applications.

    Will be False in the case that the connection is being garbage collected.

    """


class ResetStyle(Enum):
    """Describe options for "reset on return" behaviors."""

    reset_rollback = 0
    reset_commit = 1
    reset_none = 2


_ResetStyleArgType = Union[
    ResetStyle,
    Literal[True, None, False, "commit", "rollback"],
]
reset_rollback, reset_commit, reset_none = list(ResetStyle)


class _ConnDialect:
    """partial implementation of :class:`.Dialect`
    which provides DBAPI connection methods.

    When a :class:`_pool.Pool` is combined with an :class:`_engine.Engine`,
    the :class:`_engine.Engine` replaces this with its own
    :class:`.Dialect`.

    """

    is_async = False
    has_terminate = False

    def do_rollback(self, dbapi_connection: PoolProxiedConnection) -> None:
        dbapi_connection.rollback()

    def do_commit(self, dbapi_connection: PoolProxiedConnection) -> None:
        dbapi_connection.commit()

    def do_terminate(self, dbapi_connection: DBAPIConnection) -> None:
        dbapi_connection.close()

    def do_close(self, dbapi_connection: DBAPIConnection) -> None:
        dbapi_connection.close()

    def _do_ping_w_event(self, dbapi_connection: DBAPIConnection) -> bool:
        raise NotImplementedError(
            "The ping feature requires that a dialect is "
            "passed to the connection pool."
        )

    def get_driver_connection(self, connection: DBAPIConnection) -> Any:
        return connection


class _AsyncConnDialect(_ConnDialect):
    is_async = True


class _CreatorFnType(Protocol):
    def __call__(self) -> DBAPIConnection: ...


class _CreatorWRecFnType(Protocol):
    def __call__(self, rec: ConnectionPoolEntry) -> DBAPIConnection: ...


class Pool(log.Identified, event.EventTarget):
    """Abstract base class for connection pools."""

    dispatch: dispatcher[Pool]
    echo: log._EchoFlagType

    _orig_logging_name: Optional[str]
    _dialect: Union[_ConnDialect, Dialect] = _ConnDialect()
    _creator_arg: Union[_CreatorFnType, _CreatorWRecFnType]
    _invoke_creator: _CreatorWRecFnType
    _invalidate_time: float

    def __init__(
        self,
        creator: Union[_CreatorFnType, _CreatorWRecFnType],
        recycle: int = -1,
        echo: log._EchoFlagType = None,
        logging_name: Optional[str] = None,
        reset_on_return: _ResetStyleArgType = True,
        events: Optional[List[Tuple[_ListenerFnType, str]]] = None,
        dialect: Optional[Union[_ConnDialect, Dialect]] = None,
        pre_ping: bool = False,
        _dispatch: Optional[_DispatchCommon[Pool]] = None,
    ):
        """
        Construct a Pool.

        :param creator: a callable function that returns a DB-API
          connection object.  The function will be called with
          parameters.

        :param recycle: If set to a value other than -1, number of
          seconds between connection recycling, which means upon
          checkout, if this timeout is surpassed the connection will be
          closed and replaced with a newly opened connection. Defaults to -1.

        :param logging_name:  String identifier which will be used within
          the "name" field of logging records generated within the
          "sqlalchemy.pool" logger. Defaults to a hexstring of the object's
          id.

        :param echo: if True, the connection pool will log
         informational output such as when connections are invalidated
         as well as when connections are recycled to the default log handler,
         which defaults to ``sys.stdout`` for output..   If set to the string
         ``"debug"``, the logging will include pool checkouts and checkins.

         The :paramref:`_pool.Pool.echo` parameter can also be set from the
         :func:`_sa.create_engine` call by using the
         :paramref:`_sa.create_engine.echo_pool` parameter.

         .. seealso::

             :ref:`dbengine_logging` - further detail on how to configure
             logging.

        :param reset_on_return: Determine steps to take on
         connections as they are returned to the pool, which were
         not otherwise handled by a :class:`_engine.Connection`.
         Available from :func:`_sa.create_engine` via the
         :paramref:`_sa.create_engine.pool_reset_on_return` parameter.

         :paramref:`_pool.Pool.reset_on_return` can have any of these values:

         * ``"rollback"`` - call rollback() on the connection,
           to release locks and transaction resources.
           This is the default value.  The vast majority
           of use cases should leave this value set.
         * ``"commit"`` - call commit() on the connection,
           to release locks and transaction resources.
           A commit here may be desirable for databases that
           cache query plans if a commit is emitted,
           such as Microsoft SQL Server.  However, this
           value is more dangerous than 'rollback' because
           any data changes present on the transaction
           are committed unconditionally.
         * ``None`` - don't do anything on the connection.
           This setting may be appropriate if the database / DBAPI
           works in pure "autocommit" mode at all times, or if
           a custom reset handler is established using the
           :meth:`.PoolEvents.reset` event handler.

         * ``True`` - same as 'rollback', this is here for
           backwards compatibility.
         * ``False`` - same as None, this is here for
           backwards compatibility.

         For further customization of reset on return, the
         :meth:`.PoolEvents.reset` event hook may be used which can perform
         any connection activity desired on reset.

         .. seealso::

            :ref:`pool_reset_on_return`

            :meth:`.PoolEvents.reset`

        :param events: a list of 2-tuples, each of the form
         ``(callable, target)`` which will be passed to :func:`.event.listen`
         upon construction.   Provided here so that event listeners
         can be assigned via :func:`_sa.create_engine` before dialect-level
         listeners are applied.

        :param dialect: a :class:`.Dialect` that will handle the job
         of calling rollback(), close(), or commit() on DBAPI connections.
         If omitted, a built-in "stub" dialect is used.   Applications that
         make use of :func:`_sa.create_engine` should not use this parameter
         as it is handled by the engine creation strategy.

        :param pre_ping: if True, the pool will emit a "ping" (typically
         "SELECT 1", but is dialect-specific) on the connection
         upon checkout, to test if the connection is alive or not.   If not,
         the connection is transparently re-connected and upon success, all
         other pooled connections established prior to that timestamp are
         invalidated.     Requires that a dialect is passed as well to
         interpret the disconnection error.

         .. versionadded:: 1.2

        """
        if logging_name:
            self.logging_name = self._orig_logging_name = logging_name
        else:
            self._orig_logging_name = None

        log.instance_logger(self, echoflag=echo)
        self._creator = creator
        self._recycle = recycle
        self._invalidate_time = 0
        self._pre_ping = pre_ping
        self._reset_on_return = util.parse_user_argument_for_enum(
            reset_on_return,
            {
                ResetStyle.reset_rollback: ["rollback", True],
                ResetStyle.reset_none: ["none", None, False],
                ResetStyle.reset_commit: ["commit"],
            },
            "reset_on_return",
        )

        self.echo = echo

        if _dispatch:
            self.dispatch._update(_dispatch, only_propagate=False)
        if dialect:
            self._dialect = dialect
        if events:
            for fn, target in events:
                event.listen(self, target, fn)

    @util.hybridproperty
    def _is_asyncio(self) -> bool:
        return self._dialect.is_async

    @property
    def _creator(self) -> Union[_CreatorFnType, _CreatorWRecFnType]:
        return self._creator_arg

    @_creator.setter
    def _creator(
        self, creator: Union[_CreatorFnType, _CreatorWRecFnType]
    ) -> None:
        self._creator_arg = creator

        # mypy seems to get super confused assigning functions to
        # attributes
        self._invoke_creator = self._should_wrap_creator(creator)

    @_creator.deleter
    def _creator(self) -> None:
        # needed for mock testing
        del self._creator_arg
        del self._invoke_creator

    def _should_wrap_creator(
        self, creator: Union[_CreatorFnType, _CreatorWRecFnType]
    ) -> _CreatorWRecFnType:
        """Detect if creator accepts a single argument, or is sent
        as a legacy style no-arg function.

        """

        try:
            argspec = util.get_callable_argspec(self._creator, no_self=True)
        except TypeError:
            creator_fn = cast(_CreatorFnType, creator)
            return lambda rec: creator_fn()

        if argspec.defaults is not None:
            defaulted = len(argspec.defaults)
        else:
            defaulted = 0
        positionals = len(argspec[0]) - defaulted

        # look for the exact arg signature that DefaultStrategy
        # sends us
        if (argspec[0], argspec[3]) == (["connection_record"], (None,)):
            return cast(_CreatorWRecFnType, creator)
        # or just a single positional
        elif positionals == 1:
            return cast(_CreatorWRecFnType, creator)
        # all other cases, just wrap and assume legacy "creator" callable
        # thing
        else:
            creator_fn = cast(_CreatorFnType, creator)
            return lambda rec: creator_fn()

    def _close_connection(
        self, connection: DBAPIConnection, *, terminate: bool = False
    ) -> None:
        self.logger.debug(
            "%s connection %r",
            "Hard-closing" if terminate else "Closing",
            connection,
        )
        try:
            if terminate:
                self._dialect.do_terminate(connection)
            else:
                self._dialect.do_close(connection)
        except BaseException as e:
            self.logger.error(
                f"Exception {'terminating' if terminate else 'closing'} "
                f"connection %r",
                connection,
                exc_info=True,
            )
            if not isinstance(e, Exception):
                raise

    def _create_connection(self) -> ConnectionPoolEntry:
        """Called by subclasses to create a new ConnectionRecord."""

        return _ConnectionRecord(self)

    def _invalidate(
        self,
        connection: PoolProxiedConnection,
        exception: Optional[BaseException] = None,
        _checkin: bool = True,
    ) -> None:
        """Mark all connections established within the generation
        of the given connection as invalidated.

        If this pool's last invalidate time is before when the given
        connection was created, update the timestamp til now.  Otherwise,
        no action is performed.

        Connections with a start time prior to this pool's invalidation
        time will be recycled upon next checkout.
        """
        rec = getattr(connection, "_connection_record", None)
        if not rec or self._invalidate_time < rec.starttime:
            self._invalidate_time = time.time()
        if _checkin and getattr(connection, "is_valid", False):
            connection.invalidate(exception)

    def recreate(self) -> Pool:
        """Return a new :class:`_pool.Pool`, of the same class as this one
        and configured with identical creation arguments.

        This method is used in conjunction with :meth:`dispose`
        to close out an entire :class:`_pool.Pool` and create a new one in
        its place.

        """

        raise NotImplementedError()

    def dispose(self) -> None:
        """Dispose of this pool.

        This method leaves the possibility of checked-out connections
        remaining open, as it only affects connections that are
        idle in the pool.

        .. seealso::

            :meth:`Pool.recreate`

        """

        raise NotImplementedError()

    def connect(self) -> PoolProxiedConnection:
        """Return a DBAPI connection from the pool.

        The connection is instrumented such that when its
        ``close()`` method is called, the connection will be returned to
        the pool.

        """
        return _ConnectionFairy._checkout(self)

    def _return_conn(self, record: ConnectionPoolEntry) -> None:
        """Given a _ConnectionRecord, return it to the :class:`_pool.Pool`.

        This method is called when an instrumented DBAPI connection
        has its ``close()`` method called.

        """
        self._do_return_conn(record)

    def _do_get(self) -> ConnectionPoolEntry:
        """Implementation for :meth:`get`, supplied by subclasses."""

        raise NotImplementedError()

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        """Implementation for :meth:`return_conn`, supplied by subclasses."""

        raise NotImplementedError()

    def status(self) -> str:
        """Returns a brief description of the state of this pool."""
        raise NotImplementedError()


class ManagesConnection:
    """Common base for the two connection-management interfaces
    :class:`.PoolProxiedConnection` and :class:`.ConnectionPoolEntry`.

    These two objects are typically exposed in the public facing API
    via the connection pool event hooks, documented at :class:`.PoolEvents`.

    .. versionadded:: 2.0

    """

    __slots__ = ()

    dbapi_connection: Optional[DBAPIConnection]
    """A reference to the actual DBAPI connection being tracked.

    This is a :pep:`249`-compliant object that for traditional sync-style
    dialects is provided by the third-party
    DBAPI implementation in use.  For asyncio dialects, the implementation
    is typically an adapter object provided by the SQLAlchemy dialect
    itself; the underlying asyncio object is available via the
    :attr:`.ManagesConnection.driver_connection` attribute.

    SQLAlchemy's interface for the DBAPI connection is based on the
    :class:`.DBAPIConnection` protocol object

    .. seealso::

        :attr:`.ManagesConnection.driver_connection`

        :ref:`faq_dbapi_connection`

    """

    driver_connection: Optional[Any]
    """The "driver level" connection object as used by the Python
    DBAPI or database driver.

    For traditional :pep:`249` DBAPI implementations, this object will
    be the same object as that of
    :attr:`.ManagesConnection.dbapi_connection`.   For an asyncio database
    driver, this will be the ultimate "connection" object used by that
    driver, such as the ``asyncpg.Connection`` object which will not have
    standard pep-249 methods.

    .. versionadded:: 1.4.24

    .. seealso::

        :attr:`.ManagesConnection.dbapi_connection`

        :ref:`faq_dbapi_connection`

    """

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        """Info dictionary associated with the underlying DBAPI connection
        referred to by this :class:`.ManagesConnection` instance, allowing
        user-defined data to be associated with the connection.

        The data in this dictionary is persistent for the lifespan
        of the DBAPI connection itself, including across pool checkins
        and checkouts.  When the connection is invalidated
        and replaced with a new one, this dictionary is cleared.

        For a :class:`.PoolProxiedConnection` instance that's not associated
        with a :class:`.ConnectionPoolEntry`, such as if it were detached, the
        attribute returns a dictionary that is local to that
        :class:`.ConnectionPoolEntry`. Therefore the
        :attr:`.ManagesConnection.info` attribute will always provide a Python
        dictionary.

        .. seealso::

            :attr:`.ManagesConnection.record_info`


        """
        raise NotImplementedError()

    @util.ro_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        """Persistent info dictionary associated with this
        :class:`.ManagesConnection`.

        Unlike the :attr:`.ManagesConnection.info` dictionary, the lifespan
        of this dictionary is that of the :class:`.ConnectionPoolEntry`
        which owns it; therefore this dictionary will persist across
        reconnects and connection invalidation for a particular entry
        in the connection pool.

        For a :class:`.PoolProxiedConnection` instance that's not associated
        with a :class:`.ConnectionPoolEntry`, such as if it were detached, the
        attribute returns None. Contrast to the :attr:`.ManagesConnection.info`
        dictionary which is never None.


        .. seealso::

            :attr:`.ManagesConnection.info`

        """
        raise NotImplementedError()

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        """Mark the managed connection as invalidated.

        :param e: an exception object indicating a reason for the invalidation.

        :param soft: if True, the connection isn't closed; instead, this
         connection will be recycled on next checkout.

        .. seealso::

            :ref:`pool_connection_invalidation`


        """
        raise NotImplementedError()


class ConnectionPoolEntry(ManagesConnection):
    """Interface for the object that maintains an individual database
    connection on behalf of a :class:`_pool.Pool` instance.

    The :class:`.ConnectionPoolEntry` object represents the long term
    maintainance of a particular connection for a pool, including expiring or
    invalidating that connection to have it replaced with a new one, which will
    continue to be maintained by that same :class:`.ConnectionPoolEntry`
    instance. Compared to :class:`.PoolProxiedConnection`, which is the
    short-term, per-checkout connection manager, this object lasts for the
    lifespan of a particular "slot" within a connection pool.

    The :class:`.ConnectionPoolEntry` object is mostly visible to public-facing
    API code when it is delivered to connection pool event hooks, such as
    :meth:`_events.PoolEvents.connect` and :meth:`_events.PoolEvents.checkout`.

    .. versionadded:: 2.0  :class:`.ConnectionPoolEntry` provides the public
       facing interface for the :class:`._ConnectionRecord` internal class.

    """

    __slots__ = ()

    @property
    def in_use(self) -> bool:
        """Return True the connection is currently checked out"""

        raise NotImplementedError()

    def close(self) -> None:
        """Close the DBAPI connection managed by this connection pool entry."""
        raise NotImplementedError()


class _ConnectionRecord(ConnectionPoolEntry):
    """Maintains a position in a connection pool which references a pooled
    connection.

    This is an internal object used by the :class:`_pool.Pool` implementation
    to provide context management to a DBAPI connection maintained by
    that :class:`_pool.Pool`.   The public facing interface for this class
    is described by the :class:`.ConnectionPoolEntry` class.  See that
    class for public API details.

    .. seealso::

        :class:`.ConnectionPoolEntry`

        :class:`.PoolProxiedConnection`

    """

    __slots__ = (
        "__pool",
        "fairy_ref",
        "finalize_callback",
        "fresh",
        "starttime",
        "dbapi_connection",
        "__weakref__",
        "__dict__",
    )

    finalize_callback: Deque[Callable[[DBAPIConnection], None]]
    fresh: bool
    fairy_ref: Optional[weakref.ref[_ConnectionFairy]]
    starttime: float

    def __init__(self, pool: Pool, connect: bool = True):
        self.fresh = False
        self.fairy_ref = None
        self.starttime = 0
        self.dbapi_connection = None

        self.__pool = pool
        if connect:
            self.__connect()
        self.finalize_callback = deque()

    dbapi_connection: Optional[DBAPIConnection]

    @property
    def driver_connection(self) -> Optional[Any]:  # type: ignore[override]  # mypy#4125  # noqa: E501
        if self.dbapi_connection is None:
            return None
        else:
            return self.__pool._dialect.get_driver_connection(
                self.dbapi_connection
            )

    @property
    @util.deprecated(
        "2.0",
        "The _ConnectionRecord.connection attribute is deprecated; "
        "please use 'driver_connection'",
    )
    def connection(self) -> Optional[DBAPIConnection]:
        return self.dbapi_connection

    _soft_invalidate_time: float = 0

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        return {}

    @util.ro_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        return {}

    @classmethod
    def checkout(cls, pool: Pool) -> _ConnectionFairy:
        if TYPE_CHECKING:
            rec = cast(_ConnectionRecord, pool._do_get())
        else:
            rec = pool._do_get()

        try:
            dbapi_connection = rec.get_connection()
        except BaseException as err:
            with util.safe_reraise():
                rec._checkin_failed(err, _fairy_was_created=False)

            # not reached, for code linters only
            raise

        echo = pool._should_log_debug()
        fairy = _ConnectionFairy(pool, dbapi_connection, rec, echo)

        rec.fairy_ref = ref = weakref.ref(
            fairy,
            lambda ref: (
                _finalize_fairy(
                    None, rec, pool, ref, echo, transaction_was_reset=False
                )
                if _finalize_fairy is not None
                else None
            ),
        )
        _strong_ref_connection_records[ref] = rec
        if echo:
            pool.logger.debug(
                "Connection %r checked out from pool", dbapi_connection
            )
        return fairy

    def _checkin_failed(
        self, err: BaseException, _fairy_was_created: bool = True
    ) -> None:
        self.invalidate(e=err)
        self.checkin(
            _fairy_was_created=_fairy_was_created,
        )

    def checkin(self, _fairy_was_created: bool = True) -> None:
        if self.fairy_ref is None and _fairy_was_created:
            # _fairy_was_created is False for the initial get connection phase;
            # meaning there was no _ConnectionFairy and we must unconditionally
            # do a checkin.
            #
            # otherwise, if fairy_was_created==True, if fairy_ref is None here
            # that means we were checked in already, so this looks like
            # a double checkin.
            util.warn("Double checkin attempted on %s" % self)
            return
        self.fairy_ref = None
        connection = self.dbapi_connection
        pool = self.__pool
        while self.finalize_callback:
            finalizer = self.finalize_callback.pop()
            if connection is not None:
                finalizer(connection)
        if pool.dispatch.checkin:
            pool.dispatch.checkin(connection, self)

        pool._return_conn(self)

    @property
    def in_use(self) -> bool:
        return self.fairy_ref is not None

    @property
    def last_connect_time(self) -> float:
        return self.starttime

    def close(self) -> None:
        if self.dbapi_connection is not None:
            self.__close()

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        # already invalidated
        if self.dbapi_connection is None:
            return
        if soft:
            self.__pool.dispatch.soft_invalidate(
                self.dbapi_connection, self, e
            )
        else:
            self.__pool.dispatch.invalidate(self.dbapi_connection, self, e)
        if e is not None:
            self.__pool.logger.info(
                "%sInvalidate connection %r (reason: %s:%s)",
                "Soft " if soft else "",
                self.dbapi_connection,
                e.__class__.__name__,
                e,
            )
        else:
            self.__pool.logger.info(
                "%sInvalidate connection %r",
                "Soft " if soft else "",
                self.dbapi_connection,
            )

        if soft:
            self._soft_invalidate_time = time.time()
        else:
            self.__close(terminate=True)
            self.dbapi_connection = None

    def get_connection(self) -> DBAPIConnection:
        recycle = False

        # NOTE: the various comparisons here are assuming that measurable time
        # passes between these state changes.  however, time.time() is not
        # guaranteed to have sub-second precision.  comparisons of
        # "invalidation time" to "starttime" should perhaps use >= so that the
        # state change can take place assuming no measurable  time has passed,
        # however this does not guarantee correct behavior here as if time
        # continues to not pass, it will try to reconnect repeatedly until
        # these timestamps diverge, so in that sense using > is safer.  Per
        # https://stackoverflow.com/a/1938096/34549, Windows time.time() may be
        # within 16 milliseconds accuracy, so unit tests for connection
        # invalidation need a sleep of at least this long between initial start
        # time and invalidation for the logic below to work reliably.

        if self.dbapi_connection is None:
            self.info.clear()
            self.__connect()
        elif (
            self.__pool._recycle > -1
            and time.time() - self.starttime > self.__pool._recycle
        ):
            self.__pool.logger.info(
                "Connection %r exceeded timeout; recycling",
                self.dbapi_connection,
            )
            recycle = True
        elif self.__pool._invalidate_time > self.starttime:
            self.__pool.logger.info(
                "Connection %r invalidated due to pool invalidation; "
                + "recycling",
                self.dbapi_connection,
            )
            recycle = True
        elif self._soft_invalidate_time > self.starttime:
            self.__pool.logger.info(
                "Connection %r invalidated due to local soft invalidation; "
                + "recycling",
                self.dbapi_connection,
            )
            recycle = True

        if recycle:
            self.__close(terminate=True)
            self.info.clear()

            self.__connect()

        assert self.dbapi_connection is not None
        return self.dbapi_connection

    def _is_hard_or_soft_invalidated(self) -> bool:
        return (
            self.dbapi_connection is None
            or self.__pool._invalidate_time > self.starttime
            or (self._soft_invalidate_time > self.starttime)
        )

    def __close(self, *, terminate: bool = False) -> None:
        self.finalize_callback.clear()
        if self.__pool.dispatch.close:
            self.__pool.dispatch.close(self.dbapi_connection, self)
        assert self.dbapi_connection is not None
        self.__pool._close_connection(
            self.dbapi_connection, terminate=terminate
        )
        self.dbapi_connection = None

    def __connect(self) -> None:
        pool = self.__pool

        # ensure any existing connection is removed, so that if
        # creator fails, this attribute stays None
        self.dbapi_connection = None
        try:
            self.starttime = time.time()
            self.dbapi_connection = connection = pool._invoke_creator(self)
            pool.logger.debug("Created new connection %r", connection)
            self.fresh = True
        except BaseException as e:
            with util.safe_reraise():
                pool.logger.debug("Error on connect(): %s", e)
        else:
            # in SQLAlchemy 1.4 the first_connect event is not used by
            # the engine, so this will usually not be set
            if pool.dispatch.first_connect:
                pool.dispatch.first_connect.for_modify(
                    pool.dispatch
                ).exec_once_unless_exception(self.dbapi_connection, self)

            # init of the dialect now takes place within the connect
            # event, so ensure a mutex is used on the first run
            pool.dispatch.connect.for_modify(
                pool.dispatch
            )._exec_w_sync_on_first_run(self.dbapi_connection, self)


def _finalize_fairy(
    dbapi_connection: Optional[DBAPIConnection],
    connection_record: Optional[_ConnectionRecord],
    pool: Pool,
    ref: Optional[
        weakref.ref[_ConnectionFairy]
    ],  # this is None when called directly, not by the gc
    echo: Optional[log._EchoFlagType],
    transaction_was_reset: bool = False,
    fairy: Optional[_ConnectionFairy] = None,
) -> None:
    """Cleanup for a :class:`._ConnectionFairy` whether or not it's already
    been garbage collected.

    When using an async dialect no IO can happen here (without using
    a dedicated thread), since this is called outside the greenlet
    context and with an already running loop. In this case function
    will only log a message and raise a warning.
    """

    is_gc_cleanup = ref is not None

    if is_gc_cleanup:
        assert ref is not None
        _strong_ref_connection_records.pop(ref, None)
        assert connection_record is not None
        if connection_record.fairy_ref is not ref:
            return
        assert dbapi_connection is None
        dbapi_connection = connection_record.dbapi_connection

    elif fairy:
        _strong_ref_connection_records.pop(weakref.ref(fairy), None)

    # null pool is not _is_asyncio but can be used also with async dialects
    dont_restore_gced = pool._dialect.is_async

    if dont_restore_gced:
        detach = connection_record is None or is_gc_cleanup
        can_manipulate_connection = not is_gc_cleanup
        can_close_or_terminate_connection = (
            not pool._dialect.is_async or pool._dialect.has_terminate
        )
        requires_terminate_for_close = (
            pool._dialect.is_async and pool._dialect.has_terminate
        )

    else:
        detach = connection_record is None
        can_manipulate_connection = can_close_or_terminate_connection = True
        requires_terminate_for_close = False

    if dbapi_connection is not None:
        if connection_record and echo:
            pool.logger.debug(
                "Connection %r being returned to pool", dbapi_connection
            )

        try:
            if not fairy:
                assert connection_record is not None
                fairy = _ConnectionFairy(
                    pool,
                    dbapi_connection,
                    connection_record,
                    echo,
                )
            assert fairy.dbapi_connection is dbapi_connection

            fairy._reset(
                pool,
                transaction_was_reset=transaction_was_reset,
                terminate_only=detach,
                asyncio_safe=can_manipulate_connection,
            )

            if detach:
                if connection_record:
                    fairy._pool = pool
                    fairy.detach()

                if can_close_or_terminate_connection:
                    if pool.dispatch.close_detached:
                        pool.dispatch.close_detached(dbapi_connection)

                    pool._close_connection(
                        dbapi_connection,
                        terminate=requires_terminate_for_close,
                    )

        except BaseException as e:
            pool.logger.error(
                "Exception during reset or similar", exc_info=True
            )
            if connection_record:
                connection_record.invalidate(e=e)
            if not isinstance(e, Exception):
                raise
        finally:
            if detach and is_gc_cleanup and dont_restore_gced:
                message = (
                    "The garbage collector is trying to clean up "
                    f"non-checked-in connection {dbapi_connection!r}, "
                    f"""which will be {
                        'dropped, as it cannot be safely terminated'
                        if not can_close_or_terminate_connection
                        else 'terminated'
                    }.  """
                    "Please ensure that SQLAlchemy pooled connections are "
                    "returned to "
                    "the pool explicitly, either by calling ``close()`` "
                    "or by using appropriate context managers to manage "
                    "their lifecycle."
                )
                pool.logger.error(message)
                util.warn(message)

    if connection_record and connection_record.fairy_ref is not None:
        connection_record.checkin()

    # give gc some help.  See
    # test/engine/test_pool.py::PoolEventsTest::test_checkin_event_gc[True]
    # which actually started failing when pytest warnings plugin was
    # turned on, due to util.warn() above
    if fairy is not None:
        fairy.dbapi_connection = None  # type: ignore
        fairy._connection_record = None
    del dbapi_connection
    del connection_record
    del fairy


# a dictionary of the _ConnectionFairy weakrefs to _ConnectionRecord, so that
# GC under pypy will call ConnectionFairy finalizers.  linked directly to the
# weakref that will empty itself when collected so that it should not create
# any unmanaged memory references.
_strong_ref_connection_records: Dict[
    weakref.ref[_ConnectionFairy], _ConnectionRecord
] = {}


class PoolProxiedConnection(ManagesConnection):
    """A connection-like adapter for a :pep:`249` DBAPI connection, which
    includes additional methods specific to the :class:`.Pool` implementation.

    :class:`.PoolProxiedConnection` is the public-facing interface for the
    internal :class:`._ConnectionFairy` implementation object; users familiar
    with :class:`._ConnectionFairy` can consider this object to be equivalent.

    .. versionadded:: 2.0  :class:`.PoolProxiedConnection` provides the public-
       facing interface for the :class:`._ConnectionFairy` internal class.

    """

    __slots__ = ()

    if typing.TYPE_CHECKING:

        def commit(self) -> None: ...

        def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor: ...

        def rollback(self) -> None: ...

    @property
    def is_valid(self) -> bool:
        """Return True if this :class:`.PoolProxiedConnection` still refers
        to an active DBAPI connection."""

        raise NotImplementedError()

    @property
    def is_detached(self) -> bool:
        """Return True if this :class:`.PoolProxiedConnection` is detached
        from its pool."""

        raise NotImplementedError()

    def detach(self) -> None:
        """Separate this connection from its Pool.

        This means that the connection will no longer be returned to the
        pool when closed, and will instead be literally closed.  The
        associated :class:`.ConnectionPoolEntry` is de-associated from this
        DBAPI connection.

        Note that any overall connection limiting constraints imposed by a
        Pool implementation may be violated after a detach, as the detached
        connection is removed from the pool's knowledge and control.

        """

        raise NotImplementedError()

    def close(self) -> None:
        """Release this connection back to the pool.

        The :meth:`.PoolProxiedConnection.close` method shadows the
        :pep:`249` ``.close()`` method, altering its behavior to instead
        :term:`release` the proxied connection back to the connection pool.

        Upon release to the pool, whether the connection stays "opened" and
        pooled in the Python process, versus actually closed out and removed
        from the Python process, is based on the pool implementation in use and
        its configuration and current state.

        """
        raise NotImplementedError()


class _AdhocProxiedConnection(PoolProxiedConnection):
    """provides the :class:`.PoolProxiedConnection` interface for cases where
    the DBAPI connection is not actually proxied.

    This is used by the engine internals to pass a consistent
    :class:`.PoolProxiedConnection` object to consuming dialects in response to
    pool events that may not always have the :class:`._ConnectionFairy`
    available.

    """

    __slots__ = ("dbapi_connection", "_connection_record", "_is_valid")

    dbapi_connection: DBAPIConnection
    _connection_record: ConnectionPoolEntry

    def __init__(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ):
        self.dbapi_connection = dbapi_connection
        self._connection_record = connection_record
        self._is_valid = True

    @property
    def driver_connection(self) -> Any:  # type: ignore[override]  # mypy#4125
        return self._connection_record.driver_connection

    @property
    def connection(self) -> DBAPIConnection:
        return self.dbapi_connection

    @property
    def is_valid(self) -> bool:
        """Implement is_valid state attribute.

        for the adhoc proxied connection it's assumed the connection is valid
        as there is no "invalidate" routine.

        """
        return self._is_valid

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        self._is_valid = False

    @util.ro_non_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        return self._connection_record.record_info

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor:
        return self.dbapi_connection.cursor(*args, **kwargs)

    def __getattr__(self, key: Any) -> Any:
        return getattr(self.dbapi_connection, key)


class _ConnectionFairy(PoolProxiedConnection):
    """Proxies a DBAPI connection and provides return-on-dereference
    support.

    This is an internal object used by the :class:`_pool.Pool` implementation
    to provide context management to a DBAPI connection delivered by
    that :class:`_pool.Pool`.   The public facing interface for this class
    is described by the :class:`.PoolProxiedConnection` class.  See that
    class for public API details.

    The name "fairy" is inspired by the fact that the
    :class:`._ConnectionFairy` object's lifespan is transitory, as it lasts
    only for the length of a specific DBAPI connection being checked out from
    the pool, and additionally that as a transparent proxy, it is mostly
    invisible.

    .. seealso::

        :class:`.PoolProxiedConnection`

        :class:`.ConnectionPoolEntry`


    """

    __slots__ = (
        "dbapi_connection",
        "_connection_record",
        "_echo",
        "_pool",
        "_counter",
        "__weakref__",
        "__dict__",
    )

    pool: Pool
    dbapi_connection: DBAPIConnection
    _echo: log._EchoFlagType

    def __init__(
        self,
        pool: Pool,
        dbapi_connection: DBAPIConnection,
        connection_record: _ConnectionRecord,
        echo: log._EchoFlagType,
    ):
        self._pool = pool
        self._counter = 0
        self.dbapi_connection = dbapi_connection
        self._connection_record = connection_record
        self._echo = echo

    _connection_record: Optional[_ConnectionRecord]

    @property
    def driver_connection(self) -> Optional[Any]:  # type: ignore[override]  # mypy#4125  # noqa: E501
        if self._connection_record is None:
            return None
        return self._connection_record.driver_connection

    @property
    @util.deprecated(
        "2.0",
        "The _ConnectionFairy.connection attribute is deprecated; "
        "please use 'driver_connection'",
    )
    def connection(self) -> DBAPIConnection:
        return self.dbapi_connection

    @classmethod
    def _checkout(
        cls,
        pool: Pool,
        threadconns: Optional[threading.local] = None,
        fairy: Optional[_ConnectionFairy] = None,
    ) -> _ConnectionFairy:
        if not fairy:
            fairy = _ConnectionRecord.checkout(pool)

            if threadconns is not None:
                threadconns.current = weakref.ref(fairy)

        assert (
            fairy._connection_record is not None
        ), "can't 'checkout' a detached connection fairy"
        assert (
            fairy.dbapi_connection is not None
        ), "can't 'checkout' an invalidated connection fairy"

        fairy._counter += 1
        if (
            not pool.dispatch.checkout and not pool._pre_ping
        ) or fairy._counter != 1:
            return fairy

        # Pool listeners can trigger a reconnection on checkout, as well
        # as the pre-pinger.
        # there are three attempts made here, but note that if the database
        # is not accessible from a connection standpoint, those won't proceed
        # here.

        attempts = 2

        while attempts > 0:
            connection_is_fresh = fairy._connection_record.fresh
            fairy._connection_record.fresh = False
            try:
                if pool._pre_ping:
                    if not connection_is_fresh:
                        if fairy._echo:
                            pool.logger.debug(
                                "Pool pre-ping on connection %s",
                                fairy.dbapi_connection,
                            )
                        result = pool._dialect._do_ping_w_event(
                            fairy.dbapi_connection
                        )
                        if not result:
                            if fairy._echo:
                                pool.logger.debug(
                                    "Pool pre-ping on connection %s failed, "
                                    "will invalidate pool",
                                    fairy.dbapi_connection,
                                )
                            raise exc.InvalidatePoolError()
                    elif fairy._echo:
                        pool.logger.debug(
                            "Connection %s is fresh, skipping pre-ping",
                            fairy.dbapi_connection,
                        )

                pool.dispatch.checkout(
                    fairy.dbapi_connection, fairy._connection_record, fairy
                )
                return fairy
            except exc.DisconnectionError as e:
                if e.invalidate_pool:
                    pool.logger.info(
                        "Disconnection detected on checkout, "
                        "invalidating all pooled connections prior to "
                        "current timestamp (reason: %r)",
                        e,
                    )
                    fairy._connection_record.invalidate(e)
                    pool._invalidate(fairy, e, _checkin=False)
                else:
                    pool.logger.info(
                        "Disconnection detected on checkout, "
                        "invalidating individual connection %s (reason: %r)",
                        fairy.dbapi_connection,
                        e,
                    )
                    fairy._connection_record.invalidate(e)
                try:
                    fairy.dbapi_connection = (
                        fairy._connection_record.get_connection()
                    )
                except BaseException as err:
                    with util.safe_reraise():
                        fairy._connection_record._checkin_failed(
                            err,
                            _fairy_was_created=True,
                        )

                        # prevent _ConnectionFairy from being carried
                        # in the stack trace.  Do this after the
                        # connection record has been checked in, so that
                        # if the del triggers a finalize fairy, it won't
                        # try to checkin a second time.
                        del fairy

                    # never called, this is for code linters
                    raise

                attempts -= 1
            except BaseException as be_outer:
                with util.safe_reraise():
                    rec = fairy._connection_record
                    if rec is not None:
                        rec._checkin_failed(
                            be_outer,
                            _fairy_was_created=True,
                        )

                    # prevent _ConnectionFairy from being carried
                    # in the stack trace, see above
                    del fairy

                # never called, this is for code linters
                raise

        pool.logger.info("Reconnection attempts exhausted on checkout")
        fairy.invalidate()
        raise exc.InvalidRequestError("This connection is closed")

    def _checkout_existing(self) -> _ConnectionFairy:
        return _ConnectionFairy._checkout(self._pool, fairy=self)

    def _checkin(self, transaction_was_reset: bool = False) -> None:
        _finalize_fairy(
            self.dbapi_connection,
            self._connection_record,
            self._pool,
            None,
            self._echo,
            transaction_was_reset=transaction_was_reset,
            fairy=self,
        )

    def _close(self) -> None:
        self._checkin()

    def _reset(
        self,
        pool: Pool,
        transaction_was_reset: bool,
        terminate_only: bool,
        asyncio_safe: bool,
    ) -> None:
        if pool.dispatch.reset:
            pool.dispatch.reset(
                self.dbapi_connection,
                self._connection_record,
                PoolResetState(
                    transaction_was_reset=transaction_was_reset,
                    terminate_only=terminate_only,
                    asyncio_safe=asyncio_safe,
                ),
            )

        if not asyncio_safe:
            return

        if pool._reset_on_return is reset_rollback:
            if transaction_was_reset:
                if self._echo:
                    pool.logger.debug(
                        "Connection %s reset, transaction already reset",
                        self.dbapi_connection,
                    )
            else:
                if self._echo:
                    pool.logger.debug(
                        "Connection %s rollback-on-return",
                        self.dbapi_connection,
                    )
                pool._dialect.do_rollback(self)
        elif pool._reset_on_return is reset_commit:
            if self._echo:
                pool.logger.debug(
                    "Connection %s commit-on-return",
                    self.dbapi_connection,
                )
            pool._dialect.do_commit(self)

    @property
    def _logger(self) -> log._IdentifiedLoggerType:
        return self._pool.logger

    @property
    def is_valid(self) -> bool:
        return self.dbapi_connection is not None

    @property
    def is_detached(self) -> bool:
        return self._connection_record is None

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        if self._connection_record is None:
            return {}
        else:
            return self._connection_record.info

    @util.ro_non_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        if self._connection_record is None:
            return None
        else:
            return self._connection_record.record_info

    def invalidate(
        self, e: Optional[BaseException] = None, soft: bool = False
    ) -> None:
        if self.dbapi_connection is None:
            util.warn("Can't invalidate an already-closed connection.")
            return
        if self._connection_record:
            self._connection_record.invalidate(e=e, soft=soft)
        if not soft:
            # prevent any rollback / reset actions etc. on
            # the connection
            self.dbapi_connection = None  # type: ignore

            # finalize
            self._checkin()

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor:
        assert self.dbapi_connection is not None
        return self.dbapi_connection.cursor(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.dbapi_connection, key)

    def detach(self) -> None:
        if self._connection_record is not None:
            rec = self._connection_record
            rec.fairy_ref = None
            rec.dbapi_connection = None
            # TODO: should this be _return_conn?
            self._pool._do_return_conn(self._connection_record)

            # can't get the descriptor assignment to work here
            # in pylance.  mypy is OK w/ it
            self.info = self.info.copy()  # type: ignore

            self._connection_record = None

            if self._pool.dispatch.detach:
                self._pool.dispatch.detach(self.dbapi_connection, rec)

    def close(self) -> None:
        self._counter -= 1
        if self._counter == 0:
            self._checkin()

    def _close_special(self, transaction_reset: bool = False) -> None:
        self._counter -= 1
        if self._counter == 0:
            self._checkin(transaction_was_reset=transaction_reset)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_arrayterator.py ===
from functools import reduce
from operator import mul

import numpy as np
from numpy.lib import Arrayterator
from numpy.random import randint
from numpy.testing import assert_


def test():
    np.random.seed(np.arange(10))

    # Create a random array
    ndims = randint(5) + 1
    shape = tuple(randint(10) + 1 for dim in range(ndims))
    els = reduce(mul, shape)
    a = np.arange(els)
    a.shape = shape

    buf_size = randint(2 * els)
    b = Arrayterator(a, buf_size)

    # Check that each block has at most ``buf_size`` elements
    for block in b:
        assert_(len(block.flat) <= (buf_size or els))

    # Check that all elements are iterated correctly
    assert_(list(b.flat) == list(a.flat))

    # Slice arrayterator
    start = [randint(dim) for dim in shape]
    stop = [randint(dim) + 1 for dim in shape]
    step = [randint(dim) + 1 for dim in shape]
    slice_ = tuple(slice(*t) for t in zip(start, stop, step))
    c = b[slice_]
    d = a[slice_]

    # Check that each block has at most ``buf_size`` elements
    for block in c:
        assert_(len(block.flat) <= (buf_size or els))

    # Check that the arrayterator is sliced correctly
    assert_(np.all(c.__array__() == d))

    # Check that all elements are iterated correctly
    assert_(list(c.flat) == list(d.flat))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\einsumfunc.py ===
"""
Implementation of optimized einsum.

"""
import itertools
import operator

from numpy._core.multiarray import c_einsum
from numpy._core.numeric import asanyarray, tensordot
from numpy._core.overrides import array_function_dispatch

__all__ = ['einsum', 'einsum_path']

# importing string for string.ascii_letters would be too slow
# the first import before caching has been measured to take 800 µs (#23777)
# imports begin with uppercase to mimic ASCII values to avoid sorting issues
einsum_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
einsum_symbols_set = set(einsum_symbols)


def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """
    Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> _flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> _flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """

    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor

def _compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------

    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')
    >>> _find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> _find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)

    return (new_result, remaining, idx_removed, idx_contract)


def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set()
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _optimal_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []

        # Compute all unique pairs
        for curr in full_results:
            cost, positions, remaining = curr
            for con in itertools.combinations(
                range(len(input_sets) - iteration), 2
            ):

                # Find the contraction
                cont = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = cont

                # Sieve the results based on memory_limit
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Build (total_cost, positions, indices_remaining)
                total_cost = cost + _flop_count(
                    idx_contract, idx_removed, len(con), idx_dict
                )
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))

        # Update combinatorial list, if we did not find anything return best
        # path + remaining contractions
        if iter_results:
            full_results = iter_results
        else:
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path

    # If we have not found anything return single einsum contraction
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]

    path = min(full_results, key=lambda x: x[0])[1]
    return path

def _parse_possible_contraction(
        positions, input_sets, output_set, idx_dict,
        memory_limit, path_cost, naive_cost
    ):
    """Compute the cost (removed size + flops) and resultant indices for
    performing the contraction specified by ``positions``.

    Parameters
    ----------
    positions : tuple of int
        The locations of the proposed tensors to contract.
    input_sets : list of sets
        The indices found on each tensors.
    output_set : set
        The output indices of the expression.
    idx_dict : dict
        Mapping of each index to its size.
    memory_limit : int
        The total allowed size for an intermediary tensor.
    path_cost : int
        The contraction cost so far.
    naive_cost : int
        The cost of the unoptimized expression.

    Returns
    -------
    cost : (int, int)
        A tuple containing the size of any indices removed, and the flop cost.
    positions : tuple of int
        The locations of the proposed tensors to contract.
    new_input_sets : list of sets
        The resulting new list of indices if this proposed contraction
        is performed.

    """

    # Find the contraction
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract

    # Sieve the results based on memory_limit
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None

    # Build sort tuple
    old_sizes = (
        _compute_size_by_dict(input_sets[p], idx_dict) for p in positions
    )
    removed_size = sum(old_sizes) - new_size

    # NB: removed_size used to be just the size of any removed indices i.e.:
    #     helpers.compute_size_by_dict(idx_removed, idx_dict)
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)

    # Sieve based on total cost as well
    if (path_cost + cost) > naive_cost:
        return None

    # Add contraction to possible choices
    return [sort, positions, new_input_sets]


def _update_other_results(results, best):
    """Update the positions and provisional input_sets of ``results``
    based on performing the contraction result ``best``. Remove any
    involving the tensors contracted.

    Parameters
    ----------
    results : list
        List of contraction results produced by
        ``_parse_possible_contraction``.
    best : list
        The best contraction of ``results`` i.e. the one that
        will be performed.

    Returns
    -------
    mod_results : list
        The list of modified results, updated with outcome of
        ``best`` contraction.
    """

    best_con = best[1]
    bx, by = best_con
    mod_results = []

    for cost, (x, y), con_sets in results:

        # Ignore results involving tensors just contracted
        if x in best_con or y in best_con:
            continue

        # Update the input_sets
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])

        # Update the position indices
        mod_con = x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
        mod_results.append((cost, mod_con, con_sets))

    return mod_results

def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the
    number of elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The greedy contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set()
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _greedy_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    # Handle trivial cases that leaked through
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]

    # Build up a naive cost
    contract = _find_contraction(
        range(len(input_sets)), input_sets, output_set
    )
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = _flop_count(
        idx_contract, idx_removed, len(input_sets), idx_dict
    )

    # Initially iterate over all pairs
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    known_contractions = []

    path_cost = 0
    path = []

    for iteration in range(len(input_sets) - 1):

        # Iterate over all pairs on the first step, only previously
        # found pairs on subsequent steps
        for positions in comb_iter:

            # Always initially ignore outer products
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue

            result = _parse_possible_contraction(
                positions, input_sets, output_set, idx_dict,
                memory_limit, path_cost, naive_cost
            )
            if result is not None:
                known_contractions.append(result)

        # If we do not have a inner contraction, rescan pairs
        # including outer products
        if len(known_contractions) == 0:

            # Then check the outer products
            for positions in itertools.combinations(
                range(len(input_sets)), 2
            ):
                result = _parse_possible_contraction(
                    positions, input_sets, output_set, idx_dict,
                    memory_limit, path_cost, naive_cost
                )
                if result is not None:
                    known_contractions.append(result)

            # If we still did not find any remaining contractions,
            # default back to einsum like behavior
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break

        # Sort based on first index
        best = min(known_contractions, key=lambda x: x[0])

        # Now propagate as many unused contractions as possible
        # to the next iteration
        known_contractions = _update_other_results(known_contractions, best)

        # Next iteration only compute contractions with the new tensor
        # All other contractions have been accounted for
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))

        # Update path and total cost
        path.append(best[1])
        path_cost += best[0][1]

    return path


def _can_dot(inputs, result, idx_removed):
    """
    Checks if we can use BLAS (np.tensordot) call and its beneficial to do so.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation


    Returns
    -------
    type : bool
        Returns true if BLAS should and can be used, else False

    Notes
    -----
    If the operations is BLAS level 1 or 2 and is not already aligned
    we default back to einsum as the memory movement to copy is more
    costly than the operation itself.


    Examples
    --------

    # Standard GEMM operation
    >>> _can_dot(['ij', 'jk'], 'ik', set('j'))
    True

    # Can use the standard BLAS, but requires odd data movement
    >>> _can_dot(['ijj', 'jk'], 'ik', set('j'))
    False

    # DDOT where the memory is not aligned
    >>> _can_dot(['ijk', 'ikj'], '', set('ijk'))
    False

    """

    # All `dot` calls remove indices
    if len(idx_removed) == 0:
        return False

    # BLAS can only handle two operands
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # Build a few temporaries
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)

    # At this point we are a DOT, GEMV, or GEMM operation

    # Handle inner products

    # DDOT with aligned data
    if input_left == input_right:
        return True

    # DDOT without aligned data (better to use einsum)
    if set_left == set_right:
        return False

    # Handle the 4 possible (aligned) GEMV or GEMM cases

    # GEMM or GEMV no transpose
    if input_left[-rs:] == input_right[:rs]:
        return True

    # GEMM or GEMV transpose both
    if input_left[:rs] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose right
    if input_left[-rs:] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose left
    if input_left[:rs] == input_right[:rs]:
        return True

    # Einsum is faster than GEMV if we have to copy data
    if not keep_left or not keep_right:
        return False

    # We are a matrix-matrix product, but we need to copy data
    return True


def _parse_einsum_input(operands):
    """
    A reproduction of einsum c side einsum parsing in python.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> np.random.seed(123)
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b]) # may vary

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b]) # may vary
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [asanyarray(v) for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError(f"Character {s} is not a valid symbol.")

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError(f"Character {s} is not a valid symbol.")
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError(f"Character {s} is not a valid symbol.")
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError("Output character %s appeared more than once in "
                             "the output." % char)
        if char not in input_subscripts:
            raise ValueError(f"Output character {char} did not appear in the input")

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return (input_subscripts, output_subscript, operands)


def _einsum_path_dispatcher(*operands, optimize=None, einsum_call=None):
    # NOTE: technically, we should only dispatch on array-like arguments, not
    # subscripts (given as strings). But separating operands into
    # arrays/subscripts is a little tricky/slow (given einsum's two supported
    # signatures), so as a practical shortcut we dispatch on everything.
    # Strings will be ignored for dispatching since they don't define
    # __array_function__.
    return operands


@array_function_dispatch(_einsum_path_dispatcher, module='numpy')
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    optimize : {bool, list, tuple, 'greedy', 'optimal'}
        Choose the type of path. If a tuple is provided, the second argument is
        assumed to be the maximum intermediate size created. If only a single
        argument is provided the largest input or output array size is used
        as a maximum intermediate size.

        * if a list is given that starts with ``einsum_path``, uses this as the
          contraction path
        * if False no optimization is taken
        * if True defaults to the 'greedy' algorithm
        * 'optimal' An algorithm that combinatorially explores all possible
          ways of contracting the listed tensors and chooses the least costly
          path. Scales exponentially with the number of terms in the
          contraction.
        * 'greedy' An algorithm that chooses the best pair contraction
          at each step. Effectively, this algorithm searches the largest inner,
          Hadamard, and then outer products at each step. Scales cubically with
          the number of terms in the contraction. Equivalent to the 'optimal'
          path for most contractions.

        Default is 'greedy'.

    Returns
    -------
    path : list of tuples
        A list representation of the einsum path.
    string_repr : str
        A printable representation of the einsum path.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the
    end of the contraction list. This list can then be iterated over until all
    intermediate contractions are complete.

    See Also
    --------
    einsum, linalg.multi_dot

    Examples
    --------

    We can begin with a chain dot example. In this case, it is optimal to
    contract the ``b`` and ``c`` tensors first as represented by the first
    element of the path ``(1, 2)``. The resulting tensor is added to the end
    of the contraction and the remaining contraction ``(0, 1)`` is then
    completed.

    >>> np.random.seed(123)
    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    >>> print(path_info[0])
    ['einsum_path', (1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il # may vary
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
    ...                            optimize='greedy')

    >>> print(path_info[0])
    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh # may vary
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """

    # Figure out what the path really is
    path_type = optimize
    if path_type is True:
        path_type = 'greedy'
    if path_type is None:
        path_type = False

    explicit_einsum_path = False
    memory_limit = None

    # No optimization or a named path algorithm
    if (path_type is False) or isinstance(path_type, str):
        pass

    # Given an explicit path
    elif len(path_type) and (path_type[0] == 'einsum_path'):
        explicit_einsum_path = True

    # Path tuple with memory limit
    elif ((len(path_type) == 2) and isinstance(path_type[0], str) and
            isinstance(path_type[1], (int, float))):
        memory_limit = int(path_type[1])
        path_type = path_type[0]

    else:
        raise TypeError(f"Did not understand the path: {str(path_type)}")

    # Hidden option, only einsum should call this
    einsum_call_arg = einsum_call

    # Python side parsing
    input_subscripts, output_subscript, operands = (
        _parse_einsum_input(operands)
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d."
                             % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)

            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim

    # Convert broadcast inds to sets
    broadcast_indices = [set(x) for x in broadcast_indices]

    # Compute size of each input array plus the output array
    size_list = [_compute_size_by_dict(term, dimension_dict)
                 for term in input_list + [output_subscript]]
    max_size = max(size_list)

    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit

    # Compute naive cost
    # This isn't quite right, need to look into exactly how einsum does this
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = _flop_count(
        indices, inner_product, len(input_list), dimension_dict
    )

    # Compute the path
    if explicit_einsum_path:
        path = path_type[1:]
    elif (
        (path_type is False)
        or (len(input_list) in [1, 2])
        or (indices == output_set)
    ):
        # Nothing to be optimized, leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type == "greedy":
        path = _greedy_path(
            input_sets, output_set, dimension_dict, memory_arg
        )
    elif path_type == "optimal":
        path = _optimal_path(
            input_sets, output_set, dimension_dict, memory_arg
        )
    else:
        raise KeyError("Path name %s not found", path_type)

    cost_list, scale_list, size_list, contraction_list = [], [], [], []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        cost = _flop_count(
            idx_contract, idx_removed, len(contract_inds), dimension_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)

        new_bcast_inds = bcast - idx_removed

        # If we're broadcasting, nix blas
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        contraction = (
            contract_inds, idx_removed, einsum_str, input_list[:], do_blas
        )
        contraction_list.append(contraction)

    opt_cost = sum(cost_list) + 1

    if len(input_list) != 1:
        # Explicit "einsum_path" is usually trusted, but we detect this kind of
        # mistake in order to prevent from returning an intermediate value.
        raise RuntimeError(
            f"Invalid einsum_path is specified: {len(input_list) - 1} more "
            "operands has to be contracted.")

    if einsum_call_arg:
        return (operands, contraction_list)

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "current", "remaining")

    speedup = naive_cost / opt_cost
    max_i = max(size_list)

    path_print = f"  Complete contraction:  {overall_contraction}\n"
    path_print += f"         Naive scaling:  {len(indices)}\n"
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += f"      Naive FLOP count:  {naive_cost:.3e}\n"
    path_print += f"  Optimized FLOP count:  {opt_cost:.3e}\n"
    path_print += f"   Theoretical speedup:  {speedup:3.3f}\n"
    path_print += f"  Largest intermediate:  {max_i:.3e} elements\n"
    path_print += "-" * 74 + "\n"
    path_print += "%6s %24s %40s\n" % header
    path_print += "-" * 74

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += "\n%4d    %24s %40s" % path_run

    path = ['einsum_path'] + path
    return (path, path_print)


def _einsum_dispatcher(*operands, out=None, optimize=None, **kwargs):
    # Arguably we dispatch on more arguments than we really should; see note in
    # _einsum_path_dispatcher for why.
    yield from operands
    yield out


# Rewrite einsum to handle different cases
@array_function_dispatch(_einsum_dispatcher, module='numpy')
def einsum(*operands, out=None, optimize=False, **kwargs):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe', optimize=False)

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : {data-type, None}, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Defaults to False.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot
    einsum:
        Similar verbose interface is provided by the
        `einops <https://github.com/arogozhnikov/einops>`_ package to cover
        additional operations: transpose, reshape/flatten, repeat/tile,
        squeeze/unsqueeze and reductions.
        The `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_
        optimizes contraction order for einsum-like expressions
        in backend-agnostic manner.

    Notes
    -----
    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul`
        :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner`
        :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication,
        :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order,
        :py:func:`numpy.einsum_path`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
    appears only once, it is not summed, so ``np.einsum('i', a)``
    produces a view of ``a`` with no changes. A further example
    ``np.einsum('ij,jk', a, b)`` describes traditional matrix multiplication
    and is equivalent to :py:func:`np.matmul(a,b) <numpy.matmul>`.
    Repeated subscript labels in one operand take the diagonal.
    For example, ``np.einsum('ii', a)`` is equivalent to
    :py:func:`np.trace(a) <numpy.trace>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    ``np.einsum('ji', a)`` takes its transpose. Additionally,
    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``np.einsum('i->', a)`` is like :py:func:`np.sum(a) <numpy.sum>`
    if ``a`` is a 1-D array, and ``np.einsum('ii->i', a)``
    is like :py:func:`np.diag(a) <numpy.diag>` if ``a`` is a square 2-D array.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``np.einsum('...ii->...i', a)``.
    ``np.einsum('...i->...', a)`` is like
    :py:func:`np.sum(a, axis=-1) <numpy.sum>` for array ``a`` of any shape.
    To take the trace along the first and last axes,
    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
    produces a view (changed in version 1.10.0).

    `einsum` also provides an alternative way to provide the subscripts and
    operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.

    Views returned from einsum are now writeable whenever the input array
    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
    have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
    and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
    of a 2D array.

    Added the ``optimize`` argument which will optimize the contraction order
    of an einsum expression. For a contraction with three or more operands
    this can greatly increase the computational efficiency at the cost of
    a larger memory footprint during computation.

    Typically a 'greedy' algorithm is applied which empirical tests have shown
    returns the optimal path in the majority of cases. In some cases 'optimal'
    will return the superlative path through a more expensive, exhaustive
    search. For iterative calculations it may be advisable to calculate
    the optimal path once and reuse that path by supplying it as an argument.
    An example is given below.

    See :py:func:`numpy.einsum_path` for more details.

    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60

    Extract the diagonal (requires explicit form):

    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional arrays summing a single axis can be done
    with ellipsis:

    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Vector inner products:

    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30

    Matrix vector multiplication:

    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])

    Broadcasting and scalar multiplication:

    >>> np.einsum('..., ...', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(',ij', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum('i,j', np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2)+1, [0], b, [1])
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.tensordot(a,b, axes=([1,0],[0,1]))
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    Writeable returned arrays (since version 1.10.0):

    >>> a = np.zeros((3, 3))
    >>> np.einsum('ii->i', a)[:] = 1
    >>> a
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> np.einsum('ki,jk->ij', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('ki,...k->i...', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('k...,jk', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])

    Chained array operations. For more complicated contractions, speed ups
    might be achieved by repeatedly computing a 'greedy' path or pre-computing
    the 'optimal' path and repeatedly applying it, using an `einsum_path`
    insertion (since version 1.12.0). Performance improvements can be
    particularly significant with larger arrays:

    >>> a = np.ones(64).reshape(2,4,8)

    Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)

    Sub-optimal `einsum` (due to repeated path calculation time): ~330ms

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
    ...         optimize='optimal')

    Greedy `einsum` (faster optimal path approximation): ~160ms

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')

    Optimal `einsum` (best usage pattern in some use cases): ~110ms

    >>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
    ...     optimize='optimal')[0]
    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)

    """
    # Special handling if out is specified
    specified_out = out is not None

    # If no optimization, run pure einsum
    if optimize is False:
        if specified_out:
            kwargs['out'] = out
        return c_einsum(*operands, **kwargs)

    # Check the kwargs to avoid a more cryptic error later, without having to
    # repeat default values here
    valid_einsum_kwargs = ['dtype', 'order', 'casting']
    unknown_kwargs = [k for (k, v) in kwargs.items() if
                      k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError(f"Did not understand the following kwargs: {unknown_kwargs}")

    # Build the contraction list and operand
    operands, contraction_list = einsum_path(*operands, optimize=optimize,
                                             einsum_call=True)

    # Handle order kwarg for output array, c_einsum allows mixed case
    output_order = kwargs.pop('order', 'K')
    if output_order.upper() == 'A':
        if all(arr.flags.f_contiguous for arr in operands):
            output_order = 'F'
        else:
            output_order = 'C'

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # Call tensordot if still possible
        if blas:
            # Checks have already been handled
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")

            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = tensordot(
                *tmp_operands, axes=(tuple(left_pos), tuple(right_pos))
            )

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:
                if handle_out:
                    kwargs["out"] = out
                new_view = c_einsum(
                    tensor_result + '->' + results_index, new_view, **kwargs
                )

        # Call einsum
        else:
            # If out was specified
            if handle_out:
                kwargs["out"] = out

            # Do the contraction
            new_view = c_einsum(einsum_str, *tmp_operands, **kwargs)

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out
    else:
        return asanyarray(operands[0], order=output_order)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\tests.py ===
#!/usr/bin/env python
# vim: fileencoding=utf-8 :

# Tests for the `humanfriendly' package.
#
# Author: Peter Odding <peter.odding@paylogic.eu>
# Last Change: June 11, 2021
# URL: https://humanfriendly.readthedocs.io

"""Test suite for the `humanfriendly` package."""

# Standard library modules.
import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings

# Modules included in our package.
from humanfriendly import (
    InvalidDate,
    InvalidLength,
    InvalidSize,
    InvalidTimespan,
    Timer,
    coerce_boolean,
    coerce_pattern,
    format_length,
    format_number,
    format_path,
    format_size,
    format_timespan,
    parse_date,
    parse_length,
    parse_path,
    parse_size,
    parse_timespan,
    prompts,
    round_number,
)
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
    TooManyInvalidReplies,
    prompt_for_confirmation,
    prompt_for_choice,
    prompt_for_input,
)
from humanfriendly.sphinx import (
    deprecation_note_callback,
    man_role,
    pypi_role,
    setup,
    special_methods_callback,
    usage_message_callback,
)
from humanfriendly.tables import (
    format_pretty_table,
    format_robust_table,
    format_rst_table,
    format_smart_table,
)
from humanfriendly.terminal import (
    ANSI_CSI,
    ANSI_ERASE_LINE,
    ANSI_HIDE_CURSOR,
    ANSI_RESET,
    ANSI_SGR,
    ANSI_SHOW_CURSOR,
    ansi_strip,
    ansi_style,
    ansi_width,
    ansi_wrap,
    clean_terminal_output,
    connected_to_terminal,
    find_terminal_size,
    get_pager_command,
    message,
    output,
    show_pager,
    terminal_supports_colors,
    warning,
)
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
    CallableTimedOut,
    CaptureOutput,
    MockedProgram,
    PatchedAttribute,
    PatchedItem,
    TemporaryDirectory,
    TestCase,
    retry,
    run_cli,
    skip_on_raise,
    touch,
)
from humanfriendly.text import (
    compact,
    compact_empty_lines,
    concatenate,
    dedent,
    generate_slug,
    pluralize,
    random_string,
    trim_empty_lines,
)
from humanfriendly.usage import (
    find_meta_variables,
    format_usage,
    parse_usage,
    render_usage,
)

# Test dependencies.
from mock import MagicMock


class HumanFriendlyTestCase(TestCase):

    """Container for the `humanfriendly` test suite."""

    def test_case_insensitive_dict(self):
        """Test the CaseInsensitiveDict class."""
        # Test support for the dict(iterable) signature.
        assert len(CaseInsensitiveDict([('key', True), ('KEY', False)])) == 1
        # Test support for the dict(iterable, **kw) signature.
        assert len(CaseInsensitiveDict([('one', True), ('ONE', False)], one=False, two=True)) == 2
        # Test support for the dict(mapping) signature.
        assert len(CaseInsensitiveDict(dict(key=True, KEY=False))) == 1
        # Test support for the dict(mapping, **kw) signature.
        assert len(CaseInsensitiveDict(dict(one=True, ONE=False), one=False, two=True)) == 2
        # Test support for the dict(**kw) signature.
        assert len(CaseInsensitiveDict(one=True, ONE=False, two=True)) == 2
        # Test support for dict.fromkeys().
        obj = CaseInsensitiveDict.fromkeys(["One", "one", "ONE", "Two", "two", "TWO"])
        assert len(obj) == 2
        # Test support for dict.get().
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj.get('Existing_Key') == 42
        # Test support for dict.pop().
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj.pop('Existing_Key') == 42
        assert len(obj) == 0
        # Test support for dict.setdefault().
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj.setdefault('Existing_Key') == 42
        obj.setdefault('other_key', 11)
        assert obj['Other_Key'] == 11
        # Test support for dict.__contains__().
        obj = CaseInsensitiveDict(existing_key=42)
        assert 'Existing_Key' in obj
        # Test support for dict.__delitem__().
        obj = CaseInsensitiveDict(existing_key=42)
        del obj['Existing_Key']
        assert len(obj) == 0
        # Test support for dict.__getitem__().
        obj = CaseInsensitiveDict(existing_key=42)
        assert obj['Existing_Key'] == 42
        # Test support for dict.__setitem__().
        obj = CaseInsensitiveDict(existing_key=42)
        obj['Existing_Key'] = 11
        assert obj['existing_key'] == 11

    def test_case_insensitive_key(self):
        """Test the CaseInsensitiveKey class."""
        # Test the __eq__() special method.
        polite = CaseInsensitiveKey("Please don't shout")
        rude = CaseInsensitiveKey("PLEASE DON'T SHOUT")
        assert polite == rude
        # Test the __hash__() special method.
        mapping = {}
        mapping[polite] = 1
        mapping[rude] = 2
        assert len(mapping) == 1

    def test_capture_output(self):
        """Test the CaptureOutput class."""
        with CaptureOutput() as capturer:
            sys.stdout.write("Something for stdout.\n")
            sys.stderr.write("And for stderr.\n")
            assert capturer.stdout.get_lines() == ["Something for stdout."]
            assert capturer.stderr.get_lines() == ["And for stderr."]

    def test_skip_on_raise(self):
        """Test the skip_on_raise() decorator."""
        def test_fn():
            raise NotImplementedError()
        decorator_fn = skip_on_raise(NotImplementedError)
        decorated_fn = decorator_fn(test_fn)
        self.assertRaises(NotImplementedError, test_fn)
        self.assertRaises(unittest.SkipTest, decorated_fn)

    def test_retry_raise(self):
        """Test :func:`~humanfriendly.testing.retry()` based on assertion errors."""
        # Define a helper function that will raise an assertion error on the
        # first call and return a string on the second call.
        def success_helper():
            if not hasattr(success_helper, 'was_called'):
                setattr(success_helper, 'was_called', True)
                assert False
            else:
                return 'yes'
        assert retry(success_helper) == 'yes'

        # Define a helper function that always raises an assertion error.
        def failure_helper():
            assert False
        with self.assertRaises(AssertionError):
            retry(failure_helper, timeout=1)

    def test_retry_return(self):
        """Test :func:`~humanfriendly.testing.retry()` based on return values."""
        # Define a helper function that will return False on the first call and
        # return a number on the second call.
        def success_helper():
            if not hasattr(success_helper, 'was_called'):
                # On the first call we return False.
                setattr(success_helper, 'was_called', True)
                return False
            else:
                # On the second call we return a number.
                return 42
        assert retry(success_helper) == 42
        with self.assertRaises(CallableTimedOut):
            retry(lambda: False, timeout=1)

    def test_mocked_program(self):
        """Test :class:`humanfriendly.testing.MockedProgram`."""
        name = random_string()
        script = dedent('''
            # This goes to stdout.
            tr a-z A-Z
            # This goes to stderr.
            echo Fake warning >&2
        ''')
        with MockedProgram(name=name, returncode=42, script=script) as directory:
            assert os.path.isdir(directory)
            assert os.path.isfile(os.path.join(directory, name))
            program = subprocess.Popen(name, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = program.communicate(input=b'hello world\n')
            assert program.returncode == 42
            assert stdout == b'HELLO WORLD\n'
            assert stderr == b'Fake warning\n'

    def test_temporary_directory(self):
        """Test :class:`humanfriendly.testing.TemporaryDirectory`."""
        with TemporaryDirectory() as directory:
            assert os.path.isdir(directory)
            temporary_file = os.path.join(directory, 'some-file')
            with open(temporary_file, 'w') as handle:
                handle.write("Hello world!")
        assert not os.path.exists(temporary_file)
        assert not os.path.exists(directory)

    def test_touch(self):
        """Test :func:`humanfriendly.testing.touch()`."""
        with TemporaryDirectory() as directory:
            # Create a file in the temporary directory.
            filename = os.path.join(directory, random_string())
            assert not os.path.isfile(filename)
            touch(filename)
            assert os.path.isfile(filename)
            # Create a file in a subdirectory.
            filename = os.path.join(directory, random_string(), random_string())
            assert not os.path.isfile(filename)
            touch(filename)
            assert os.path.isfile(filename)

    def test_patch_attribute(self):
        """Test :class:`humanfriendly.testing.PatchedAttribute`."""
        class Subject(object):
            my_attribute = 42
        instance = Subject()
        assert instance.my_attribute == 42
        with PatchedAttribute(instance, 'my_attribute', 13) as return_value:
            assert return_value is instance
            assert instance.my_attribute == 13
        assert instance.my_attribute == 42

    def test_patch_item(self):
        """Test :class:`humanfriendly.testing.PatchedItem`."""
        instance = dict(my_item=True)
        assert instance['my_item'] is True
        with PatchedItem(instance, 'my_item', False) as return_value:
            assert return_value is instance
            assert instance['my_item'] is False
        assert instance['my_item'] is True

    def test_run_cli_intercepts_exit(self):
        """Test that run_cli() intercepts SystemExit."""
        returncode, output = run_cli(lambda: sys.exit(42))
        self.assertEqual(returncode, 42)

    def test_run_cli_intercepts_error(self):
        """Test that run_cli() intercepts exceptions."""
        returncode, output = run_cli(self.run_cli_raise_other)
        self.assertEqual(returncode, 1)

    def run_cli_raise_other(self):
        """run_cli() sample that raises an exception."""
        raise ValueError()

    def test_run_cli_intercepts_output(self):
        """Test that run_cli() intercepts output."""
        expected_output = random_string() + "\n"
        returncode, output = run_cli(lambda: sys.stdout.write(expected_output))
        self.assertEqual(returncode, 0)
        self.assertEqual(output, expected_output)

    def test_caching_decorator(self):
        """Test the caching decorator."""
        # Confirm that the caching decorator works.
        a = cached(lambda: random.random())
        b = cached(lambda: random.random())
        assert a() == a()
        assert b() == b()
        # Confirm that functions have their own cache.
        assert a() != b()

    def test_compact(self):
        """Test :func:`humanfriendly.text.compact()`."""
        assert compact(' a \n\n b ') == 'a b'
        assert compact('''
            %s template notation
        ''', 'Simple') == 'Simple template notation'
        assert compact('''
            More {type} template notation
        ''', type='readable') == 'More readable template notation'

    def test_compact_empty_lines(self):
        """Test :func:`humanfriendly.text.compact_empty_lines()`."""
        # Simple strings pass through untouched.
        assert compact_empty_lines('foo') == 'foo'
        # Horizontal whitespace remains untouched.
        assert compact_empty_lines('\tfoo') == '\tfoo'
        # Line breaks should be preserved.
        assert compact_empty_lines('foo\nbar') == 'foo\nbar'
        # Vertical whitespace should be preserved.
        assert compact_empty_lines('foo\n\nbar') == 'foo\n\nbar'
        # Vertical whitespace should be compressed.
        assert compact_empty_lines('foo\n\n\nbar') == 'foo\n\nbar'
        assert compact_empty_lines('foo\n\n\n\nbar') == 'foo\n\nbar'
        assert compact_empty_lines('foo\n\n\n\n\nbar') == 'foo\n\nbar'

    def test_dedent(self):
        """Test :func:`humanfriendly.text.dedent()`."""
        assert dedent('\n line 1\n  line 2\n\n') == 'line 1\n line 2\n'
        assert dedent('''
            Dedented, %s text
        ''', 'interpolated') == 'Dedented, interpolated text\n'
        assert dedent('''
            Dedented, {op} text
        ''', op='formatted') == 'Dedented, formatted text\n'

    def test_pluralization(self):
        """Test :func:`humanfriendly.text.pluralize()`."""
        assert pluralize(1, 'word') == '1 word'
        assert pluralize(2, 'word') == '2 words'
        assert pluralize(1, 'box', 'boxes') == '1 box'
        assert pluralize(2, 'box', 'boxes') == '2 boxes'

    def test_generate_slug(self):
        """Test :func:`humanfriendly.text.generate_slug()`."""
        # Test the basic functionality.
        self.assertEqual('some-random-text', generate_slug('Some Random Text!'))
        # Test that previous output doesn't change.
        self.assertEqual('some-random-text', generate_slug('some-random-text'))
        # Test that inputs which can't be converted to a slug raise an exception.
        with self.assertRaises(ValueError):
            generate_slug(' ')
        with self.assertRaises(ValueError):
            generate_slug('-')

    def test_boolean_coercion(self):
        """Test :func:`humanfriendly.coerce_boolean()`."""
        for value in [True, 'TRUE', 'True', 'true', 'on', 'yes', '1']:
            self.assertEqual(True, coerce_boolean(value))
        for value in [False, 'FALSE', 'False', 'false', 'off', 'no', '0']:
            self.assertEqual(False, coerce_boolean(value))
        with self.assertRaises(ValueError):
            coerce_boolean('not a boolean')

    def test_pattern_coercion(self):
        """Test :func:`humanfriendly.coerce_pattern()`."""
        empty_pattern = re.compile('')
        # Make sure strings are converted to compiled regular expressions.
        assert isinstance(coerce_pattern('foobar'), type(empty_pattern))
        # Make sure compiled regular expressions pass through untouched.
        assert empty_pattern is coerce_pattern(empty_pattern)
        # Make sure flags are respected.
        pattern = coerce_pattern('foobar', re.IGNORECASE)
        assert pattern.match('FOOBAR')
        # Make sure invalid values raise the expected exception.
        with self.assertRaises(ValueError):
            coerce_pattern([])

    def test_format_timespan(self):
        """Test :func:`humanfriendly.format_timespan()`."""
        minute = 60
        hour = minute * 60
        day = hour * 24
        week = day * 7
        year = week * 52
        assert '1 nanosecond' == format_timespan(0.000000001, detailed=True)
        assert '500 nanoseconds' == format_timespan(0.0000005, detailed=True)
        assert '1 microsecond' == format_timespan(0.000001, detailed=True)
        assert '500 microseconds' == format_timespan(0.0005, detailed=True)
        assert '1 millisecond' == format_timespan(0.001, detailed=True)
        assert '500 milliseconds' == format_timespan(0.5, detailed=True)
        assert '0.5 seconds' == format_timespan(0.5, detailed=False)
        assert '0 seconds' == format_timespan(0)
        assert '0.54 seconds' == format_timespan(0.54321)
        assert '1 second' == format_timespan(1)
        assert '3.14 seconds' == format_timespan(math.pi)
        assert '1 minute' == format_timespan(minute)
        assert '1 minute and 20 seconds' == format_timespan(80)
        assert '2 minutes' == format_timespan(minute * 2)
        assert '1 hour' == format_timespan(hour)
        assert '2 hours' == format_timespan(hour * 2)
        assert '1 day' == format_timespan(day)
        assert '2 days' == format_timespan(day * 2)
        assert '1 week' == format_timespan(week)
        assert '2 weeks' == format_timespan(week * 2)
        assert '1 year' == format_timespan(year)
        assert '2 years' == format_timespan(year * 2)
        assert '6 years, 5 weeks, 4 days, 3 hours, 2 minutes and 500 milliseconds' == \
            format_timespan(year * 6 + week * 5 + day * 4 + hour * 3 + minute * 2 + 0.5, detailed=True)
        assert '1 year, 2 weeks and 3 days' == \
            format_timespan(year + week * 2 + day * 3 + hour * 12)
        # Make sure milliseconds are never shown separately when detailed=False.
        # https://github.com/xolox/python-humanfriendly/issues/10
        assert '1 minute, 1 second and 100 milliseconds' == format_timespan(61.10, detailed=True)
        assert '1 minute and 1.1 seconds' == format_timespan(61.10, detailed=False)
        # Test for loss of precision as reported in issue 11:
        # https://github.com/xolox/python-humanfriendly/issues/11
        assert '1 minute and 0.3 seconds' == format_timespan(60.300)
        assert '5 minutes and 0.3 seconds' == format_timespan(300.300)
        assert '1 second and 15 milliseconds' == format_timespan(1.015, detailed=True)
        assert '10 seconds and 15 milliseconds' == format_timespan(10.015, detailed=True)
        assert '1 microsecond and 50 nanoseconds' == format_timespan(0.00000105, detailed=True)
        # Test the datetime.timedelta support:
        # https://github.com/xolox/python-humanfriendly/issues/27
        now = datetime.datetime.now()
        then = now - datetime.timedelta(hours=23)
        assert '23 hours' == format_timespan(now - then)

    def test_parse_timespan(self):
        """Test :func:`humanfriendly.parse_timespan()`."""
        self.assertEqual(0, parse_timespan('0'))
        self.assertEqual(0, parse_timespan('0s'))
        self.assertEqual(0.000000001, parse_timespan('1ns'))
        self.assertEqual(0.000000051, parse_timespan('51ns'))
        self.assertEqual(0.000001, parse_timespan('1us'))
        self.assertEqual(0.000052, parse_timespan('52us'))
        self.assertEqual(0.001, parse_timespan('1ms'))
        self.assertEqual(0.001, parse_timespan('1 millisecond'))
        self.assertEqual(0.5, parse_timespan('500 milliseconds'))
        self.assertEqual(0.5, parse_timespan('0.5 seconds'))
        self.assertEqual(5, parse_timespan('5s'))
        self.assertEqual(5, parse_timespan('5 seconds'))
        self.assertEqual(60 * 2, parse_timespan('2m'))
        self.assertEqual(60 * 2, parse_timespan('2 minutes'))
        self.assertEqual(60 * 3, parse_timespan('3 min'))
        self.assertEqual(60 * 3, parse_timespan('3 mins'))
        self.assertEqual(60 * 60 * 3, parse_timespan('3 h'))
        self.assertEqual(60 * 60 * 3, parse_timespan('3 hours'))
        self.assertEqual(60 * 60 * 24 * 4, parse_timespan('4d'))
        self.assertEqual(60 * 60 * 24 * 4, parse_timespan('4 days'))
        self.assertEqual(60 * 60 * 24 * 7 * 5, parse_timespan('5 w'))
        self.assertEqual(60 * 60 * 24 * 7 * 5, parse_timespan('5 weeks'))
        with self.assertRaises(InvalidTimespan):
            parse_timespan('1z')

    def test_parse_date(self):
        """Test :func:`humanfriendly.parse_date()`."""
        self.assertEqual((2013, 6, 17, 0, 0, 0), parse_date('2013-06-17'))
        self.assertEqual((2013, 6, 17, 2, 47, 42), parse_date('2013-06-17 02:47:42'))
        self.assertEqual((2016, 11, 30, 0, 47, 17), parse_date(u'2016-11-30 00:47:17'))
        with self.assertRaises(InvalidDate):
            parse_date('2013-06-XY')

    def test_format_size(self):
        """Test :func:`humanfriendly.format_size()`."""
        self.assertEqual('0 bytes', format_size(0))
        self.assertEqual('1 byte', format_size(1))
        self.assertEqual('42 bytes', format_size(42))
        self.assertEqual('1 KB', format_size(1000 ** 1))
        self.assertEqual('1 MB', format_size(1000 ** 2))
        self.assertEqual('1 GB', format_size(1000 ** 3))
        self.assertEqual('1 TB', format_size(1000 ** 4))
        self.assertEqual('1 PB', format_size(1000 ** 5))
        self.assertEqual('1 EB', format_size(1000 ** 6))
        self.assertEqual('1 ZB', format_size(1000 ** 7))
        self.assertEqual('1 YB', format_size(1000 ** 8))
        self.assertEqual('1 KiB', format_size(1024 ** 1, binary=True))
        self.assertEqual('1 MiB', format_size(1024 ** 2, binary=True))
        self.assertEqual('1 GiB', format_size(1024 ** 3, binary=True))
        self.assertEqual('1 TiB', format_size(1024 ** 4, binary=True))
        self.assertEqual('1 PiB', format_size(1024 ** 5, binary=True))
        self.assertEqual('1 EiB', format_size(1024 ** 6, binary=True))
        self.assertEqual('1 ZiB', format_size(1024 ** 7, binary=True))
        self.assertEqual('1 YiB', format_size(1024 ** 8, binary=True))
        self.assertEqual('45 KB', format_size(1000 * 45))
        self.assertEqual('2.9 TB', format_size(1000 ** 4 * 2.9))

    def test_parse_size(self):
        """Test :func:`humanfriendly.parse_size()`."""
        self.assertEqual(0, parse_size('0B'))
        self.assertEqual(42, parse_size('42'))
        self.assertEqual(42, parse_size('42B'))
        self.assertEqual(1000, parse_size('1k'))
        self.assertEqual(1024, parse_size('1k', binary=True))
        self.assertEqual(1000, parse_size('1 KB'))
        self.assertEqual(1000, parse_size('1 kilobyte'))
        self.assertEqual(1024, parse_size('1 kilobyte', binary=True))
        self.assertEqual(1000 ** 2 * 69, parse_size('69 MB'))
        self.assertEqual(1000 ** 3, parse_size('1 GB'))
        self.assertEqual(1000 ** 4, parse_size('1 TB'))
        self.assertEqual(1000 ** 5, parse_size('1 PB'))
        self.assertEqual(1000 ** 6, parse_size('1 EB'))
        self.assertEqual(1000 ** 7, parse_size('1 ZB'))
        self.assertEqual(1000 ** 8, parse_size('1 YB'))
        self.assertEqual(1000 ** 3 * 1.5, parse_size('1.5 GB'))
        self.assertEqual(1024 ** 8 * 1.5, parse_size('1.5 YiB'))
        with self.assertRaises(InvalidSize):
            parse_size('1q')
        with self.assertRaises(InvalidSize):
            parse_size('a')

    def test_format_length(self):
        """Test :func:`humanfriendly.format_length()`."""
        self.assertEqual('0 metres', format_length(0))
        self.assertEqual('1 metre', format_length(1))
        self.assertEqual('42 metres', format_length(42))
        self.assertEqual('1 km', format_length(1 * 1000))
        self.assertEqual('15.3 cm', format_length(0.153))
        self.assertEqual('1 cm', format_length(1e-02))
        self.assertEqual('1 mm', format_length(1e-03))
        self.assertEqual('1 nm', format_length(1e-09))

    def test_parse_length(self):
        """Test :func:`humanfriendly.parse_length()`."""
        self.assertEqual(0, parse_length('0m'))
        self.assertEqual(42, parse_length('42'))
        self.assertEqual(1.5, parse_length('1.5'))
        self.assertEqual(42, parse_length('42m'))
        self.assertEqual(1000, parse_length('1km'))
        self.assertEqual(0.153, parse_length('15.3 cm'))
        self.assertEqual(1e-02, parse_length('1cm'))
        self.assertEqual(1e-03, parse_length('1mm'))
        self.assertEqual(1e-09, parse_length('1nm'))
        with self.assertRaises(InvalidLength):
            parse_length('1z')
        with self.assertRaises(InvalidLength):
            parse_length('a')

    def test_format_number(self):
        """Test :func:`humanfriendly.format_number()`."""
        self.assertEqual('1', format_number(1))
        self.assertEqual('1.5', format_number(1.5))
        self.assertEqual('1.56', format_number(1.56789))
        self.assertEqual('1.567', format_number(1.56789, 3))
        self.assertEqual('1,000', format_number(1000))
        self.assertEqual('1,000', format_number(1000.12, 0))
        self.assertEqual('1,000,000', format_number(1000000))
        self.assertEqual('1,000,000.42', format_number(1000000.42))
        # Regression test for https://github.com/xolox/python-humanfriendly/issues/40.
        self.assertEqual('-285.67', format_number(-285.67))

    def test_round_number(self):
        """Test :func:`humanfriendly.round_number()`."""
        self.assertEqual('1', round_number(1))
        self.assertEqual('1', round_number(1.0))
        self.assertEqual('1.00', round_number(1, keep_width=True))
        self.assertEqual('3.14', round_number(3.141592653589793))

    def test_format_path(self):
        """Test :func:`humanfriendly.format_path()`."""
        friendly_path = os.path.join('~', '.vimrc')
        absolute_path = os.path.join(os.environ['HOME'], '.vimrc')
        self.assertEqual(friendly_path, format_path(absolute_path))

    def test_parse_path(self):
        """Test :func:`humanfriendly.parse_path()`."""
        friendly_path = os.path.join('~', '.vimrc')
        absolute_path = os.path.join(os.environ['HOME'], '.vimrc')
        self.assertEqual(absolute_path, parse_path(friendly_path))

    def test_pretty_tables(self):
        """Test :func:`humanfriendly.tables.format_pretty_table()`."""
        # The simplest case possible :-).
        data = [['Just one column']]
        assert format_pretty_table(data) == dedent("""
            -------------------
            | Just one column |
            -------------------
        """).strip()
        # A bit more complex: two rows, three columns, varying widths.
        data = [['One', 'Two', 'Three'], ['1', '2', '3']]
        assert format_pretty_table(data) == dedent("""
            ---------------------
            | One | Two | Three |
            | 1   | 2   | 3     |
            ---------------------
        """).strip()
        # A table including column names.
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        assert ansi_strip(format_pretty_table(data, column_names)) == dedent("""
            ---------------------
            | One | Two | Three |
            ---------------------
            | 1   | 2   | 3     |
            | a   | b   | c     |
            ---------------------
        """).strip()
        # A table that contains a column with only numeric data (will be right aligned).
        column_names = ['Just a label', 'Important numbers']
        data = [['Row one', '15'], ['Row two', '300']]
        assert ansi_strip(format_pretty_table(data, column_names)) == dedent("""
            ------------------------------------
            | Just a label | Important numbers |
            ------------------------------------
            | Row one      |                15 |
            | Row two      |               300 |
            ------------------------------------
        """).strip()

    def test_robust_tables(self):
        """Test :func:`humanfriendly.tables.format_robust_table()`."""
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        assert ansi_strip(format_robust_table(data, column_names)) == dedent("""
            --------
            One: 1
            Two: 2
            Three: 3
            --------
            One: a
            Two: b
            Three: c
            --------
        """).strip()
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'Here comes a\nmulti line column!']]
        assert ansi_strip(format_robust_table(data, column_names)) == dedent("""
            ------------------
            One: 1
            Two: 2
            Three: 3
            ------------------
            One: a
            Two: b
            Three:
            Here comes a
            multi line column!
            ------------------
        """).strip()

    def test_smart_tables(self):
        """Test :func:`humanfriendly.tables.format_smart_table()`."""
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        assert ansi_strip(format_smart_table(data, column_names)) == dedent("""
            ---------------------
            | One | Two | Three |
            ---------------------
            | 1   | 2   | 3     |
            | a   | b   | c     |
            ---------------------
        """).strip()
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'Here comes a\nmulti line column!']]
        assert ansi_strip(format_smart_table(data, column_names)) == dedent("""
            ------------------
            One: 1
            Two: 2
            Three: 3
            ------------------
            One: a
            Two: b
            Three:
            Here comes a
            multi line column!
            ------------------
        """).strip()

    def test_rst_tables(self):
        """Test :func:`humanfriendly.tables.format_rst_table()`."""
        # Generate a table with column names.
        column_names = ['One', 'Two', 'Three']
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        self.assertEqual(
            format_rst_table(data, column_names),
            dedent("""
                ===  ===  =====
                One  Two  Three
                ===  ===  =====
                1    2    3
                a    b    c
                ===  ===  =====
            """).rstrip(),
        )
        # Generate a table without column names.
        data = [['1', '2', '3'], ['a', 'b', 'c']]
        self.assertEqual(
            format_rst_table(data),
            dedent("""
                =  =  =
                1  2  3
                a  b  c
                =  =  =
            """).rstrip(),
        )

    def test_concatenate(self):
        """Test :func:`humanfriendly.text.concatenate()`."""
        assert concatenate([]) == ''
        assert concatenate(['one']) == 'one'
        assert concatenate(['one', 'two']) == 'one and two'
        assert concatenate(['one', 'two', 'three']) == 'one, two and three'
        # Test the 'conjunction' option.
        assert concatenate(['one', 'two', 'three'], conjunction='or') == 'one, two or three'
        # Test the 'serial_comma' option.
        assert concatenate(['one', 'two', 'three'], serial_comma=True) == 'one, two, and three'

    def test_split(self):
        """Test :func:`humanfriendly.text.split()`."""
        from humanfriendly.text import split
        self.assertEqual(split(''), [])
        self.assertEqual(split('foo'), ['foo'])
        self.assertEqual(split('foo, bar'), ['foo', 'bar'])
        self.assertEqual(split('foo, bar, baz'), ['foo', 'bar', 'baz'])
        self.assertEqual(split('foo,bar,baz'), ['foo', 'bar', 'baz'])

    def test_timer(self):
        """Test :func:`humanfriendly.Timer`."""
        for seconds, text in ((1, '1 second'),
                              (2, '2 seconds'),
                              (60, '1 minute'),
                              (60 * 2, '2 minutes'),
                              (60 * 60, '1 hour'),
                              (60 * 60 * 2, '2 hours'),
                              (60 * 60 * 24, '1 day'),
                              (60 * 60 * 24 * 2, '2 days'),
                              (60 * 60 * 24 * 7, '1 week'),
                              (60 * 60 * 24 * 7 * 2, '2 weeks')):
            t = Timer(time.time() - seconds)
            self.assertEqual(round_number(t.elapsed_time, keep_width=True), '%i.00' % seconds)
            self.assertEqual(str(t), text)
        # Test rounding to seconds.
        t = Timer(time.time() - 2.2)
        self.assertEqual(t.rounded, '2 seconds')
        # Test automatic timer.
        automatic_timer = Timer()
        time.sleep(1)
        # XXX The following normalize_timestamp(ndigits=0) calls are intended
        #     to compensate for unreliable clock sources in virtual machines
        #     like those encountered on Travis CI, see also:
        #     https://travis-ci.org/xolox/python-humanfriendly/jobs/323944263
        self.assertEqual(normalize_timestamp(automatic_timer.elapsed_time, 0), '1.00')
        # Test resumable timer.
        resumable_timer = Timer(resumable=True)
        for i in range(2):
            with resumable_timer:
                time.sleep(1)
        self.assertEqual(normalize_timestamp(resumable_timer.elapsed_time, 0), '2.00')
        # Make sure Timer.__enter__() returns the timer object.
        with Timer(resumable=True) as timer:
            assert timer is not None

    def test_spinner(self):
        """Test :func:`humanfriendly.Spinner`."""
        stream = StringIO()
        spinner = Spinner(label='test spinner', total=4, stream=stream, interactive=True)
        for progress in [1, 2, 3, 4]:
            spinner.step(progress=progress)
            time.sleep(0.2)
        spinner.clear()
        output = stream.getvalue()
        output = (output.replace(ANSI_SHOW_CURSOR, '')
                        .replace(ANSI_HIDE_CURSOR, ''))
        lines = [line for line in output.split(ANSI_ERASE_LINE) if line]
        self.assertTrue(len(lines) > 0)
        self.assertTrue(all('test spinner' in line for line in lines))
        self.assertTrue(all('%' in line for line in lines))
        self.assertEqual(sorted(set(lines)), sorted(lines))

    def test_automatic_spinner(self):
        """
        Test :func:`humanfriendly.AutomaticSpinner`.

        There's not a lot to test about the :class:`.AutomaticSpinner` class,
        but by at least running it here we are assured that the code functions
        on all supported Python versions. :class:`.AutomaticSpinner` is built
        on top of the :class:`.Spinner` class so at least we also have the
        tests for the :class:`.Spinner` class to back us up.
        """
        with AutomaticSpinner(label='test spinner'):
            time.sleep(1)

    def test_prompt_for_choice(self):
        """Test :func:`humanfriendly.prompts.prompt_for_choice()`."""
        # Choice selection without any options should raise an exception.
        with self.assertRaises(ValueError):
            prompt_for_choice([])
        # If there's only one option no prompt should be rendered so we expect
        # the following code to not raise an EOFError exception (despite
        # connecting standard input to /dev/null).
        with open(os.devnull) as handle:
            with PatchedAttribute(sys, 'stdin', handle):
                only_option = 'only one option (shortcut)'
                assert prompt_for_choice([only_option]) == only_option
        # Choice selection by full string match.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'foo'):
            assert prompt_for_choice(['foo', 'bar']) == 'foo'
        # Choice selection by substring input.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'f'):
            assert prompt_for_choice(['foo', 'bar']) == 'foo'
        # Choice selection by number.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: '2'):
            assert prompt_for_choice(['foo', 'bar']) == 'bar'
        # Choice selection by going with the default.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            assert prompt_for_choice(['foo', 'bar'], default='bar') == 'bar'
        # Invalid substrings are refused.
        replies = ['', 'q', 'z']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            assert prompt_for_choice(['foo', 'bar', 'baz']) == 'baz'
        # Choice selection by substring input requires an unambiguous substring match.
        replies = ['a', 'q']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            assert prompt_for_choice(['foo', 'bar', 'baz', 'qux']) == 'qux'
        # Invalid numbers are refused.
        replies = ['42', '2']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            assert prompt_for_choice(['foo', 'bar', 'baz']) == 'bar'
        # Test that interactive prompts eventually give up on invalid replies.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            with self.assertRaises(TooManyInvalidReplies):
                prompt_for_choice(['a', 'b', 'c'])

    def test_prompt_for_confirmation(self):
        """Test :func:`humanfriendly.prompts.prompt_for_confirmation()`."""
        # Test some (more or less) reasonable replies that indicate agreement.
        for reply in 'yes', 'Yes', 'YES', 'y', 'Y':
            with PatchedAttribute(prompts, 'interactive_prompt', lambda p: reply):
                assert prompt_for_confirmation("Are you sure?") is True
        # Test some (more or less) reasonable replies that indicate disagreement.
        for reply in 'no', 'No', 'NO', 'n', 'N':
            with PatchedAttribute(prompts, 'interactive_prompt', lambda p: reply):
                assert prompt_for_confirmation("Are you sure?") is False
        # Test that empty replies select the default choice.
        for default_choice in True, False:
            with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
                assert prompt_for_confirmation("Are you sure?", default=default_choice) is default_choice
        # Test that a warning is shown when no input nor a default is given.
        replies = ['', 'y']
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
            with CaptureOutput(merged=True) as capturer:
                assert prompt_for_confirmation("Are you sure?") is True
                assert "there's no default choice" in capturer.get_text()
        # Test that the default reply is shown in uppercase.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'y'):
            for default_value, expected_text in (True, 'Y/n'), (False, 'y/N'), (None, 'y/n'):
                with CaptureOutput(merged=True) as capturer:
                    assert prompt_for_confirmation("Are you sure?", default=default_value) is True
                    assert expected_text in capturer.get_text()
        # Test that interactive prompts eventually give up on invalid replies.
        with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
            with self.assertRaises(TooManyInvalidReplies):
                prompt_for_confirmation("Are you sure?")

    def test_prompt_for_input(self):
        """Test :func:`humanfriendly.prompts.prompt_for_input()`."""
        with open(os.devnull) as handle:
            with PatchedAttribute(sys, 'stdin', handle):
                # If standard input isn't connected to a terminal the default value should be returned.
                default_value = "To seek the holy grail!"
                assert prompt_for_input("What is your quest?", default=default_value) == default_value
                # If standard input isn't connected to a terminal and no default value
                # is given the EOFError exception should be propagated to the caller.
                with self.assertRaises(EOFError):
                    prompt_for_input("What is your favorite color?")

    def test_cli(self):
        """Test the command line interface."""
        # Test that the usage message is printed by default.
        returncode, output = run_cli(main)
        assert 'Usage:' in output
        # Test that the usage message can be requested explicitly.
        returncode, output = run_cli(main, '--help')
        assert 'Usage:' in output
        # Test handling of invalid command line options.
        returncode, output = run_cli(main, '--unsupported-option')
        assert returncode != 0
        # Test `humanfriendly --format-number'.
        returncode, output = run_cli(main, '--format-number=1234567')
        assert output.strip() == '1,234,567'
        # Test `humanfriendly --format-size'.
        random_byte_count = random.randint(1024, 1024 * 1024)
        returncode, output = run_cli(main, '--format-size=%i' % random_byte_count)
        assert output.strip() == format_size(random_byte_count)
        # Test `humanfriendly --format-size --binary'.
        random_byte_count = random.randint(1024, 1024 * 1024)
        returncode, output = run_cli(main, '--format-size=%i' % random_byte_count, '--binary')
        assert output.strip() == format_size(random_byte_count, binary=True)
        # Test `humanfriendly --format-length'.
        random_len = random.randint(1024, 1024 * 1024)
        returncode, output = run_cli(main, '--format-length=%i' % random_len)
        assert output.strip() == format_length(random_len)
        random_len = float(random_len) / 12345.6
        returncode, output = run_cli(main, '--format-length=%f' % random_len)
        assert output.strip() == format_length(random_len)
        # Test `humanfriendly --format-table'.
        returncode, output = run_cli(main, '--format-table', '--delimiter=\t', input='1\t2\t3\n4\t5\t6\n7\t8\t9')
        assert output.strip() == dedent('''
            -------------
            | 1 | 2 | 3 |
            | 4 | 5 | 6 |
            | 7 | 8 | 9 |
            -------------
        ''').strip()
        # Test `humanfriendly --format-timespan'.
        random_timespan = random.randint(5, 600)
        returncode, output = run_cli(main, '--format-timespan=%i' % random_timespan)
        assert output.strip() == format_timespan(random_timespan)
        # Test `humanfriendly --parse-size'.
        returncode, output = run_cli(main, '--parse-size=5 KB')
        assert int(output) == parse_size('5 KB')
        # Test `humanfriendly --parse-size'.
        returncode, output = run_cli(main, '--parse-size=5 YiB')
        assert int(output) == parse_size('5 YB', binary=True)
        # Test `humanfriendly --parse-length'.
        returncode, output = run_cli(main, '--parse-length=5 km')
        assert int(output) == parse_length('5 km')
        returncode, output = run_cli(main, '--parse-length=1.05 km')
        assert float(output) == parse_length('1.05 km')
        # Test `humanfriendly --run-command'.
        returncode, output = run_cli(main, '--run-command', 'bash', '-c', 'sleep 2 && exit 42')
        assert returncode == 42
        # Test `humanfriendly --demo'. The purpose of this test is
        # to ensure that the demo runs successfully on all versions
        # of Python and outputs the expected sections (recognized by
        # their headings) without triggering exceptions. This was
        # written as a regression test after issue #28 was reported:
        # https://github.com/xolox/python-humanfriendly/issues/28
        returncode, output = run_cli(main, '--demo')
        assert returncode == 0
        lines = [ansi_strip(line) for line in output.splitlines()]
        assert "Text styles:" in lines
        assert "Foreground colors:" in lines
        assert "Background colors:" in lines
        assert "256 color mode (standard colors):" in lines
        assert "256 color mode (high-intensity colors):" in lines
        assert "256 color mode (216 colors):" in lines
        assert "256 color mode (gray scale colors):" in lines

    def test_ansi_style(self):
        """Test :func:`humanfriendly.terminal.ansi_style()`."""
        assert ansi_style(bold=True) == '%s1%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(faint=True) == '%s2%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(italic=True) == '%s3%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(underline=True) == '%s4%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(inverse=True) == '%s7%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(strike_through=True) == '%s9%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color='blue') == '%s34%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(background='blue') == '%s44%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color='blue', bright=True) == '%s94%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color=214) == '%s38;5;214%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(background=214) == '%s39;5;214%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color=(0, 0, 0)) == '%s38;2;0;0;0%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(color=(255, 255, 255)) == '%s38;2;255;255;255%s' % (ANSI_CSI, ANSI_SGR)
        assert ansi_style(background=(50, 100, 150)) == '%s48;2;50;100;150%s' % (ANSI_CSI, ANSI_SGR)
        with self.assertRaises(ValueError):
            ansi_style(color='unknown')

    def test_ansi_width(self):
        """Test :func:`humanfriendly.terminal.ansi_width()`."""
        text = "Whatever"
        # Make sure ansi_width() works as expected on strings without ANSI escape sequences.
        assert len(text) == ansi_width(text)
        # Wrap a text in ANSI escape sequences and make sure ansi_width() treats it as expected.
        wrapped = ansi_wrap(text, bold=True)
        # Make sure ansi_wrap() changed the text.
        assert wrapped != text
        # Make sure ansi_wrap() added additional bytes.
        assert len(wrapped) > len(text)
        # Make sure the result of ansi_width() stays the same.
        assert len(text) == ansi_width(wrapped)

    def test_ansi_wrap(self):
        """Test :func:`humanfriendly.terminal.ansi_wrap()`."""
        text = "Whatever"
        # Make sure ansi_wrap() does nothing when no keyword arguments are given.
        assert text == ansi_wrap(text)
        # Make sure ansi_wrap() starts the text with the CSI sequence.
        assert ansi_wrap(text, bold=True).startswith(ANSI_CSI)
        # Make sure ansi_wrap() ends the text by resetting the ANSI styles.
        assert ansi_wrap(text, bold=True).endswith(ANSI_RESET)

    def test_html_to_ansi(self):
        """Test the :func:`humanfriendly.terminal.html_to_ansi()` function."""
        assert html_to_ansi("Just some plain text") == "Just some plain text"
        # Hyperlinks.
        assert html_to_ansi('<a href="https://python.org">python.org</a>') == \
            '\x1b[0m\x1b[4;94mpython.org\x1b[0m (\x1b[0m\x1b[4;94mhttps://python.org\x1b[0m)'
        # Make sure `mailto:' prefixes are stripped (they're not at all useful in a terminal).
        assert html_to_ansi('<a href="mailto:peter@peterodding.com">peter@peterodding.com</a>') == \
            '\x1b[0m\x1b[4;94mpeter@peterodding.com\x1b[0m'
        # Bold text.
        assert html_to_ansi("Let's try <b>bold</b>") == "Let's try \x1b[0m\x1b[1mbold\x1b[0m"
        assert html_to_ansi("Let's try <span style=\"font-weight: bold\">bold</span>") == \
            "Let's try \x1b[0m\x1b[1mbold\x1b[0m"
        # Italic text.
        assert html_to_ansi("Let's try <i>italic</i>") == \
            "Let's try \x1b[0m\x1b[3mitalic\x1b[0m"
        assert html_to_ansi("Let's try <span style=\"font-style: italic\">italic</span>") == \
            "Let's try \x1b[0m\x1b[3mitalic\x1b[0m"
        # Underlined text.
        assert html_to_ansi("Let's try <ins>underline</ins>") == \
            "Let's try \x1b[0m\x1b[4munderline\x1b[0m"
        assert html_to_ansi("Let's try <span style=\"text-decoration: underline\">underline</span>") == \
            "Let's try \x1b[0m\x1b[4munderline\x1b[0m"
        # Strike-through text.
        assert html_to_ansi("Let's try <s>strike-through</s>") == \
            "Let's try \x1b[0m\x1b[9mstrike-through\x1b[0m"
        assert html_to_ansi("Let's try <span style=\"text-decoration: line-through\">strike-through</span>") == \
            "Let's try \x1b[0m\x1b[9mstrike-through\x1b[0m"
        # Pre-formatted text.
        assert html_to_ansi("Let's try <code>pre-formatted</code>") == \
            "Let's try \x1b[0m\x1b[33mpre-formatted\x1b[0m"
        # Text colors (with a 6 digit hexadecimal color value).
        assert html_to_ansi("Let's try <span style=\"color: #AABBCC\">text colors</s>") == \
            "Let's try \x1b[0m\x1b[38;2;170;187;204mtext colors\x1b[0m"
        # Background colors (with an rgb(N, N, N) expression).
        assert html_to_ansi("Let's try <span style=\"background-color: rgb(50, 50, 50)\">background colors</s>") == \
            "Let's try \x1b[0m\x1b[48;2;50;50;50mbackground colors\x1b[0m"
        # Line breaks.
        assert html_to_ansi("Let's try some<br>line<br>breaks") == \
            "Let's try some\nline\nbreaks"
        # Check that decimal entities are decoded.
        assert html_to_ansi("&#38;") == "&"
        # Check that named entities are decoded.
        assert html_to_ansi("&amp;") == "&"
        assert html_to_ansi("&gt;") == ">"
        assert html_to_ansi("&lt;") == "<"
        # Check that hexadecimal entities are decoded.
        assert html_to_ansi("&#x26;") == "&"
        # Check that the text callback is actually called.

        def callback(text):
            return text.replace(':wink:', ';-)')

        assert ':wink:' not in html_to_ansi('<b>:wink:</b>', callback=callback)
        # Check that the text callback doesn't process preformatted text.
        assert ':wink:' in html_to_ansi('<code>:wink:</code>', callback=callback)
        # Try a somewhat convoluted but nevertheless real life example from my
        # personal chat archives that causes humanfriendly releases 4.15 and
        # 4.15.1 to raise an exception.
        assert html_to_ansi(u'''
            Tweakers zit er idd nog steeds:<br><br>
            peter@peter-work&gt; curl -s <a href="tweakers.net">tweakers.net</a> | grep -i hosting<br>
            &lt;a href="<a href="http://www.true.nl/webhosting/">http://www.true.nl/webhosting/</a>"
                rel="external" id="true" title="Hosting door True"&gt;&lt;/a&gt;<br>
            Hosting door &lt;a href="<a href="http://www.true.nl/vps/">http://www.true.nl/vps/</a>"
                title="VPS hosting" rel="external"&gt;True</a>
        ''')

    def test_generate_output(self):
        """Test the :func:`humanfriendly.terminal.output()` function."""
        text = "Standard output generated by output()"
        with CaptureOutput(merged=False) as capturer:
            output(text)
            self.assertEqual([text], capturer.stdout.get_lines())
            self.assertEqual([], capturer.stderr.get_lines())

    def test_generate_message(self):
        """Test the :func:`humanfriendly.terminal.message()` function."""
        text = "Standard error generated by message()"
        with CaptureOutput(merged=False) as capturer:
            message(text)
            self.assertEqual([], capturer.stdout.get_lines())
            self.assertEqual([text], capturer.stderr.get_lines())

    def test_generate_warning(self):
        """Test the :func:`humanfriendly.terminal.warning()` function."""
        from capturer import CaptureOutput
        text = "Standard error generated by warning()"
        with CaptureOutput(merged=False) as capturer:
            warning(text)
            self.assertEqual([], capturer.stdout.get_lines())
            self.assertEqual([ansi_wrap(text, color='red')], self.ignore_coverage_warning(capturer.stderr))

    def ignore_coverage_warning(self, stream):
        """
        Filter out coverage.py warning from standard error.

        This is intended to remove the following line from the lines captured
        on the standard error stream:

        Coverage.py warning: No data was collected. (no-data-collected)
        """
        return [line for line in stream.get_lines() if 'no-data-collected' not in line]

    def test_clean_output(self):
        """Test :func:`humanfriendly.terminal.clean_terminal_output()`."""
        # Simple output should pass through unharmed (single line).
        assert clean_terminal_output('foo') == ['foo']
        # Simple output should pass through unharmed (multiple lines).
        assert clean_terminal_output('foo\nbar') == ['foo', 'bar']
        # Carriage returns and preceding substrings are removed.
        assert clean_terminal_output('foo\rbar\nbaz') == ['bar', 'baz']
        # Carriage returns move the cursor to the start of the line without erasing text.
        assert clean_terminal_output('aaa\rab') == ['aba']
        # Backspace moves the cursor one position back without erasing text.
        assert clean_terminal_output('aaa\b\bb') == ['aba']
        # Trailing empty lines should be stripped.
        assert clean_terminal_output('foo\nbar\nbaz\n\n\n') == ['foo', 'bar', 'baz']

    def test_find_terminal_size(self):
        """Test :func:`humanfriendly.terminal.find_terminal_size()`."""
        lines, columns = find_terminal_size()
        # We really can't assert any minimum or maximum values here because it
        # simply doesn't make any sense; it's impossible for me to anticipate
        # on what environments this test suite will run in the future.
        assert lines > 0
        assert columns > 0
        # The find_terminal_size_using_ioctl() function is the default
        # implementation and it will likely work fine. This makes it hard to
        # test the fall back code paths though. However there's an easy way to
        # make find_terminal_size_using_ioctl() fail ...
        saved_stdin = sys.stdin
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            # What do you mean this is brute force?! ;-)
            sys.stdin = StringIO()
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            # Now find_terminal_size_using_ioctl() should fail even though
            # find_terminal_size_using_stty() might work fine.
            lines, columns = find_terminal_size()
            assert lines > 0
            assert columns > 0
            # There's also an ugly way to make `stty size' fail: The
            # subprocess.Popen class uses os.execvp() underneath, so if we
            # clear the $PATH it will break.
            saved_path = os.environ['PATH']
            try:
                os.environ['PATH'] = ''
                # Now find_terminal_size_using_stty() should fail.
                lines, columns = find_terminal_size()
                assert lines > 0
                assert columns > 0
            finally:
                os.environ['PATH'] = saved_path
        finally:
            sys.stdin = saved_stdin
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr

    def test_terminal_capabilities(self):
        """Test the functions that check for terminal capabilities."""
        from capturer import CaptureOutput
        for test_stream in connected_to_terminal, terminal_supports_colors:
            # This test suite should be able to run interactively as well as
            # non-interactively, so we can't expect or demand that standard streams
            # will always be connected to a terminal. Fortunately Capturer enables
            # us to fake it :-).
            for stream in sys.stdout, sys.stderr:
                with CaptureOutput():
                    assert test_stream(stream)
            # Test something that we know can never be a terminal.
            with open(os.devnull) as handle:
                assert not test_stream(handle)
            # Verify that objects without isatty() don't raise an exception.
            assert not test_stream(object())

    def test_show_pager(self):
        """Test :func:`humanfriendly.terminal.show_pager()`."""
        original_pager = os.environ.get('PAGER', None)
        try:
            # We specifically avoid `less' because it would become awkward to
            # run the test suite in an interactive terminal :-).
            os.environ['PAGER'] = 'cat'
            # Generate a significant amount of random text spread over multiple
            # lines that we expect to be reported literally on the terminal.
            random_text = "\n".join(random_string(25) for i in range(50))
            # Run the pager command and validate the output.
            with CaptureOutput() as capturer:
                show_pager(random_text)
                assert random_text in capturer.get_text()
        finally:
            if original_pager is not None:
                # Restore the original $PAGER value.
                os.environ['PAGER'] = original_pager
            else:
                # Clear the custom $PAGER value.
                os.environ.pop('PAGER')

    def test_get_pager_command(self):
        """Test :func:`humanfriendly.terminal.get_pager_command()`."""
        # Make sure --RAW-CONTROL-CHARS isn't used when it's not needed.
        assert '--RAW-CONTROL-CHARS' not in get_pager_command("Usage message")
        # Make sure --RAW-CONTROL-CHARS is used when it's needed.
        assert '--RAW-CONTROL-CHARS' in get_pager_command(ansi_wrap("Usage message", bold=True))
        # Make sure that less-specific options are only used when valid.
        options_specific_to_less = ['--no-init', '--quit-if-one-screen']
        for pager in 'cat', 'less':
            original_pager = os.environ.get('PAGER', None)
            try:
                # Set $PAGER to `cat' or `less'.
                os.environ['PAGER'] = pager
                # Get the pager command line.
                command_line = get_pager_command()
                # Check for less-specific options.
                if pager == 'less':
                    assert all(opt in command_line for opt in options_specific_to_less)
                else:
                    assert not any(opt in command_line for opt in options_specific_to_less)
            finally:
                if original_pager is not None:
                    # Restore the original $PAGER value.
                    os.environ['PAGER'] = original_pager
                else:
                    # Clear the custom $PAGER value.
                    os.environ.pop('PAGER')

    def test_find_meta_variables(self):
        """Test :func:`humanfriendly.usage.find_meta_variables()`."""
        assert sorted(find_meta_variables("""
            Here's one example: --format-number=VALUE
            Here's another example: --format-size=BYTES
            A final example: --format-timespan=SECONDS
            This line doesn't contain a META variable.
        """)) == sorted(['VALUE', 'BYTES', 'SECONDS'])

    def test_parse_usage_simple(self):
        """Test :func:`humanfriendly.usage.parse_usage()` (a simple case)."""
        introduction, options = self.preprocess_parse_result("""
            Usage: my-fancy-app [OPTIONS]

            Boring description.

            Supported options:

              -h, --help

                Show this message and exit.
        """)
        # The following fragments are (expected to be) part of the introduction.
        assert "Usage: my-fancy-app [OPTIONS]" in introduction
        assert "Boring description." in introduction
        assert "Supported options:" in introduction
        # The following fragments are (expected to be) part of the documented options.
        assert "-h, --help" in options
        assert "Show this message and exit." in options

    def test_parse_usage_tricky(self):
        """Test :func:`humanfriendly.usage.parse_usage()` (a tricky case)."""
        introduction, options = self.preprocess_parse_result("""
            Usage: my-fancy-app [OPTIONS]

            Here's the introduction to my-fancy-app. Some of the lines in the
            introduction start with a command line option just to confuse the
            parsing algorithm :-)

            For example
            --an-awesome-option
            is still part of the introduction.

            Supported options:

              -a, --an-awesome-option

                Explanation why this is an awesome option.

              -b, --a-boring-option

                Explanation why this is a boring option.
        """)
        # The following fragments are (expected to be) part of the introduction.
        assert "Usage: my-fancy-app [OPTIONS]" in introduction
        assert any('still part of the introduction' in p for p in introduction)
        assert "Supported options:" in introduction
        # The following fragments are (expected to be) part of the documented options.
        assert "-a, --an-awesome-option" in options
        assert "Explanation why this is an awesome option." in options
        assert "-b, --a-boring-option" in options
        assert "Explanation why this is a boring option." in options

    def test_parse_usage_commas(self):
        """Test :func:`humanfriendly.usage.parse_usage()` against option labels containing commas."""
        introduction, options = self.preprocess_parse_result("""
            Usage: my-fancy-app [OPTIONS]

            Some introduction goes here.

            Supported options:

              -f, --first-option

                Explanation of first option.

              -s, --second-option=WITH,COMMA

                This should be a separate option's description.
        """)
        # The following fragments are (expected to be) part of the introduction.
        assert "Usage: my-fancy-app [OPTIONS]" in introduction
        assert "Some introduction goes here." in introduction
        assert "Supported options:" in introduction
        # The following fragments are (expected to be) part of the documented options.
        assert "-f, --first-option" in options
        assert "Explanation of first option." in options
        assert "-s, --second-option=WITH,COMMA" in options
        assert "This should be a separate option's description." in options

    def preprocess_parse_result(self, text):
        """Ignore leading/trailing whitespace in usage parsing tests."""
        return tuple([p.strip() for p in r] for r in parse_usage(dedent(text)))

    def test_format_usage(self):
        """Test :func:`humanfriendly.usage.format_usage()`."""
        # Test that options are highlighted.
        usage_text = "Just one --option"
        formatted_text = format_usage(usage_text)
        assert len(formatted_text) > len(usage_text)
        assert formatted_text.startswith("Just one ")
        # Test that the "Usage: ..." line is highlighted.
        usage_text = "Usage: humanfriendly [OPTIONS]"
        formatted_text = format_usage(usage_text)
        assert len(formatted_text) > len(usage_text)
        assert usage_text in formatted_text
        assert not formatted_text.startswith(usage_text)
        # Test that meta variables aren't erroneously highlighted.
        usage_text = (
            "--valid-option=VALID_METAVAR\n"
            "VALID_METAVAR is bogus\n"
            "INVALID_METAVAR should not be highlighted\n"
        )
        formatted_text = format_usage(usage_text)
        formatted_lines = formatted_text.splitlines()
        # Make sure the meta variable in the second line is highlighted.
        assert ANSI_CSI in formatted_lines[1]
        # Make sure the meta variable in the third line isn't highlighted.
        assert ANSI_CSI not in formatted_lines[2]

    def test_render_usage(self):
        """Test :func:`humanfriendly.usage.render_usage()`."""
        assert render_usage("Usage: some-command WITH ARGS") == "**Usage:** `some-command WITH ARGS`"
        assert render_usage("Supported options:") == "**Supported options:**"
        assert 'code-block' in render_usage(dedent("""
            Here comes a shell command:

              $ echo test
              test
        """))
        assert all(token in render_usage(dedent("""
            Supported options:

              -n, --dry-run

                Don't change anything.
        """)) for token in ('`-n`', '`--dry-run`'))

    def test_deprecated_args(self):
        """Test the deprecated_args() decorator function."""
        @deprecated_args('foo', 'bar')
        def test_function(**options):
            assert options['foo'] == 'foo'
            assert options.get('bar') in (None, 'bar')
            return 42
        fake_fn = MagicMock()
        with PatchedAttribute(warnings, 'warn', fake_fn):
            assert test_function('foo', 'bar') == 42
            with self.assertRaises(TypeError):
                test_function('foo', 'bar', 'baz')
        assert fake_fn.was_called

    def test_alias_proxy_deprecation_warning(self):
        """Test that the DeprecationProxy class emits deprecation warnings."""
        fake_fn = MagicMock()
        with PatchedAttribute(warnings, 'warn', fake_fn):
            module = sys.modules[__name__]
            aliases = dict(concatenate='humanfriendly.text.concatenate')
            proxy = DeprecationProxy(module, aliases)
            assert proxy.concatenate == concatenate
        assert fake_fn.was_called

    def test_alias_proxy_sphinx_compensation(self):
        """Test that the DeprecationProxy class emits deprecation warnings."""
        with PatchedItem(sys.modules, 'sphinx', types.ModuleType('sphinx')):
            define_aliases(__name__, concatenate='humanfriendly.text.concatenate')
            assert "concatenate" in dir(sys.modules[__name__])
            assert "concatenate" in get_aliases(__name__)

    def test_alias_proxy_sphinx_integration(self):
        """Test that aliases can be injected into generated documentation."""
        module = sys.modules[__name__]
        define_aliases(__name__, concatenate='humanfriendly.text.concatenate')
        lines = module.__doc__.splitlines()
        deprecation_note_callback(app=None, what=None, name=None, obj=module, options=None, lines=lines)
        # Check that something was injected.
        assert "\n".join(lines) != module.__doc__

    def test_sphinx_customizations(self):
        """Test the :mod:`humanfriendly.sphinx` module."""
        class FakeApp(object):

            def __init__(self):
                self.callbacks = {}
                self.roles = {}

            def __documented_special_method__(self):
                """Documented unofficial special method."""
                pass

            def __undocumented_special_method__(self):
                # Intentionally not documented :-).
                pass

            def add_role(self, name, callback):
                self.roles[name] = callback

            def connect(self, event, callback):
                self.callbacks.setdefault(event, []).append(callback)

            def bogus_usage(self):
                """Usage: This is not supposed to be reformatted!"""
                pass

        # Test event callback registration.
        fake_app = FakeApp()
        setup(fake_app)
        assert man_role == fake_app.roles['man']
        assert pypi_role == fake_app.roles['pypi']
        assert deprecation_note_callback in fake_app.callbacks['autodoc-process-docstring']
        assert special_methods_callback in fake_app.callbacks['autodoc-skip-member']
        assert usage_message_callback in fake_app.callbacks['autodoc-process-docstring']
        # Test that `special methods' which are documented aren't skipped.
        assert special_methods_callback(
            app=None, what=None, name=None,
            obj=FakeApp.__documented_special_method__,
            skip=True, options=None,
        ) is False
        # Test that `special methods' which are undocumented are skipped.
        assert special_methods_callback(
            app=None, what=None, name=None,
            obj=FakeApp.__undocumented_special_method__,
            skip=True, options=None,
        ) is True
        # Test formatting of usage messages. obj/lines
        from humanfriendly import cli, sphinx
        # We expect the docstring in the `cli' module to be reformatted
        # (because it contains a usage message in the expected format).
        assert self.docstring_is_reformatted(cli)
        # We don't expect the docstring in the `sphinx' module to be
        # reformatted (because it doesn't contain a usage message).
        assert not self.docstring_is_reformatted(sphinx)
        # We don't expect the docstring of the following *method* to be
        # reformatted because only *module* docstrings should be reformatted.
        assert not self.docstring_is_reformatted(fake_app.bogus_usage)

    def docstring_is_reformatted(self, entity):
        """Check whether :func:`.usage_message_callback()` reformats a module's docstring."""
        lines = trim_empty_lines(entity.__doc__).splitlines()
        saved_lines = list(lines)
        usage_message_callback(
            app=None, what=None, name=None,
            obj=entity, options=None, lines=lines,
        )
        return lines != saved_lines


def normalize_timestamp(value, ndigits=1):
    """
    Round timestamps to the given number of digits.

    This helps to make the test suite less sensitive to timing issues caused by
    multitasking, processor scheduling, etc.
    """
    return '%.2f' % round(float(value), ndigits=ndigits)