
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\benchmark_controlnet.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import importlib.util
import time
from statistics import mean

import torch
from demo_utils import PipelineInfo
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
)
from engine_builder import EngineType, get_engine_paths
from pipeline_stable_diffusion import StableDiffusionPipeline

"""
Benchmark script for SDXL-Turbo with control net for engines like PyTorch or Stable Fast.

Setup for Stable Fast (see https://github.com/chengzeyi/stable-fast/blob/main/README.md for more info):
    git clone https://github.com/chengzeyi/stable-fast.git
    cd stable-fast
    git submodule update --init
    pip3 install torch torchvision torchaudio ninja
    pip3 install -e '.[dev,xformers,triton,transformers,diffusers]' -v
    sudo apt install libgoogle-perftools-dev
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so
"""


def get_canny_image():
    import cv2
    import numpy as np
    from PIL import Image

    # Test Image can be downloaded from https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png
    image = Image.open("input_image_vermeer.png").convert("RGB")

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def compile_stable_fast(pipeline, enable_cuda_graph=True):
    from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig, compile

    config = CompilationConfig.Default()

    if importlib.util.find_spec("xformers") is not None:
        config.enable_xformers = True

    if importlib.util.find_spec("triton") is not None:
        config.enable_triton = True

    config.enable_cuda_graph = enable_cuda_graph

    pipeline = compile(pipeline, config)
    return pipeline


def compile_torch(pipeline, use_nhwc=False):
    if use_nhwc:
        pipeline.unet.to(memory_format=torch.channels_last)

    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

    if hasattr(pipeline, "controlnet"):
        if use_nhwc:
            pipeline.controlnet.to(memory_format=torch.channels_last)
        pipeline.controlnet = torch.compile(pipeline.controlnet, mode="reduce-overhead", fullgraph=True)
    return pipeline


def load_pipeline(name, engine, use_control_net=False, use_nhwc=False, enable_cuda_graph=True):
    gc.collect()
    torch.cuda.empty_cache()
    before_memory = torch.cuda.memory_allocated()

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

    if use_control_net:
        assert "xl" in name
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            name,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            name,
            vae=vae,
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
    pipeline.safety_checker = None

    gc.collect()
    after_memory = torch.cuda.memory_allocated()
    print(f"Loaded model with {after_memory - before_memory} bytes allocated")

    if engine == "stable_fast":
        pipeline = compile_stable_fast(pipeline, enable_cuda_graph=enable_cuda_graph)
    elif engine == "torch":
        pipeline = compile_torch(pipeline, use_nhwc=use_nhwc)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def get_prompt():
    return "little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed"


def load_ort_cuda_pipeline(name, engine, use_control_net=False, enable_cuda_graph=True, work_dir="."):
    version = PipelineInfo.supported_models()[name]
    guidance_scale = 0.0
    pipeline_info = PipelineInfo(
        version,
        use_vae=True,
        use_fp16_vae=True,
        do_classifier_free_guidance=(guidance_scale > 1.0),
        controlnet=["canny"] if use_control_net else [],
    )

    engine_type = EngineType.ORT_CUDA if engine == "ort_cuda" else EngineType.ORT_TRT
    onnx_dir, engine_dir, output_dir, framework_model_dir, _ = get_engine_paths(
        work_dir=work_dir, pipeline_info=pipeline_info, engine_type=engine_type
    )

    pipeline = StableDiffusionPipeline(
        pipeline_info,
        scheduler="EulerA",
        max_batch_size=32,
        use_cuda_graph=enable_cuda_graph,
        framework_model_dir=framework_model_dir,
        output_dir=output_dir,
        engine_type=engine_type,
    )

    pipeline.backend.build_engines(
        engine_dir=engine_dir,
        framework_model_dir=framework_model_dir,
        onnx_dir=onnx_dir,
        device_id=torch.cuda.current_device(),
    )

    return pipeline


def test_ort_cuda(
    pipeline,
    batch_size=1,
    steps=4,
    control_image=None,
    warmup_runs=3,
    test_runs=10,
    seed=123,
    verbose=False,
    image_height=512,
    image_width=512,
):
    if batch_size > 4 and pipeline.pipeline_info.version == "xl-1.0":
        pipeline.backend.enable_vae_slicing()

    pipeline.load_resources(image_height, image_width, batch_size)

    warmup_prompt = "warm up"
    for _ in range(warmup_runs):
        images, _ = pipeline.run(
            [warmup_prompt] * batch_size,
            [""] * batch_size,
            image_height=image_height,
            image_width=image_width,
            denoising_steps=steps,
            guidance=0.0,
            seed=seed,
            controlnet_images=[control_image],
            controlnet_scales=torch.FloatTensor([0.5]),
            output_type="image",
        )
        assert len(images) == batch_size

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    prompt = get_prompt()

    latency_list = []
    images = None
    for _ in range(test_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        images, _ = pipeline.run(
            [prompt] * batch_size,
            [""] * batch_size,
            image_height=image_height,
            image_width=image_width,
            denoising_steps=steps,
            guidance=0.0,
            seed=seed,
            controlnet_images=[control_image],
            controlnet_scales=torch.FloatTensor([0.5]),
            output_type="pil",
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start_time
        latency_list.append(seconds)

    if verbose:
        print(latency_list)

    return images, latency_list


def test(pipeline, batch_size=1, steps=4, control_image=None, warmup_runs=3, test_runs=10, seed=123, verbose=False):
    control_net_args = {}
    if hasattr(pipeline, "controlnet"):
        control_net_args = {
            "image": control_image,
            "controlnet_conditioning_scale": 0.5,
        }

    warmup_prompt = "warm up"
    for _ in range(warmup_runs):
        images = pipeline(
            prompt=warmup_prompt,
            num_inference_steps=steps,
            num_images_per_prompt=batch_size,
            guidance_scale=0.0,
            **control_net_args,
        ).images
        assert len(images) == batch_size

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    prompt = get_prompt()

    latency_list = []
    images = None
    for _ in range(test_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        images = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            num_images_per_prompt=batch_size,
            guidance_scale=0.0,
            generator=generator,
            **control_net_args,
        ).images
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start_time
        latency_list.append(seconds)

    if verbose:
        print(latency_list)

    return images, latency_list


def arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Stable Diffusion pipeline (optional control net for SDXL)")
    parser.add_argument(
        "--engine",
        type=str,
        default="torch",
        choices=["torch", "stable_fast", "ort_cuda", "ort_trt"],
        help="Backend engine: torch, stable_fast or ort_cuda",
    )

    parser.add_argument(
        "--name",
        type=str,
        choices=list(PipelineInfo.supported_models().keys()),
        default="stabilityai/sdxl-turbo",
        help="Stable diffusion model name. Default is stabilityai/sdxl-turbo",
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        default=".",
        help="working directory for ort_cuda or ort_trt",
    )

    parser.add_argument(
        "--use_control_net",
        action="store_true",
        help="Use control net diffusers/controlnet-canny-sdxl-1.0",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Denoising steps",
    )

    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=3,
        help="Number of warmup runs before measurement",
    )

    parser.add_argument(
        "--use_nhwc",
        action="store_true",
        help="use channel last format for torch compile",
    )

    parser.add_argument(
        "--enable_cuda_graph",
        action="store_true",
        help="enable cuda graph for stable fast",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print more information",
    )

    args = parser.parse_args()
    return args


def main():
    args = arguments()

    with torch.no_grad():
        if args.engine == "ort_cuda":
            pipeline = load_ort_cuda_pipeline(
                args.name,
                args.engine,
                use_control_net=args.use_control_net,
                enable_cuda_graph=args.enable_cuda_graph,
                work_dir=args.work_dir,
            )
        else:
            pipeline = load_pipeline(
                args.name,
                args.engine,
                use_control_net=args.use_control_net,
                use_nhwc=args.use_nhwc,
                enable_cuda_graph=args.enable_cuda_graph,
            )

        canny_image = get_canny_image()

        if args.engine == "ort_cuda":
            images, latency_list = test_ort_cuda(
                pipeline,
                args.batch_size,
                args.steps,
                control_image=canny_image,
                warmup_runs=args.warmup_runs,
                verbose=args.verbose,
            )
        elif args.engine == "stable_fast":
            from sfast.utils.compute_precision import low_compute_precision

            with low_compute_precision():
                images, latency_list = test(
                    pipeline,
                    args.batch_size,
                    args.steps,
                    control_image=canny_image,
                    warmup_runs=args.warmup_runs,
                    verbose=args.verbose,
                )
        else:
            images, latency_list = test(
                pipeline,
                args.batch_size,
                args.steps,
                control_image=canny_image,
                warmup_runs=args.warmup_runs,
                verbose=args.verbose,
            )

        # Save the first output image to inspect the result.
        if images:
            images[0].save(
                f"{args.engine}_{args.name.replace('/', '_')}_{args.batch_size}_{args.steps}_c{int(args.use_control_net)}.png"
            )

        result = {
            "engine": args.engine,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "control_net": args.use_control_net,
            "nhwc": args.use_nhwc,
            "enable_cuda_graph": args.enable_cuda_graph,
            "average_latency_in_ms": mean(latency_list) * 1000,
        }
        print(result)


main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\bluetooth_emulation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: BluetoothEmulation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class CentralState(enum.Enum):
    '''
    Indicates the various states of Central.
    '''
    ABSENT = "absent"
    POWERED_OFF = "powered-off"
    POWERED_ON = "powered-on"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class GATTOperationType(enum.Enum):
    '''
    Indicates the various types of GATT event.
    '''
    CONNECTION = "connection"
    DISCOVERY = "discovery"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ManufacturerData:
    '''
    Stores the manufacturer data
    '''
    #: Company identifier
    #: https://bitbucket.org/bluetooth-SIG/public/src/main/assigned_numbers/company_identifiers/company_identifiers.yaml
    #: https://usb.org/developers
    key: int

    #: Manufacturer-specific data
    data: str

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['data'] = self.data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=int(json['key']),
            data=str(json['data']),
        )


@dataclass
class ScanRecord:
    '''
    Stores the byte data of the advertisement packet sent by a Bluetooth device.
    '''
    name: typing.Optional[str] = None

    uuids: typing.Optional[typing.List[str]] = None

    #: Stores the external appearance description of the device.
    appearance: typing.Optional[int] = None

    #: Stores the transmission power of a broadcasting device.
    tx_power: typing.Optional[int] = None

    #: Key is the company identifier and the value is an array of bytes of
    #: manufacturer specific data.
    manufacturer_data: typing.Optional[typing.List[ManufacturerData]] = None

    def to_json(self):
        json = dict()
        if self.name is not None:
            json['name'] = self.name
        if self.uuids is not None:
            json['uuids'] = [i for i in self.uuids]
        if self.appearance is not None:
            json['appearance'] = self.appearance
        if self.tx_power is not None:
            json['txPower'] = self.tx_power
        if self.manufacturer_data is not None:
            json['manufacturerData'] = [i.to_json() for i in self.manufacturer_data]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']) if 'name' in json else None,
            uuids=[str(i) for i in json['uuids']] if 'uuids' in json else None,
            appearance=int(json['appearance']) if 'appearance' in json else None,
            tx_power=int(json['txPower']) if 'txPower' in json else None,
            manufacturer_data=[ManufacturerData.from_json(i) for i in json['manufacturerData']] if 'manufacturerData' in json else None,
        )


@dataclass
class ScanEntry:
    '''
    Stores the advertisement packet information that is sent by a Bluetooth device.
    '''
    device_address: str

    rssi: int

    scan_record: ScanRecord

    def to_json(self):
        json = dict()
        json['deviceAddress'] = self.device_address
        json['rssi'] = self.rssi
        json['scanRecord'] = self.scan_record.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            device_address=str(json['deviceAddress']),
            rssi=int(json['rssi']),
            scan_record=ScanRecord.from_json(json['scanRecord']),
        )


@dataclass
class CharacteristicProperties:
    '''
    Describes the properties of a characteristic. This follows Bluetooth Core
    Specification BT 4.2 Vol 3 Part G 3.3.1. Characteristic Properties.
    '''
    broadcast: typing.Optional[bool] = None

    read: typing.Optional[bool] = None

    write_without_response: typing.Optional[bool] = None

    write: typing.Optional[bool] = None

    notify: typing.Optional[bool] = None

    indicate: typing.Optional[bool] = None

    authenticated_signed_writes: typing.Optional[bool] = None

    extended_properties: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.broadcast is not None:
            json['broadcast'] = self.broadcast
        if self.read is not None:
            json['read'] = self.read
        if self.write_without_response is not None:
            json['writeWithoutResponse'] = self.write_without_response
        if self.write is not None:
            json['write'] = self.write
        if self.notify is not None:
            json['notify'] = self.notify
        if self.indicate is not None:
            json['indicate'] = self.indicate
        if self.authenticated_signed_writes is not None:
            json['authenticatedSignedWrites'] = self.authenticated_signed_writes
        if self.extended_properties is not None:
            json['extendedProperties'] = self.extended_properties
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            broadcast=bool(json['broadcast']) if 'broadcast' in json else None,
            read=bool(json['read']) if 'read' in json else None,
            write_without_response=bool(json['writeWithoutResponse']) if 'writeWithoutResponse' in json else None,
            write=bool(json['write']) if 'write' in json else None,
            notify=bool(json['notify']) if 'notify' in json else None,
            indicate=bool(json['indicate']) if 'indicate' in json else None,
            authenticated_signed_writes=bool(json['authenticatedSignedWrites']) if 'authenticatedSignedWrites' in json else None,
            extended_properties=bool(json['extendedProperties']) if 'extendedProperties' in json else None,
        )


def enable(
        state: CentralState,
        le_supported: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enable the BluetoothEmulation domain.

    :param state: State of the simulated central.
    :param le_supported: If the simulated central supports low-energy.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state.to_json()
    params['leSupported'] = le_supported
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.enable',
        'params': params,
    }
    json = yield cmd_dict


def set_simulated_central_state(
        state: CentralState
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the state of the simulated central.

    :param state: State of the simulated central.
    '''
    params: T_JSON_DICT = dict()
    params['state'] = state.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.setSimulatedCentralState',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable the BluetoothEmulation domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.disable',
    }
    json = yield cmd_dict


def simulate_preconnected_peripheral(
        address: str,
        name: str,
        manufacturer_data: typing.List[ManufacturerData],
        known_service_uuids: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates a peripheral with ``address``, ``name`` and ``knownServiceUuids``
    that has already been connected to the system.

    :param address:
    :param name:
    :param manufacturer_data:
    :param known_service_uuids:
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['name'] = name
    params['manufacturerData'] = [i.to_json() for i in manufacturer_data]
    params['knownServiceUuids'] = [i for i in known_service_uuids]
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulatePreconnectedPeripheral',
        'params': params,
    }
    json = yield cmd_dict


def simulate_advertisement(
        entry: ScanEntry
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates an advertisement packet described in ``entry`` being received by
    the central.

    :param entry:
    '''
    params: T_JSON_DICT = dict()
    params['entry'] = entry.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulateAdvertisement',
        'params': params,
    }
    json = yield cmd_dict


def simulate_gatt_operation_response(
        address: str,
        type_: GATTOperationType,
        code: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Simulates the response code from the peripheral with ``address`` for a
    GATT operation of ``type``. The ``code`` value follows the HCI Error Codes from
    Bluetooth Core Specification Vol 2 Part D 1.3 List Of Error Codes.

    :param address:
    :param type_:
    :param code:
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['type'] = type_.to_json()
    params['code'] = code
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.simulateGATTOperationResponse',
        'params': params,
    }
    json = yield cmd_dict


def add_service(
        address: str,
        service_uuid: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Adds a service with ``serviceUuid`` to the peripheral with ``address``.

    :param address:
    :param service_uuid:
    :returns: An identifier that uniquely represents this service.
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['serviceUuid'] = service_uuid
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.addService',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['serviceId'])


def remove_service(
        address: str,
        service_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the service respresented by ``serviceId`` from the peripheral with
    ``address``.

    :param address:
    :param service_id:
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['serviceId'] = service_id
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.removeService',
        'params': params,
    }
    json = yield cmd_dict


def add_characteristic(
        address: str,
        service_id: str,
        characteristic_uuid: str,
        properties: CharacteristicProperties
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Adds a characteristic with ``characteristicUuid`` and ``properties`` to the
    service represented by ``serviceId`` in the peripheral with ``address``.

    :param address:
    :param service_id:
    :param characteristic_uuid:
    :param properties:
    :returns: An identifier that uniquely represents this characteristic.
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['serviceId'] = service_id
    params['characteristicUuid'] = characteristic_uuid
    params['properties'] = properties.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.addCharacteristic',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['characteristicId'])


def remove_characteristic(
        address: str,
        service_id: str,
        characteristic_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes the characteristic respresented by ``characteristicId`` from the
    service respresented by ``serviceId`` in the peripheral with ``address``.

    :param address:
    :param service_id:
    :param characteristic_id:
    '''
    params: T_JSON_DICT = dict()
    params['address'] = address
    params['serviceId'] = service_id
    params['characteristicId'] = characteristic_id
    cmd_dict: T_JSON_DICT = {
        'method': 'BluetoothEmulation.removeCharacteristic',
        'params': params,
    }
    json = yield cmd_dict


@event_class('BluetoothEmulation.gattOperationReceived')
@dataclass
class GattOperationReceived:
    '''
    Event for when a GATT operation of ``type`` to the peripheral with ``address``
    happened.
    '''
    address: str
    type_: GATTOperationType

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> GattOperationReceived:
        return cls(
            address=str(json['address']),
            type_=GATTOperationType.from_json(json['type'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\joint_rv.py ===
"""
Joint Random Variables Module

See Also
========
sympy.stats.rv
sympy.stats.frv
sympy.stats.crv
sympy.stats.drv
"""
from math import prod

from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
                            ProductDomain, RandomSymbol, random_symbols,
                            SingleDomain, _symbol_converter)
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module

# __all__ = ['marginal_distribution']

class JointPSpace(ProductPSpace):
    """
    Represents a joint probability space. Represented using symbols for
    each component and a distribution.
    """
    def __new__(cls, sym, dist):
        if isinstance(dist, SingleContinuousDistribution):
            return SingleContinuousPSpace(sym, dist)
        if isinstance(dist, SingleDiscreteDistribution):
            return SingleDiscretePSpace(sym, dist)
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym, dist)

    @property
    def set(self):
        return self.domain.set

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[1]

    @property
    def value(self):
        return JointRandomSymbol(self.symbol, self)

    @property
    def component_count(self):
        _set = self.distribution.set
        if isinstance(_set, ProductSet):
            return S(len(_set.args))
        elif isinstance(_set, Product):
            return _set.limits[0][-1]
        return S.One

    @property
    def pdf(self):
        sym = [Indexed(self.symbol, i) for i in range(self.component_count)]
        return self.distribution(*sym)

    @property
    def domain(self):
        rvs = random_symbols(self.distribution)
        if not rvs:
            return SingleDomain(self.symbol, self.distribution.set)
        return ProductDomain(*[rv.pspace.domain for rv in rvs])

    def component_domain(self, index):
        return self.set.args[index]

    def marginal_distribution(self, *indices):
        count = self.component_count
        if count.atoms(Symbol):
            raise ValueError("Marginal distributions cannot be computed "
                                "for symbolic dimensions. It is a work under progress.")
        orig = [Indexed(self.symbol, i) for i in range(count)]
        all_syms = [Symbol(str(i)) for i in orig]
        replace_dict = dict(zip(all_syms, orig))
        sym = tuple(Symbol(str(Indexed(self.symbol, i))) for i in indices)
        limits = [[i,] for i in all_syms if i not in sym]
        index = 0
        for i in range(count):
            if i not in indices:
                limits[index].append(self.distribution.set.args[i])
                limits[index] = tuple(limits[index])
                index += 1
        if self.distribution.is_Continuous:
            f = Lambda(sym, integrate(self.distribution(*all_syms), *limits))
        elif self.distribution.is_Discrete:
            f = Lambda(sym, summation(self.distribution(*all_syms), *limits))
        return f.xreplace(replace_dict)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        syms = tuple(self.value[i] for i in range(self.component_count))
        rvs = rvs or syms
        if not any(i in rvs for i in syms):
            return expr
        expr = expr*self.pdf
        for rv in rvs:
            if isinstance(rv, Indexed):
                expr = expr.xreplace({rv: Indexed(str(rv.base), rv.args[1])})
            elif isinstance(rv, RandomSymbol):
                expr = expr.xreplace({rv: rv.symbol})
        if self.value in random_symbols(expr):
            raise NotImplementedError(filldedent('''
            Expectations of expression with unindexed joint random symbols
            cannot be calculated yet.'''))
        limits = tuple((Indexed(str(rv.base),rv.args[1]),
            self.distribution.set.args[rv.args[1]]) for rv in syms)
        return Integral(expr, *limits)

    def where(self, condition):
        raise NotImplementedError()

    def compute_density(self, expr):
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {RandomSymbol(self.symbol, self): self.distribution.sample(size,
                    library=library, seed=seed)}

    def probability(self, condition):
        raise NotImplementedError()


class SampleJointScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        from scipy import stats as scipy_stats
        scipy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: scipy_stats.multivariate_normal.rvs(
                mean=matrix2numpy(dist.mu).flatten(),
                cov=matrix2numpy(dist.sigma), size=size, random_state=rand_state),
            'MultivariateBetaDistribution': lambda dist, size: scipy_stats.dirichlet.rvs(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size, random_state=rand_state),
            'MultinomialDistribution': lambda dist, size: scipy_stats.multinomial.rvs(
                n=int(dist.n), p=list2numpy(dist.p, float).flatten(), size=size, random_state=rand_state)
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = scipy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        samples = scipy_rv_map[dist.__class__.__name__](dist, size)
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointNumpy:
    """Returns the sample from numpy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        numpy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: rand_state.multivariate_normal(
                mean=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), size=size),
            'MultivariateBetaDistribution': lambda dist, size: rand_state.dirichlet(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size),
            'MultinomialDistribution': lambda dist, size: rand_state.multinomial(
                n=int(dist.n), pvals=list2numpy(dist.p, float).flatten(), size=size)
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = numpy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        samples = numpy_rv_map[dist.__class__.__name__](dist, prod(size))
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {
            'MultivariateNormalDistribution': lambda dist:
                pymc.MvNormal('X', mu=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), shape=(1, dist.mu.shape[0])),
            'MultivariateBetaDistribution': lambda dist:
                pymc.Dirichlet('X', a=list2numpy(dist.alpha, float).flatten()),
            'MultinomialDistribution': lambda dist:
                pymc.Multinomial('X', n=int(dist.n),
                p=list2numpy(dist.p, float).flatten(), shape=(1, len(dist.p)))
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = pymc_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        import logging
        logging.getLogger("pymc3").setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samples = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)[:]['X']
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))


_get_sample_class_jrv = {
    'scipy': SampleJointScipy,
    'pymc3': SampleJointPymc,
    'pymc': SampleJointPymc,
    'numpy': SampleJointNumpy
}

class JointDistribution(Distribution, NamedArgsMixin):
    """
    Represented by the random variables part of the joint distribution.
    Contains methods for PDF, CDF, sampling, marginal densities, etc.
    """

    _argnames = ('pdf', )

    def __new__(cls, *args):
        args = list(map(sympify, args))
        for i in range(len(args)):
            if isinstance(args[i], list):
                args[i] = ImmutableMatrix(args[i])
        return Basic.__new__(cls, *args)

    @property
    def domain(self):
        return ProductDomain(self.symbols)

    @property
    def pdf(self):
        return self.density.args[1]

    def cdf(self, other):
        if not isinstance(other, dict):
            raise ValueError("%s should be of type dict, got %s"%(other, type(other)))
        rvs = other.keys()
        _set = self.domain.set.sets
        expr = self.pdf(tuple(i.args[0] for i in self.symbols))
        for i in range(len(other)):
            if rvs[i].is_Continuous:
                density = Integral(expr, (rvs[i], _set[i].inf,
                    other[rvs[i]]))
            elif rvs[i].is_Discrete:
                density = Sum(expr, (rvs[i], _set[i].inf,
                    other[rvs[i]]))
        return density

    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """

        libraries = ('scipy', 'numpy', 'pymc3', 'pymc')
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        samps = _get_sample_class_jrv[library](self, size, seed=seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

    def __call__(self, *args):
        return self.pdf(*args)

class JointRandomSymbol(RandomSymbol):
    """
    Representation of random symbols with joint probability distributions
    to allow indexing."
    """
    def __getitem__(self, key):
        if isinstance(self.pspace, JointPSpace):
            if (self.pspace.component_count <= key) == True:
                raise ValueError("Index keys for %s can only up to %s." %
                    (self.name, self.pspace.component_count - 1))
            return Indexed(self, key)



class MarginalDistribution(Distribution):
    """
    Represents the marginal distribution of a joint probability space.

    Initialised using a probability distribution and random variables(or
    their indexed components) which should be a part of the resultant
    distribution.
    """

    def __new__(cls, dist, *rvs):
        if len(rvs) == 1 and iterable(rvs[0]):
            rvs = tuple(rvs[0])
        if not all(isinstance(rv, (Indexed, RandomSymbol)) for rv in rvs):
            raise ValueError(filldedent('''Marginal distribution can be
             intitialised only in terms of random variables or indexed random
             variables'''))
        rvs = Tuple.fromiter(rv for rv in rvs)
        if not isinstance(dist, JointDistribution) and len(random_symbols(dist)) == 0:
            return dist
        return Basic.__new__(cls, dist, rvs)

    def check(self):
        pass

    @property
    def set(self):
        rvs = [i for i in self.args[1] if isinstance(i, RandomSymbol)]
        return ProductSet(*[rv.pspace.set for rv in rvs])

    @property
    def symbols(self):
        rvs = self.args[1]
        return {rv.pspace.symbol for rv in rvs}

    def pdf(self, *x):
        expr, rvs = self.args[0], self.args[1]
        marginalise_out = [i for i in random_symbols(expr) if i not in rvs]
        if isinstance(expr, JointDistribution):
            count = len(expr.domain.args)
            x = Dummy('x', real=True)
            syms = tuple(Indexed(x, i) for i in count)
            expr = expr.pdf(syms)
        else:
            syms = tuple(rv.pspace.symbol if isinstance(rv, RandomSymbol) else rv.args[0] for rv in rvs)
        return Lambda(syms, self.compute_pdf(expr, marginalise_out))(*x)

    def compute_pdf(self, expr, rvs):
        for rv in rvs:
            lpdf = 1
            if isinstance(rv, RandomSymbol):
                lpdf = rv.pspace.pdf
            expr = self.marginalise_out(expr*lpdf, rv)
        return expr

    def marginalise_out(self, expr, rv):
        from sympy.concrete.summations import Sum
        if isinstance(rv, RandomSymbol):
            dom = rv.pspace.set
        elif isinstance(rv, Indexed):
            dom = rv.base.component_domain(
                rv.pspace.component_domain(rv.args[1]))
        expr = expr.xreplace({rv: rv.pspace.symbol})
        if rv.pspace.is_Continuous:
            #TODO: Modify to support integration
            #for all kinds of sets.
            expr = Integral(expr, (rv.pspace.symbol, dom))
        elif rv.pspace.is_Discrete:
            #incorporate this into `Sum`/`summation`
            if dom in (S.Integers, S.Naturals, S.Naturals0):
                dom = (dom.inf, dom.sup)
            expr = Sum(expr, (rv.pspace.symbol, dom))
        return expr

    def __call__(self, *args):
        return self.pdf(*args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\expintegrals.py ===
from .functions import defun, defun_wrapped

@defun_wrapped
def _erf_complex(ctx, z):
    z2 = ctx.square_exp_arg(z, -1)
    #z2 = -z**2
    v = (2/ctx.sqrt(ctx.pi))*z * ctx.hyp1f1((1,2),(3,2), z2)
    if not ctx._re(z):
        v = ctx._im(v)*ctx.j
    return v

@defun_wrapped
def _erfc_complex(ctx, z):
    if ctx.re(z) > 2:
        z2 = ctx.square_exp_arg(z)
        nz2 = ctx.fneg(z2, exact=True)
        v = ctx.exp(nz2)/ctx.sqrt(ctx.pi) * ctx.hyperu((1,2),(1,2), z2)
    else:
        v = 1 - ctx._erf_complex(z)
    if not ctx._re(z):
        v = 1+ctx._im(v)*ctx.j
    return v

@defun
def erf(ctx, z):
    z = ctx.convert(z)
    if ctx._is_real_type(z):
        try:
            return ctx._erf(z)
        except NotImplementedError:
            pass
    if ctx._is_complex_type(z) and not z.imag:
        try:
            return type(z)(ctx._erf(z.real))
        except NotImplementedError:
            pass
    return ctx._erf_complex(z)

@defun
def erfc(ctx, z):
    z = ctx.convert(z)
    if ctx._is_real_type(z):
        try:
            return ctx._erfc(z)
        except NotImplementedError:
            pass
    if ctx._is_complex_type(z) and not z.imag:
        try:
            return type(z)(ctx._erfc(z.real))
        except NotImplementedError:
            pass
    return ctx._erfc_complex(z)

@defun
def square_exp_arg(ctx, z, mult=1, reciprocal=False):
    prec = ctx.prec*4+20
    if reciprocal:
        z2 = ctx.fmul(z, z, prec=prec)
        z2 = ctx.fdiv(ctx.one, z2, prec=prec)
    else:
        z2 = ctx.fmul(z, z, prec=prec)
    if mult != 1:
        z2 = ctx.fmul(z2, mult, exact=True)
    return z2

@defun_wrapped
def erfi(ctx, z):
    if not z:
        return z
    z2 = ctx.square_exp_arg(z)
    v = (2/ctx.sqrt(ctx.pi)*z) * ctx.hyp1f1((1,2), (3,2), z2)
    if not ctx._re(z):
        v = ctx._im(v)*ctx.j
    return v

@defun_wrapped
def erfinv(ctx, x):
    xre = ctx._re(x)
    if (xre != x) or (xre < -1) or (xre > 1):
        return ctx.bad_domain("erfinv(x) is defined only for -1 <= x <= 1")
    x = xre
    #if ctx.isnan(x): return x
    if not x: return x
    if x == 1: return ctx.inf
    if x == -1: return ctx.ninf
    if abs(x) < 0.9:
        a = 0.53728*x**3 + 0.813198*x
    else:
        # An asymptotic formula
        u = ctx.ln(2/ctx.pi/(abs(x)-1)**2)
        a = ctx.sign(x) * ctx.sqrt(u - ctx.ln(u))/ctx.sqrt(2)
    ctx.prec += 10
    return ctx.findroot(lambda t: ctx.erf(t)-x, a)

@defun_wrapped
def npdf(ctx, x, mu=0, sigma=1):
    sigma = ctx.convert(sigma)
    return ctx.exp(-(x-mu)**2/(2*sigma**2)) / (sigma*ctx.sqrt(2*ctx.pi))

@defun_wrapped
def ncdf(ctx, x, mu=0, sigma=1):
    a = (x-mu)/(sigma*ctx.sqrt(2))
    if a < 0:
        return ctx.erfc(-a)/2
    else:
        return (1+ctx.erf(a))/2

@defun_wrapped
def betainc(ctx, a, b, x1=0, x2=1, regularized=False):
    if x1 == x2:
        v = 0
    elif not x1:
        if x1 == 0 and x2 == 1:
            v = ctx.beta(a, b)
        else:
            v = x2**a * ctx.hyp2f1(a, 1-b, a+1, x2) / a
    else:
        m, d = ctx.nint_distance(a)
        if m <= 0:
            if d < -ctx.prec:
                h = +ctx.eps
                ctx.prec *= 2
                a += h
            elif d < -4:
                ctx.prec -= d
        s1 = x2**a * ctx.hyp2f1(a,1-b,a+1,x2)
        s2 = x1**a * ctx.hyp2f1(a,1-b,a+1,x1)
        v = (s1 - s2) / a
    if regularized:
        v /= ctx.beta(a,b)
    return v

@defun
def gammainc(ctx, z, a=0, b=None, regularized=False):
    regularized = bool(regularized)
    z = ctx.convert(z)
    if a is None:
        a = ctx.zero
        lower_modified = False
    else:
        a = ctx.convert(a)
        lower_modified = a != ctx.zero
    if b is None:
        b = ctx.inf
        upper_modified = False
    else:
        b = ctx.convert(b)
        upper_modified = b != ctx.inf
    # Complete gamma function
    if not (upper_modified or lower_modified):
        if regularized:
            if ctx.re(z) < 0:
                return ctx.inf
            elif ctx.re(z) > 0:
                return ctx.one
            else:
                return ctx.nan
        return ctx.gamma(z)
    if a == b:
        return ctx.zero
    # Standardize
    if ctx.re(a) > ctx.re(b):
        return -ctx.gammainc(z, b, a, regularized)
    # Generalized gamma
    if upper_modified and lower_modified:
        return +ctx._gamma3(z, a, b, regularized)
    # Upper gamma
    elif lower_modified:
        return ctx._upper_gamma(z, a, regularized)
    # Lower gamma
    elif upper_modified:
        return ctx._lower_gamma(z, b, regularized)

@defun
def _lower_gamma(ctx, z, b, regularized=False):
    # Pole
    if ctx.isnpint(z):
        return type(z)(ctx.inf)
    G = [z] * regularized
    negb = ctx.fneg(b, exact=True)
    def h(z):
        T1 = [ctx.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
        return (T1,)
    return ctx.hypercomb(h, [z])

@defun
def _upper_gamma(ctx, z, a, regularized=False):
    # Fast integer case, when available
    if ctx.isint(z):
        try:
            if regularized:
                # Gamma pole
                if ctx.isnpint(z):
                    return type(z)(ctx.zero)
                orig = ctx.prec
                try:
                    ctx.prec += 10
                    return ctx._gamma_upper_int(z, a) / ctx.gamma(z)
                finally:
                    ctx.prec = orig
            else:
                return ctx._gamma_upper_int(z, a)
        except NotImplementedError:
            pass
    # hypercomb is unable to detect the exact zeros, so handle them here
    if z == 2 and a == -1:
        return (z+a)*0
    if z == 3 and (a == -1-1j or a == -1+1j):
        return (z+a)*0
    nega = ctx.fneg(a, exact=True)
    G = [z] * regularized
    # Use 2F0 series when possible; fall back to lower gamma representation
    try:
        def h(z):
            r = z-1
            return [([ctx.exp(nega), a], [1, r], [], G, [1, -r], [], 1/nega)]
        return ctx.hypercomb(h, [z], force_series=True)
    except ctx.NoConvergence:
        def h(z):
            T1 = [], [1, z-1], [z], G, [], [], 0
            T2 = [-ctx.exp(nega), a, z], [1, z, -1], [], G, [1], [1+z], a
            return T1, T2
        return ctx.hypercomb(h, [z])

@defun
def _gamma3(ctx, z, a, b, regularized=False):
    pole = ctx.isnpint(z)
    if regularized and pole:
        return ctx.zero
    try:
        ctx.prec += 15
        # We don't know in advance whether it's better to write as a difference
        # of lower or upper gamma functions, so try both
        T1 = ctx.gammainc(z, a, regularized=regularized)
        T2 = ctx.gammainc(z, b, regularized=regularized)
        R = T1 - T2
        if ctx.mag(R) - max(ctx.mag(T1), ctx.mag(T2)) > -10:
            return R
        if not pole:
            T1 = ctx.gammainc(z, 0, b, regularized=regularized)
            T2 = ctx.gammainc(z, 0, a, regularized=regularized)
            R = T1 - T2
            # May be ok, but should probably at least print a warning
            # about possible cancellation
            if 1: #ctx.mag(R) - max(ctx.mag(T1), ctx.mag(T2)) > -10:
                return R
    finally:
        ctx.prec -= 15
    raise NotImplementedError

@defun_wrapped
def expint(ctx, n, z):
    if ctx.isint(n) and ctx._is_real_type(z):
        try:
            return ctx._expint_int(n, z)
        except NotImplementedError:
            pass
    if ctx.isnan(n) or ctx.isnan(z):
        return z*n
    if z == ctx.inf:
        return 1/z
    if z == 0:
        # integral from 1 to infinity of t^n
        if ctx.re(n) <= 1:
            # TODO: reasonable sign of infinity
            return type(z)(ctx.inf)
        else:
            return ctx.one/(n-1)
    if n == 0:
        return ctx.exp(-z)/z
    if n == -1:
        return ctx.exp(-z)*(z+1)/z**2
    return z**(n-1) * ctx.gammainc(1-n, z)

@defun_wrapped
def li(ctx, z, offset=False):
    if offset:
        if z == 2:
            return ctx.zero
        return ctx.ei(ctx.ln(z)) - ctx.ei(ctx.ln2)
    if not z:
        return z
    if z == 1:
        return ctx.ninf
    return ctx.ei(ctx.ln(z))

@defun
def ei(ctx, z):
    try:
        return ctx._ei(z)
    except NotImplementedError:
        return ctx._ei_generic(z)

@defun_wrapped
def _ei_generic(ctx, z):
    # Note: the following is currently untested because mp and fp
    # both use special-case ei code
    if z == ctx.inf:
        return z
    if z == ctx.ninf:
        return ctx.zero
    if ctx.mag(z) > 1:
        try:
            r = ctx.one/z
            v = ctx.exp(z)*ctx.hyper([1,1],[],r,
                maxterms=ctx.prec, force_series=True)/z
            im = ctx._im(z)
            if im > 0:
                v += ctx.pi*ctx.j
            if im < 0:
                v -= ctx.pi*ctx.j
            return v
        except ctx.NoConvergence:
            pass
    v = z*ctx.hyp2f2(1,1,2,2,z) + ctx.euler
    if ctx._im(z):
        v += 0.5*(ctx.log(z) - ctx.log(ctx.one/z))
    else:
        v += ctx.log(abs(z))
    return v

@defun
def e1(ctx, z):
    try:
        return ctx._e1(z)
    except NotImplementedError:
        return ctx.expint(1, z)

@defun
def ci(ctx, z):
    try:
        return ctx._ci(z)
    except NotImplementedError:
        return ctx._ci_generic(z)

@defun_wrapped
def _ci_generic(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf: return ctx.zero
        if z == ctx.ninf: return ctx.pi*1j
    jz = ctx.fmul(ctx.j,z,exact=True)
    njz = ctx.fneg(jz,exact=True)
    v = 0.5*(ctx.ei(jz) + ctx.ei(njz))
    zreal = ctx._re(z)
    zimag = ctx._im(z)
    if zreal == 0:
        if zimag > 0: v += ctx.pi*0.5j
        if zimag < 0: v -= ctx.pi*0.5j
    if zreal < 0:
        if zimag >= 0: v += ctx.pi*1j
        if zimag <  0: v -= ctx.pi*1j
    if ctx._is_real_type(z) and zreal > 0:
        v = ctx._re(v)
    return v

@defun
def si(ctx, z):
    try:
        return ctx._si(z)
    except NotImplementedError:
        return ctx._si_generic(z)

@defun_wrapped
def _si_generic(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf: return 0.5*ctx.pi
        if z == ctx.ninf: return -0.5*ctx.pi
    # Suffers from cancellation near 0
    if ctx.mag(z) >= -1:
        jz = ctx.fmul(ctx.j,z,exact=True)
        njz = ctx.fneg(jz,exact=True)
        v = (-0.5j)*(ctx.ei(jz) - ctx.ei(njz))
        zreal = ctx._re(z)
        if zreal > 0:
            v -= 0.5*ctx.pi
        if zreal < 0:
            v += 0.5*ctx.pi
        if ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    else:
        return z*ctx.hyp1f2((1,2),(3,2),(3,2),-0.25*z*z)

@defun_wrapped
def chi(ctx, z):
    nz = ctx.fneg(z, exact=True)
    v = 0.5*(ctx.ei(z) + ctx.ei(nz))
    zreal = ctx._re(z)
    zimag = ctx._im(z)
    if zimag > 0:
        v += ctx.pi*0.5j
    elif zimag < 0:
        v -= ctx.pi*0.5j
    elif zreal < 0:
        v += ctx.pi*1j
    return v

@defun_wrapped
def shi(ctx, z):
    # Suffers from cancellation near 0
    if ctx.mag(z) >= -1:
        nz = ctx.fneg(z, exact=True)
        v = 0.5*(ctx.ei(z) - ctx.ei(nz))
        zimag = ctx._im(z)
        if zimag > 0: v -= 0.5j*ctx.pi
        if zimag < 0: v += 0.5j*ctx.pi
        return v
    else:
        return z * ctx.hyp1f2((1,2),(3,2),(3,2),0.25*z*z)

@defun_wrapped
def fresnels(ctx, z):
    if z == ctx.inf:
        return ctx.mpf(0.5)
    if z == ctx.ninf:
        return ctx.mpf(-0.5)
    return ctx.pi*z**3/6*ctx.hyp1f2((3,4),(3,2),(7,4),-ctx.pi**2*z**4/16)

@defun_wrapped
def fresnelc(ctx, z):
    if z == ctx.inf:
        return ctx.mpf(0.5)
    if z == ctx.ninf:
        return ctx.mpf(-0.5)
    return z*ctx.hyp1f2((1,4),(1,2),(5,4),-ctx.pi**2*z**4/16)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\fill.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Alias,
    Bool,
    Integer,
    Set,
    NoneSet,
    Typed,
    MinMax,
)
from openpyxl.descriptors.excel import (
    Relation,
    Percentage,
)
from openpyxl.descriptors.nested import NestedNoneSet, NestedValue
from openpyxl.descriptors.sequence import NestedSequence
from openpyxl.descriptors.excel import ExtensionList as OfficeArtExtensionList
from openpyxl.xml.constants import DRAWING_NS

from .colors import (
    ColorChoice,
    HSLColor,
    SystemColor,
    SchemeColor,
    PRESET_COLORS,
    RGBPercent,
)

from .effect import (
    AlphaBiLevelEffect,
    AlphaCeilingEffect,
    AlphaFloorEffect,
    AlphaInverseEffect,
    AlphaModulateEffect,
    AlphaModulateFixedEffect,
    AlphaReplaceEffect,
    BiLevelEffect,
    BlurEffect,
    ColorChangeEffect,
    ColorReplaceEffect,
    DuotoneEffect,
    FillOverlayEffect,
    GrayscaleEffect,
    HSLEffect,
    LuminanceEffect,
    TintEffect,
)

"""
Fill elements from drawing main schema
"""

class PatternFillProperties(Serialisable):

    tagname = "pattFill"
    namespace = DRAWING_NS

    prst = NoneSet(values=(['pct5', 'pct10', 'pct20', 'pct25', 'pct30',
                            'pct40', 'pct50', 'pct60', 'pct70', 'pct75', 'pct80', 'pct90', 'horz',
                            'vert', 'ltHorz', 'ltVert', 'dkHorz', 'dkVert', 'narHorz', 'narVert',
                            'dashHorz', 'dashVert', 'cross', 'dnDiag', 'upDiag', 'ltDnDiag',
                            'ltUpDiag', 'dkDnDiag', 'dkUpDiag', 'wdDnDiag', 'wdUpDiag', 'dashDnDiag',
                            'dashUpDiag', 'diagCross', 'smCheck', 'lgCheck', 'smGrid', 'lgGrid',
                            'dotGrid', 'smConfetti', 'lgConfetti', 'horzBrick', 'diagBrick',
                            'solidDmnd', 'openDmnd', 'dotDmnd', 'plaid', 'sphere', 'weave', 'divot',
                            'shingle', 'wave', 'trellis', 'zigZag']))
    preset = Alias("prst")
    fgClr = Typed(expected_type=ColorChoice, allow_none=True)
    foreground = Alias("fgClr")
    bgClr = Typed(expected_type=ColorChoice, allow_none=True)
    background = Alias("bgClr")

    __elements__ = ("fgClr", "bgClr")

    def __init__(self,
                 prst=None,
                 fgClr=None,
                 bgClr=None,
                ):
        self.prst = prst
        self.fgClr = fgClr
        self.bgClr = bgClr


class RelativeRect(Serialisable):

    tagname = "rect"
    namespace = DRAWING_NS

    l = Percentage(allow_none=True)
    left = Alias('l')
    t = Percentage(allow_none=True)
    top = Alias('t')
    r = Percentage(allow_none=True)
    right = Alias('r')
    b = Percentage(allow_none=True)
    bottom = Alias('b')

    def __init__(self,
                 l=None,
                 t=None,
                 r=None,
                 b=None,
                ):
        self.l = l
        self.t = t
        self.r = r
        self.b = b


class StretchInfoProperties(Serialisable):

    tagname = "stretch"
    namespace = DRAWING_NS

    fillRect = Typed(expected_type=RelativeRect, allow_none=True)

    def __init__(self,
                 fillRect=RelativeRect(),
                ):
        self.fillRect = fillRect


class GradientStop(Serialisable):

    tagname = "gs"
    namespace = DRAWING_NS

    pos = MinMax(min=0, max=100000, allow_none=True)
    # Color Choice Group
    scrgbClr = Typed(expected_type=RGBPercent, allow_none=True)
    RGBPercent = Alias('scrgbClr')
    srgbClr = NestedValue(expected_type=str, allow_none=True) # needs pattern and can have transform
    RGB = Alias('srgbClr')
    hslClr = Typed(expected_type=HSLColor, allow_none=True)
    sysClr = Typed(expected_type=SystemColor, allow_none=True)
    schemeClr = Typed(expected_type=SchemeColor, allow_none=True)
    prstClr = NestedNoneSet(values=PRESET_COLORS)

    __elements__ = ('scrgbClr', 'srgbClr', 'hslClr', 'sysClr', 'schemeClr', 'prstClr')

    def __init__(self,
                 pos=None,
                 scrgbClr=None,
                 srgbClr=None,
                 hslClr=None,
                 sysClr=None,
                 schemeClr=None,
                 prstClr=None,
                ):
        if pos is None:
            pos = 0
        self.pos = pos

        self.scrgbClr = scrgbClr
        self.srgbClr = srgbClr
        self.hslClr = hslClr
        self.sysClr = sysClr
        self.schemeClr = schemeClr
        self.prstClr = prstClr


class LinearShadeProperties(Serialisable):

    tagname = "lin"
    namespace = DRAWING_NS

    ang = Integer()
    scaled = Bool(allow_none=True)

    def __init__(self,
                 ang=None,
                 scaled=None,
                ):
        self.ang = ang
        self.scaled = scaled


class PathShadeProperties(Serialisable):

    tagname = "path"
    namespace = DRAWING_NS

    path = Set(values=(['shape', 'circle', 'rect']))
    fillToRect = Typed(expected_type=RelativeRect, allow_none=True)

    def __init__(self,
                 path=None,
                 fillToRect=None,
                ):
        self.path = path
        self.fillToRect = fillToRect


class GradientFillProperties(Serialisable):

    tagname = "gradFill"
    namespace = DRAWING_NS

    flip = NoneSet(values=(['x', 'y', 'xy']))
    rotWithShape = Bool(allow_none=True)

    gsLst = NestedSequence(expected_type=GradientStop, count=False)
    stop_list = Alias("gsLst")

    lin = Typed(expected_type=LinearShadeProperties, allow_none=True)
    linear = Alias("lin")
    path = Typed(expected_type=PathShadeProperties, allow_none=True)

    tileRect = Typed(expected_type=RelativeRect, allow_none=True)

    __elements__ = ('gsLst', 'lin', 'path', 'tileRect')

    def __init__(self,
                 flip=None,
                 rotWithShape=None,
                 gsLst=(),
                 lin=None,
                 path=None,
                 tileRect=None,
                ):
        self.flip = flip
        self.rotWithShape = rotWithShape
        self.gsLst = gsLst
        self.lin = lin
        self.path = path
        self.tileRect = tileRect


class SolidColorFillProperties(Serialisable):

    tagname = "solidFill"

    # uses element group EG_ColorChoice
    scrgbClr = Typed(expected_type=RGBPercent, allow_none=True)
    RGBPercent = Alias('scrgbClr')
    srgbClr = NestedValue(expected_type=str, allow_none=True) # needs pattern and can have transform
    RGB = Alias('srgbClr')
    hslClr = Typed(expected_type=HSLColor, allow_none=True)
    sysClr = Typed(expected_type=SystemColor, allow_none=True)
    schemeClr = Typed(expected_type=SchemeColor, allow_none=True)
    prstClr = NestedNoneSet(values=PRESET_COLORS)

    __elements__ = ('scrgbClr', 'srgbClr', 'hslClr', 'sysClr', 'schemeClr', 'prstClr')

    def __init__(self,
                 scrgbClr=None,
                 srgbClr=None,
                 hslClr=None,
                 sysClr=None,
                 schemeClr=None,
                 prstClr=None,
                ):
        self.scrgbClr = scrgbClr
        self.srgbClr = srgbClr
        self.hslClr = hslClr
        self.sysClr = sysClr
        self.schemeClr = schemeClr
        self.prstClr = prstClr


class Blip(Serialisable):

    tagname = "blip"
    namespace = DRAWING_NS

    # Using attribute groupAG_Blob
    cstate = NoneSet(values=(['email', 'screen', 'print', 'hqprint']))
    embed = Relation() # rId
    link = Relation() # hyperlink
    noGrp = Bool(allow_none=True)
    noSelect = Bool(allow_none=True)
    noRot = Bool(allow_none=True)
    noChangeAspect = Bool(allow_none=True)
    noMove = Bool(allow_none=True)
    noResize = Bool(allow_none=True)
    noEditPoints = Bool(allow_none=True)
    noAdjustHandles = Bool(allow_none=True)
    noChangeArrowheads = Bool(allow_none=True)
    noChangeShapeType = Bool(allow_none=True)
    # some elements are choice
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)
    alphaBiLevel = Typed(expected_type=AlphaBiLevelEffect, allow_none=True)
    alphaCeiling = Typed(expected_type=AlphaCeilingEffect, allow_none=True)
    alphaFloor = Typed(expected_type=AlphaFloorEffect, allow_none=True)
    alphaInv = Typed(expected_type=AlphaInverseEffect, allow_none=True)
    alphaMod = Typed(expected_type=AlphaModulateEffect, allow_none=True)
    alphaModFix = Typed(expected_type=AlphaModulateFixedEffect, allow_none=True)
    alphaRepl = Typed(expected_type=AlphaReplaceEffect, allow_none=True)
    biLevel = Typed(expected_type=BiLevelEffect, allow_none=True)
    blur = Typed(expected_type=BlurEffect, allow_none=True)
    clrChange = Typed(expected_type=ColorChangeEffect, allow_none=True)
    clrRepl = Typed(expected_type=ColorReplaceEffect, allow_none=True)
    duotone = Typed(expected_type=DuotoneEffect, allow_none=True)
    fillOverlay = Typed(expected_type=FillOverlayEffect, allow_none=True)
    grayscl = Typed(expected_type=GrayscaleEffect, allow_none=True)
    hsl = Typed(expected_type=HSLEffect, allow_none=True)
    lum = Typed(expected_type=LuminanceEffect, allow_none=True)
    tint = Typed(expected_type=TintEffect, allow_none=True)

    __elements__ = ('alphaBiLevel', 'alphaCeiling', 'alphaFloor', 'alphaInv',
                    'alphaMod', 'alphaModFix', 'alphaRepl', 'biLevel', 'blur', 'clrChange',
                    'clrRepl', 'duotone', 'fillOverlay', 'grayscl', 'hsl', 'lum', 'tint')

    def __init__(self,
                 cstate=None,
                 embed=None,
                 link=None,
                 noGrp=None,
                 noSelect=None,
                 noRot=None,
                 noChangeAspect=None,
                 noMove=None,
                 noResize=None,
                 noEditPoints=None,
                 noAdjustHandles=None,
                 noChangeArrowheads=None,
                 noChangeShapeType=None,
                 extLst=None,
                 alphaBiLevel=None,
                 alphaCeiling=None,
                 alphaFloor=None,
                 alphaInv=None,
                 alphaMod=None,
                 alphaModFix=None,
                 alphaRepl=None,
                 biLevel=None,
                 blur=None,
                 clrChange=None,
                 clrRepl=None,
                 duotone=None,
                 fillOverlay=None,
                 grayscl=None,
                 hsl=None,
                 lum=None,
                 tint=None,
                ):
        self.cstate = cstate
        self.embed = embed
        self.link = link
        self.noGrp = noGrp
        self.noSelect = noSelect
        self.noRot = noRot
        self.noChangeAspect = noChangeAspect
        self.noMove = noMove
        self.noResize = noResize
        self.noEditPoints = noEditPoints
        self.noAdjustHandles = noAdjustHandles
        self.noChangeArrowheads = noChangeArrowheads
        self.noChangeShapeType = noChangeShapeType
        self.extLst = extLst
        self.alphaBiLevel = alphaBiLevel
        self.alphaCeiling = alphaCeiling
        self.alphaFloor = alphaFloor
        self.alphaInv = alphaInv
        self.alphaMod = alphaMod
        self.alphaModFix = alphaModFix
        self.alphaRepl = alphaRepl
        self.biLevel = biLevel
        self.blur = blur
        self.clrChange = clrChange
        self.clrRepl = clrRepl
        self.duotone = duotone
        self.fillOverlay = fillOverlay
        self.grayscl = grayscl
        self.hsl = hsl
        self.lum = lum
        self.tint = tint


class TileInfoProperties(Serialisable):

    tx = Integer(allow_none=True)
    ty = Integer(allow_none=True)
    sx = Integer(allow_none=True)
    sy = Integer(allow_none=True)
    flip = NoneSet(values=(['x', 'y', 'xy']))
    algn = Set(values=(['tl', 't', 'tr', 'l', 'ctr', 'r', 'bl', 'b', 'br']))

    def __init__(self,
                 tx=None,
                 ty=None,
                 sx=None,
                 sy=None,
                 flip=None,
                 algn=None,
                ):
        self.tx = tx
        self.ty = ty
        self.sx = sx
        self.sy = sy
        self.flip = flip
        self.algn = algn


class BlipFillProperties(Serialisable):

    tagname = "blipFill"

    dpi = Integer(allow_none=True)
    rotWithShape = Bool(allow_none=True)

    blip = Typed(expected_type=Blip, allow_none=True)
    srcRect = Typed(expected_type=RelativeRect, allow_none=True)
    tile = Typed(expected_type=TileInfoProperties, allow_none=True)
    stretch = Typed(expected_type=StretchInfoProperties, allow_none=True)

    __elements__ = ("blip", "srcRect", "tile", "stretch")

    def __init__(self,
                 dpi=None,
                 rotWithShape=None,
                 blip=None,
                 tile=None,
                 stretch=StretchInfoProperties(),
                 srcRect=None,
                ):
        self.dpi = dpi
        self.rotWithShape = rotWithShape
        self.blip = blip
        self.tile = tile
        self.stretch = stretch
        self.srcRect = srcRect

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\logic.py ===
"""Logic expressions handling

NOTE
----

at present this is mainly needed for facts.py, feel free however to improve
this stuff for general purpose.
"""

from __future__ import annotations
from typing import Optional

# Type of a fuzzy bool
FuzzyBool = Optional[bool]


def _torf(args):
    """Return True if all args are True, False if they
    are all False, else None.

    >>> from sympy.core.logic import _torf
    >>> _torf((True, True))
    True
    >>> _torf((False, False))
    False
    >>> _torf((True, False))
    """
    sawT = sawF = False
    for a in args:
        if a is True:
            if sawF:
                return
            sawT = True
        elif a is False:
            if sawT:
                return
            sawF = True
        else:
            return
    return sawT


def _fuzzy_group(args, quick_exit=False):
    """Return True if all args are True, None if there is any None else False
    unless ``quick_exit`` is True (then return None as soon as a second False
    is seen.

     ``_fuzzy_group`` is like ``fuzzy_and`` except that it is more
    conservative in returning a False, waiting to make sure that all
    arguments are True or False and returning None if any arguments are
    None. It also has the capability of permiting only a single False and
    returning None if more than one is seen. For example, the presence of a
    single transcendental amongst rationals would indicate that the group is
    no longer rational; but a second transcendental in the group would make the
    determination impossible.


    Examples
    ========

    >>> from sympy.core.logic import _fuzzy_group

    By default, multiple Falses mean the group is broken:

    >>> _fuzzy_group([False, False, True])
    False

    If multiple Falses mean the group status is unknown then set
    `quick_exit` to True so None can be returned when the 2nd False is seen:

    >>> _fuzzy_group([False, False, True], quick_exit=True)

    But if only a single False is seen then the group is known to
    be broken:

    >>> _fuzzy_group([False, True, True], quick_exit=True)
    False

    """
    saw_other = False
    for a in args:
        if a is True:
            continue
        if a is None:
            return
        if quick_exit and saw_other:
            return
        saw_other = True
    return not saw_other


def fuzzy_bool(x):
    """Return True, False or None according to x.

    Whereas bool(x) returns True or False, fuzzy_bool allows
    for the None value and non-false values (which become None), too.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_bool
    >>> from sympy.abc import x
    >>> fuzzy_bool(x), fuzzy_bool(None)
    (None, None)
    >>> bool(x), bool(None)
    (True, False)

    """
    if x is None:
        return None
    if x in (True, False):
        return bool(x)


def fuzzy_and(args):
    """Return True (all True), False (any False) or None.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_and
    >>> from sympy import Dummy

    If you had a list of objects to test the commutivity of
    and you want the fuzzy_and logic applied, passing an
    iterator will allow the commutativity to only be computed
    as many times as necessary. With this list, False can be
    returned after analyzing the first symbol:

    >>> syms = [Dummy(commutative=False), Dummy()]
    >>> fuzzy_and(s.is_commutative for s in syms)
    False

    That False would require less work than if a list of pre-computed
    items was sent:

    >>> fuzzy_and([s.is_commutative for s in syms])
    False
    """

    rv = True
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is False:
            return False
        if rv:  # this will stop updating if a None is ever trapped
            rv = ai
    return rv


def fuzzy_not(v):
    """
    Not in fuzzy logic

    Return None if `v` is None else `not v`.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_not
    >>> fuzzy_not(True)
    False
    >>> fuzzy_not(None)
    >>> fuzzy_not(False)
    True

    """
    if v is None:
        return v
    else:
        return not v


def fuzzy_or(args):
    """
    Or in fuzzy logic. Returns True (any True), False (all False), or None

    See the docstrings of fuzzy_and and fuzzy_not for more info.  fuzzy_or is
    related to the two by the standard De Morgan's law.

    >>> from sympy.core.logic import fuzzy_or
    >>> fuzzy_or([True, False])
    True
    >>> fuzzy_or([True, None])
    True
    >>> fuzzy_or([False, False])
    False
    >>> print(fuzzy_or([False, None]))
    None

    """
    rv = False
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is True:
            return True
        if rv is False:  # this will stop updating if a None is ever trapped
            rv = ai
    return rv


def fuzzy_xor(args):
    """Return None if any element of args is not True or False, else
    True (if there are an odd number of True elements), else False."""
    t = 0
    for a in args:
        ai = fuzzy_bool(a)
        if ai:
            t += 1
        elif ai is None:
            return
    return t % 2 == 1


def fuzzy_nand(args):
    """Return False if all args are True, True if they are all False,
    else None."""
    return fuzzy_not(fuzzy_and(args))


class Logic:
    """Logical expression"""
    # {} 'op' -> LogicClass
    op_2class: dict[str, type[Logic]] = {}

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __getnewargs__(self):
        return self.args

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(a, b):
        if not isinstance(b, type(a)):
            return False
        else:
            return a.args == b.args

    def __ne__(a, b):
        if not isinstance(b, type(a)):
            return True
        else:
            return a.args != b.args

    def __lt__(self, other):
        if self.__cmp__(other) == -1:
            return True
        return False

    def __cmp__(self, other):
        if type(self) is not type(other):
            a = str(type(self))
            b = str(type(other))
        else:
            a = self.args
            b = other.args
        return (a > b) - (a < b)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join(str(a) for a in self.args))

    __repr__ = __str__

    @staticmethod
    def fromstring(text):
        """Logic from string with space around & and | but none after !.

           e.g.

           !a & b | c
        """
        lexpr = None  # current logical expression
        schedop = None  # scheduled operation
        for term in text.split():
            # operation symbol
            if term in '&|':
                if schedop is not None:
                    raise ValueError(
                        'double op forbidden: "%s %s"' % (term, schedop))
                if lexpr is None:
                    raise ValueError(
                        '%s cannot be in the beginning of expression' % term)
                schedop = term
                continue
            if '&' in term or '|' in term:
                raise ValueError('& and | must have space around them')
            if term[0] == '!':
                if len(term) == 1:
                    raise ValueError('do not include space after "!"')
                term = Not(term[1:])

            # already scheduled operation, e.g. '&'
            if schedop:
                lexpr = Logic.op_2class[schedop](lexpr, term)
                schedop = None
                continue

            # this should be atom
            if lexpr is not None:
                raise ValueError(
                    'missing op between "%s" and "%s"' % (lexpr, term))

            lexpr = term

        # let's check that we ended up in correct state
        if schedop is not None:
            raise ValueError('premature end-of-expression in "%s"' % text)
        if lexpr is None:
            raise ValueError('"%s" is empty' % text)

        # everything looks good now
        return lexpr


class AndOr_Base(Logic):

    def __new__(cls, *args):
        bargs = []
        for a in args:
            if a == cls.op_x_notx:
                return a
            elif a == (not cls.op_x_notx):
                continue    # skip this argument
            bargs.append(a)

        args = sorted(set(cls.flatten(bargs)), key=hash)

        for a in args:
            if Not(a) in args:
                return cls.op_x_notx

        if len(args) == 1:
            return args.pop()
        elif len(args) == 0:
            return not cls.op_x_notx

        return Logic.__new__(cls, *args)

    @classmethod
    def flatten(cls, args):
        # quick-n-dirty flattening for And and Or
        args_queue = list(args)
        res = []

        while True:
            try:
                arg = args_queue.pop(0)
            except IndexError:
                break
            if isinstance(arg, Logic):
                if isinstance(arg, cls):
                    args_queue.extend(arg.args)
                    continue
            res.append(arg)

        args = tuple(res)
        return args


class And(AndOr_Base):
    op_x_notx = False

    def _eval_propagate_not(self):
        # !(a&b&c ...) == !a | !b | !c ...
        return Or(*[Not(a) for a in self.args])

    # (a|b|...) & c == (a&c) | (b&c) | ...
    def expand(self):

        # first locate Or
        for i, arg in enumerate(self.args):
            if isinstance(arg, Or):
                arest = self.args[:i] + self.args[i + 1:]

                orterms = [And(*(arest + (a,))) for a in arg.args]
                for j in range(len(orterms)):
                    if isinstance(orterms[j], Logic):
                        orterms[j] = orterms[j].expand()

                res = Or(*orterms)
                return res

        return self


class Or(AndOr_Base):
    op_x_notx = True

    def _eval_propagate_not(self):
        # !(a|b|c ...) == !a & !b & !c ...
        return And(*[Not(a) for a in self.args])


class Not(Logic):

    def __new__(cls, arg):
        if isinstance(arg, str):
            return Logic.__new__(cls, arg)

        elif isinstance(arg, bool):
            return not arg
        elif isinstance(arg, Not):
            return arg.args[0]

        elif isinstance(arg, Logic):
            # XXX this is a hack to expand right from the beginning
            arg = arg._eval_propagate_not()
            return arg

        else:
            raise ValueError('Not: unknown argument %r' % (arg,))

    @property
    def arg(self):
        return self.args[0]


Logic.op_2class['&'] = And
Logic.op_2class['|'] = Or
Logic.op_2class['!'] = Not

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\transforms.py ===
"""
Discrete Fourier Transform, Number Theoretic Transform,
Walsh Hadamard Transform, Mobius Transform
"""

from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int


#----------------------------------------------------------------------------#
#                                                                            #
#                         Discrete Fourier Transform                         #
#                                                                            #
#----------------------------------------------------------------------------#

def _fourier_transform(seq, dps, inverse=False):
    """Utility function for the Discrete Fourier Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of numeric coefficients "
                        "for Fourier Transform")

    a = [sympify(arg) for arg in seq]
    if any(x.has(Symbol) for x in a):
        raise ValueError("Expected non-symbolic coefficients")

    n = len(a)
    if n < 2:
        return a

    b = n.bit_length() - 1
    if n&(n - 1): # not a power of 2
        b += 1
        n = 2**b

    a += [S.Zero]*(n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    ang = -2*pi/n if inverse else 2*pi/n

    if dps is not None:
        ang = ang.evalf(dps + 2)

    w = [cos(ang*i) + I*sin(ang*i) for i in range(n // 2)]

    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], expand_mul(a[i + j + hf]*w[ut * j])
                a[i + j], a[i + j + hf] = u + v, u - v
        h *= 2

    if inverse:
        a = [(x/n).evalf(dps) for x in a] if dps is not None \
                            else [x/n for x in a]

    return a


def fft(seq, dps=None):
    r"""
    Performs the Discrete Fourier Transform (**DFT**) in the complex domain.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FFT* requires the number of sample points to be a power of 2.

    This method should be used with default arguments only for short sequences
    as the complexity of expressions increases with the size of the sequence.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import fft, ifft

    >>> fft([1, 2, 3, 4])
    [10, -2 - 2*I, -2, -2 + 2*I]
    >>> ifft(_)
    [1, 2, 3, 4]

    >>> ifft([1, 2, 3, 4])
    [5/2, -1/2 + I/2, -1/2, -1/2 - I/2]
    >>> fft(_)
    [1, 2, 3, 4]

    >>> ifft([1, 7, 3, 4], dps=15)
    [3.75, -0.5 - 0.75*I, -1.75, -0.5 + 0.75*I]
    >>> fft(_)
    [1.0, 7.0, 3.0, 4.0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
    .. [2] https://mathworld.wolfram.com/FastFourierTransform.html

    """

    return _fourier_transform(seq, dps=dps)


def ifft(seq, dps=None):
    return _fourier_transform(seq, dps=dps, inverse=True)

ifft.__doc__ = fft.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                         Number Theoretic Transform                         #
#                                                                            #
#----------------------------------------------------------------------------#

def _number_theoretic_transform(seq, prime, inverse=False):
    """Utility function for the Number Theoretic Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of integer coefficients "
                        "for Number Theoretic Transform")

    p = as_int(prime)
    if not isprime(p):
        raise ValueError("Expected prime modulus for "
                        "Number Theoretic Transform")

    a = [as_int(x) % p for x in seq]

    n = len(a)
    if n < 1:
        return a

    b = n.bit_length() - 1
    if n&(n - 1):
        b += 1
        n = 2**b

    if (p - 1) % n:
        raise ValueError("Expected prime modulus of the form (m*2**k + 1)")

    a += [0]*(n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    pr = primitive_root(p)

    rt = pow(pr, (p - 1) // n, p)
    if inverse:
        rt = pow(rt, p - 2, p)

    w = [1]*(n // 2)
    for i in range(1, n // 2):
        w[i] = w[i - 1]*rt % p

    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]*w[ut * j]
                a[i + j], a[i + j + hf] = (u + v) % p, (u - v) % p
        h *= 2

    if inverse:
        rv = pow(n, p - 2, p)
        a = [x*rv % p for x in a]

    return a


def ntt(seq, prime):
    r"""
    Performs the Number Theoretic Transform (**NTT**), which specializes the
    Discrete Fourier Transform (**DFT**) over quotient ring `Z/pZ` for prime
    `p` instead of complex numbers `C`.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 NTT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy import ntt, intt
    >>> ntt([1, 2, 3, 4], prime=3*2**8 + 1)
    [10, 643, 767, 122]
    >>> intt(_, 3*2**8 + 1)
    [1, 2, 3, 4]
    >>> intt([1, 2, 3, 4], prime=3*2**8 + 1)
    [387, 415, 384, 353]
    >>> ntt(_, prime=3*2**8 + 1)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] http://www.apfloat.org/ntt.html
    .. [2] https://mathworld.wolfram.com/NumberTheoreticTransform.html
    .. [3] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    return _number_theoretic_transform(seq, prime=prime)


def intt(seq, prime):
    return _number_theoretic_transform(seq, prime=prime, inverse=True)

intt.__doc__ = ntt.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                          Walsh Hadamard Transform                          #
#                                                                            #
#----------------------------------------------------------------------------#

def _walsh_hadamard_transform(seq, inverse=False):
    """Utility function for the Walsh Hadamard Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of coefficients "
                        "for Walsh Hadamard Transform")

    a = [sympify(arg) for arg in seq]
    n = len(a)
    if n < 2:
        return a

    if n&(n - 1):
        n = 2**n.bit_length()

    a += [S.Zero]*(n - len(a))
    h = 2
    while h <= n:
        hf = h // 2
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]
                a[i + j], a[i + j + hf] = u + v, u - v
        h *= 2

    if inverse:
        a = [x/n for x in a]

    return a


def fwht(seq):
    r"""
    Performs the Walsh Hadamard Transform (**WHT**), and uses Hadamard
    ordering for the sequence.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which WHT is to be applied.

    Examples
    ========

    >>> from sympy import fwht, ifwht
    >>> fwht([4, 2, 2, 0, 0, 2, -2, 0])
    [8, 0, 8, 0, 8, 8, 0, 0]
    >>> ifwht(_)
    [4, 2, 2, 0, 0, 2, -2, 0]

    >>> ifwht([19, -1, 11, -9, -7, 13, -15, 5])
    [2, 0, 4, 0, 3, 10, 0, 0]
    >>> fwht(_)
    [19, -1, 11, -9, -7, 13, -15, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_transform
    .. [2] https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

    """

    return _walsh_hadamard_transform(seq)


def ifwht(seq):
    return _walsh_hadamard_transform(seq, inverse=True)

ifwht.__doc__ = fwht.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                    Mobius Transform for Subset Lattice                     #
#                                                                            #
#----------------------------------------------------------------------------#

def _mobius_transform(seq, sgn, subset):
    r"""Utility function for performing Mobius Transform using
    Yate's Dynamic Programming method"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of coefficients")

    a = [sympify(arg) for arg in seq]

    n = len(a)
    if n < 2:
        return a

    if n&(n - 1):
        n = 2**n.bit_length()

    a += [S.Zero]*(n - len(a))

    if subset:
        i = 1
        while i < n:
            for j in range(n):
                if j & i:
                    a[j] += sgn*a[j ^ i]
            i *= 2

    else:
        i = 1
        while i < n:
            for j in range(n):
                if j & i:
                    continue
                a[j] += sgn*a[j ^ i]
            i *= 2

    return a


def mobius_transform(seq, subset=True):
    r"""
    Performs the Mobius Transform for subset lattice with indices of
    sequence as bitmasks.

    The indices of each argument, considered as bit strings, correspond
    to subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset/superset based on bitmasks (indices) requires
    the size of sequence to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which Mobius Transform is to be applied.
    subset : bool
        Specifies if Mobius Transform is applied by enumerating subsets
        or supersets of the given set.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy import mobius_transform, inverse_mobius_transform
    >>> x, y, z = symbols('x y z')

    >>> mobius_transform([x, y, z])
    [x, x + y, x + z, x + y + z]
    >>> inverse_mobius_transform(_)
    [x, y, z, 0]

    >>> mobius_transform([x, y, z], subset=False)
    [x + y + z, y, z, 0]
    >>> inverse_mobius_transform(_, subset=False)
    [x, y, z, 0]

    >>> mobius_transform([1, 2, 3, 4])
    [1, 3, 4, 10]
    >>> inverse_mobius_transform(_)
    [1, 2, 3, 4]
    >>> mobius_transform([1, 2, 3, 4], subset=False)
    [10, 6, 7, 4]
    >>> inverse_mobius_transform(_, subset=False)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_inversion_formula
    .. [2] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf
    .. [3] https://arxiv.org/pdf/1211.0189.pdf

    """

    return _mobius_transform(seq, sgn=+1, subset=subset)

def inverse_mobius_transform(seq, subset=True):
    return _mobius_transform(seq, sgn=-1, subset=subset)

inverse_mobius_transform.__doc__ = mobius_transform.__doc__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\curve.py ===
"""Curves in 2-dimensional Euclidean space.

Contains
========
Curve

"""

from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import diff
from sympy.core.containers import Tuple
from sympy.core.symbol import _symbol
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point
from sympy.integrals import integrate
from sympy.matrices import Matrix, rot_axis3
from sympy.utilities.iterables import is_sequence

from mpmath.libmp.libmpf import prec_to_dps


class Curve(GeometrySet):
    """A curve in space.

    A curve is defined by parametric functions for the coordinates, a
    parameter and the lower and upper bounds for the parameter value.

    Parameters
    ==========

    function : list of functions
    limits : 3-tuple
        Function parameter and lower and upper bounds.

    Attributes
    ==========

    functions
    parameter
    limits

    Raises
    ======

    ValueError
        When `functions` are specified incorrectly.
        When `limits` are specified incorrectly.

    Examples
    ========

    >>> from sympy import Curve, sin, cos, interpolate
    >>> from sympy.abc import t, a
    >>> C = Curve((sin(t), cos(t)), (t, 0, 2))
    >>> C.functions
    (sin(t), cos(t))
    >>> C.limits
    (t, 0, 2)
    >>> C.parameter
    t
    >>> C = Curve((t, interpolate([1, 4, 9, 16], t)), (t, 0, 1)); C
    Curve((t, t**2), (t, 0, 1))
    >>> C.subs(t, 4)
    Point2D(4, 16)
    >>> C.arbitrary_point(a)
    Point2D(a, a**2)

    See Also
    ========

    sympy.core.function.Function
    sympy.polys.polyfuncs.interpolate

    """

    def __new__(cls, function, limits):
        if not is_sequence(function) or len(function) != 2:
            raise ValueError("Function argument should be (x(t), y(t)) "
                "but got %s" % str(function))
        if not is_sequence(limits) or len(limits) != 3:
            raise ValueError("Limit argument should be (t, tmin, tmax) "
                "but got %s" % str(limits))

        return GeometryEntity.__new__(cls, Tuple(*function), Tuple(*limits))

    def __call__(self, f):
        return self.subs(self.parameter, f)

    def _eval_subs(self, old, new):
        if old == self.parameter:
            return Point(*[f.subs(old, new) for f in self.functions])

    def _eval_evalf(self, prec=15, **options):
        f, (t, a, b) = self.args
        dps = prec_to_dps(prec)
        f = tuple([i.evalf(n=dps, **options) for i in f])
        a, b = [i.evalf(n=dps, **options) for i in (a, b)]
        return self.func(f, (t, a, b))

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't'.
            The Curve's parameter is selected with None or self.parameter
            otherwise the provided symbol is used.

        Returns
        =======

        Point :
            Returns a point in parametric form.

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        Examples
        ========

        >>> from sympy import Curve, Symbol
        >>> from sympy.abc import s
        >>> C = Curve([2*s, s**2], (s, 0, 2))
        >>> C.arbitrary_point()
        Point2D(2*t, t**2)
        >>> C.arbitrary_point(C.parameter)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(None)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(Symbol('a'))
        Point2D(2*a, a**2)

        See Also
        ========

        sympy.geometry.point.Point

        """
        if parameter is None:
            return Point(*self.functions)

        tnew = _symbol(parameter, self.parameter, real=True)
        t = self.parameter
        if (tnew.name != t.name and
                tnew.name in (f.name for f in self.free_symbols)):
            raise ValueError('Symbol %s already appears in object '
                'and cannot be used as a parameter.' % tnew.name)
        return Point(*[w.subs(t, tnew) for w in self.functions])

    @property
    def free_symbols(self):
        """Return a set of symbols other than the bound symbols used to
        parametrically define the Curve.

        Returns
        =======

        set :
            Set of all non-parameterized symbols.

        Examples
        ========

        >>> from sympy.abc import t, a
        >>> from sympy import Curve
        >>> Curve((t, t**2), (t, 0, 2)).free_symbols
        set()
        >>> Curve((t, t**2), (t, a, 2)).free_symbols
        {a}

        """
        free = set()
        for a in self.functions + self.limits[1:]:
            free |= a.free_symbols
        free = free.difference({self.parameter})
        return free

    @property
    def ambient_dimension(self):
        """The dimension of the curve.

        Returns
        =======

        int :
            the dimension of curve.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.ambient_dimension
        2

        """

        return len(self.args[0])

    @property
    def functions(self):
        """The functions specifying the curve.

        Returns
        =======

        functions :
            list of parameterized coordinate functions.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.functions
        (t, t**2)

        See Also
        ========

        parameter

        """
        return self.args[0]

    @property
    def limits(self):
        """The limits for the curve.

        Returns
        =======

        limits : tuple
            Contains parameter and lower and upper limits.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**3], (t, -2, 2))
        >>> C.limits
        (t, -2, 2)

        See Also
        ========

        plot_interval

        """
        return self.args[1]

    @property
    def parameter(self):
        """The curve function variable.

        Returns
        =======

        Symbol :
            returns a bound symbol.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**2], (t, 0, 2))
        >>> C.parameter
        t

        See Also
        ========

        functions

        """
        return self.args[1][0]

    @property
    def length(self):
        """The curve length.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import t
        >>> Curve((t, t), (t, 0, 1)).length
        sqrt(2)

        """
        integrand = sqrt(sum(diff(func, self.limits[0])**2 for func in self.functions))
        return integrate(integrand, self.limits)

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't';
            otherwise the provided symbol is used.

        Returns
        =======

        List :
            the plot interval as below:
                [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Curve, sin
        >>> from sympy.abc import x, s
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval()
        [t, 1, 2]
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval(s)
        [s, 1, 2]

        See Also
        ========

        limits : Returns limits of the parameter interval

        """
        t = _symbol(parameter, self.parameter, real=True)
        return [t] + list(self.limits[1:])

    def rotate(self, angle=0, pt=None):
        """This function is used to rotate a curve along given point ``pt`` at given angle(in radian).

        Parameters
        ==========

        angle :
            the angle at which the curve will be rotated(in radian) in counterclockwise direction.
            default value of angle is 0.

        pt : Point
            the point along which the curve will be rotated.
            If no point given, the curve will be rotated around origin.

        Returns
        =======

        Curve :
            returns a curve rotated at given angle along given point.

        Examples
        ========

        >>> from sympy import Curve, pi
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).rotate(pi/2)
        Curve((-x, x), (x, 0, 1))

        """
        if pt:
            pt = -Point(pt, dim=2)
        else:
            pt = Point(0,0)
        rv = self.translate(*pt.args)
        f = list(rv.functions)
        f.append(0)
        f = Matrix(1, 3, f)
        f *= rot_axis3(angle)
        rv = self.func(f[0, :2].tolist()[0], self.limits)
        pt = -pt
        return rv.translate(*pt.args)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since Curve is not made up of Points.

        Returns
        =======

        Curve :
            returns scaled curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).scale(2)
        Curve((2*x, x), (x, 0, 1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        fx, fy = self.functions
        return self.func((fx*x, fy*y), self.limits)

    def translate(self, x=0, y=0):
        """Translate the Curve by (x, y).

        Returns
        =======

        Curve :
            returns a translated curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).translate(1, 2)
        Curve((x + 1, x + 2), (x, 0, 1))

        """
        fx, fy = self.functions
        return self.func((fx + x, fy + y), self.limits)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\terminal\html.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: February 29, 2020
# URL: https://humanfriendly.readthedocs.io

"""Convert HTML with simple text formatting to text with ANSI escape sequences."""

# Standard library modules.
import re

# Modules included in our package.
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style

# Public identifiers that require documentation.
__all__ = ('HTMLConverter', 'html_to_ansi')


def html_to_ansi(data, callback=None):
    """
    Convert HTML with simple text formatting to text with ANSI escape sequences.

    :param data: The HTML to convert (a string).
    :param callback: Optional callback to pass to :class:`HTMLConverter`.
    :returns: Text with ANSI escape sequences (a string).

    Please refer to the documentation of the :class:`HTMLConverter` class for
    details about the conversion process (like which tags are supported) and an
    example with a screenshot.
    """
    converter = HTMLConverter(callback=callback)
    return converter(data)


class HTMLConverter(HTMLParser):

    """
    Convert HTML with simple text formatting to text with ANSI escape sequences.

    The following text styles are supported:

    - Bold: ``<b>``, ``<strong>`` and ``<span style="font-weight: bold;">``
    - Italic: ``<i>``, ``<em>`` and ``<span style="font-style: italic;">``
    - Strike-through: ``<del>``, ``<s>`` and ``<span style="text-decoration: line-through;">``
    - Underline: ``<ins>``, ``<u>`` and ``<span style="text-decoration: underline">``

    Colors can be specified as follows:

    - Foreground color: ``<span style="color: #RRGGBB;">``
    - Background color: ``<span style="background-color: #RRGGBB;">``

    Here's a small demonstration:

    .. code-block:: python

       from humanfriendly.text import dedent
       from humanfriendly.terminal import html_to_ansi

       print(html_to_ansi(dedent('''
         <b>Hello world!</b>
         <i>Is this thing on?</i>
         I guess I can <u>underline</u> or <s>strike-through</s> text?
         And what about <span style="color: red">color</span>?
       ''')))

       rainbow_colors = [
           '#FF0000', '#E2571E', '#FF7F00', '#FFFF00', '#00FF00',
           '#96BF33', '#0000FF', '#4B0082', '#8B00FF', '#FFFFFF',
       ]
       html_rainbow = "".join('<span style="color: %s">o</span>' % c for c in rainbow_colors)
       print(html_to_ansi("Let's try a rainbow: %s" % html_rainbow))

    Here's what the results look like:

      .. image:: images/html-to-ansi.png

    Some more details:

    - Nested tags are supported, within reasonable limits.

    - Text in ``<code>`` and ``<pre>`` tags will be highlighted in a
      different color from the main text (currently this is yellow).

    - ``<a href="URL">TEXT</a>`` is converted to the format "TEXT (URL)" where
      the uppercase symbols are highlighted in light blue with an underline.

    - ``<div>``, ``<p>`` and ``<pre>`` tags are considered block level tags
      and are wrapped in vertical whitespace to prevent their content from
      "running into" surrounding text. This may cause runs of multiple empty
      lines to be emitted. As a *workaround* the :func:`__call__()` method
      will automatically call :func:`.compact_empty_lines()` on the generated
      output before returning it to the caller. Of course this won't work
      when `output` is set to something like :data:`sys.stdout`.

    - ``<br>`` is converted to a single plain text line break.

    Implementation notes:

    - A list of dictionaries with style information is used as a stack where
      new styling can be pushed and a pop will restore the previous styling.
      When new styling is pushed, it is merged with (but overrides) the current
      styling.

    - If you're going to be converting a lot of HTML it might be useful from
      a performance standpoint to re-use an existing :class:`HTMLConverter`
      object for unrelated HTML fragments, in this case take a look at the
      :func:`__call__()` method (it makes this use case very easy).

    .. versionadded:: 4.15
       :class:`humanfriendly.terminal.HTMLConverter` was added to the
       `humanfriendly` package during the initial development of my new
       `chat-archive <https://chat-archive.readthedocs.io/>`_ project, whose
       command line interface makes for a great demonstration of the
       flexibility that this feature provides (hint: check out how the search
       keyword highlighting combines with the regular highlighting).
    """

    BLOCK_TAGS = ('div', 'p', 'pre')
    """The names of tags that are padded with vertical whitespace."""

    def __init__(self, *args, **kw):
        """
        Initialize an :class:`HTMLConverter` object.

        :param callback: Optional keyword argument to specify a function that
                         will be called to process text fragments before they
                         are emitted on the output stream. Note that link text
                         and preformatted text fragments are not processed by
                         this callback.
        :param output: Optional keyword argument to redirect the output to the
                       given file-like object. If this is not given a new
                       :class:`~python3:io.StringIO` object is created.
        """
        # Hide our optional keyword arguments from the superclass.
        self.callback = kw.pop("callback", None)
        self.output = kw.pop("output", None)
        # Initialize the superclass.
        HTMLParser.__init__(self, *args, **kw)

    def __call__(self, data):
        """
        Reset the parser, convert some HTML and get the text with ANSI escape sequences.

        :param data: The HTML to convert to text (a string).
        :returns: The converted text (only in case `output` is
                  a :class:`~python3:io.StringIO` object).
        """
        self.reset()
        self.feed(data)
        self.close()
        if isinstance(self.output, StringIO):
            return compact_empty_lines(self.output.getvalue())

    @property
    def current_style(self):
        """Get the current style from the top of the stack (a dictionary)."""
        return self.stack[-1] if self.stack else {}

    def close(self):
        """
        Close previously opened ANSI escape sequences.

        This method overrides the same method in the superclass to ensure that
        an :data:`.ANSI_RESET` code is emitted when parsing reaches the end of
        the input but a style is still active. This is intended to prevent
        malformed HTML from messing up terminal output.
        """
        if any(self.stack):
            self.output.write(ANSI_RESET)
            self.stack = []
        HTMLParser.close(self)

    def emit_style(self, style=None):
        """
        Emit an ANSI escape sequence for the given or current style to the output stream.

        :param style: A dictionary with arguments for :func:`.ansi_style()` or
                      :data:`None`, in which case the style at the top of the
                      stack is emitted.
        """
        # Clear the current text styles.
        self.output.write(ANSI_RESET)
        # Apply a new text style?
        style = self.current_style if style is None else style
        if style:
            self.output.write(ansi_style(**style))

    def handle_charref(self, value):
        """
        Process a decimal or hexadecimal numeric character reference.

        :param value: The decimal or hexadecimal value (a string).
        """
        self.output.write(unichr(int(value[1:], 16) if value.startswith('x') else int(value)))

    def handle_data(self, data):
        """
        Process textual data.

        :param data: The decoded text (a string).
        """
        if self.link_url:
            # Link text is captured literally so that we can reliably check
            # whether the text and the URL of the link are the same string.
            self.link_text = data
        elif self.callback and self.preformatted_text_level == 0:
            # Text that is not part of a link and not preformatted text is
            # passed to the user defined callback to allow for arbitrary
            # pre-processing.
            data = self.callback(data)
        # All text is emitted unmodified on the output stream.
        self.output.write(data)

    def handle_endtag(self, tag):
        """
        Process the end of an HTML tag.

        :param tag: The name of the tag (a string).
        """
        if tag in ('a', 'b', 'code', 'del', 'em', 'i', 'ins', 'pre', 's', 'strong', 'span', 'u'):
            old_style = self.current_style
            # The following conditional isn't necessary for well formed
            # HTML but prevents raising exceptions on malformed HTML.
            if self.stack:
                self.stack.pop(-1)
            new_style = self.current_style
            if tag == 'a':
                if self.urls_match(self.link_text, self.link_url):
                    # Don't render the URL when it's part of the link text.
                    self.emit_style(new_style)
                else:
                    self.emit_style(new_style)
                    self.output.write(' (')
                    self.emit_style(old_style)
                    self.output.write(self.render_url(self.link_url))
                    self.emit_style(new_style)
                    self.output.write(')')
            else:
                self.emit_style(new_style)
            if tag in ('code', 'pre'):
                self.preformatted_text_level -= 1
        if tag in self.BLOCK_TAGS:
            # Emit an empty line after block level tags.
            self.output.write('\n\n')

    def handle_entityref(self, name):
        """
        Process a named character reference.

        :param name: The name of the character reference (a string).
        """
        self.output.write(unichr(name2codepoint[name]))

    def handle_starttag(self, tag, attrs):
        """
        Process the start of an HTML tag.

        :param tag: The name of the tag (a string).
        :param attrs: A list of tuples with two strings each.
        """
        if tag in self.BLOCK_TAGS:
            # Emit an empty line before block level tags.
            self.output.write('\n\n')
        if tag == 'a':
            self.push_styles(color='blue', bright=True, underline=True)
            # Store the URL that the link points to for later use, so that we
            # can render the link text before the URL (with the reasoning that
            # this is the most intuitive way to present a link in a plain text
            # interface).
            self.link_url = next((v for n, v in attrs if n == 'href'), '')
        elif tag == 'b' or tag == 'strong':
            self.push_styles(bold=True)
        elif tag == 'br':
            self.output.write('\n')
        elif tag == 'code' or tag == 'pre':
            self.push_styles(color='yellow')
            self.preformatted_text_level += 1
        elif tag == 'del' or tag == 's':
            self.push_styles(strike_through=True)
        elif tag == 'em' or tag == 'i':
            self.push_styles(italic=True)
        elif tag == 'ins' or tag == 'u':
            self.push_styles(underline=True)
        elif tag == 'span':
            styles = {}
            css = next((v for n, v in attrs if n == 'style'), "")
            for rule in css.split(';'):
                name, _, value = rule.partition(':')
                name = name.strip()
                value = value.strip()
                if name == 'background-color':
                    styles['background'] = self.parse_color(value)
                elif name == 'color':
                    styles['color'] = self.parse_color(value)
                elif name == 'font-style' and value == 'italic':
                    styles['italic'] = True
                elif name == 'font-weight' and value == 'bold':
                    styles['bold'] = True
                elif name == 'text-decoration' and value == 'line-through':
                    styles['strike_through'] = True
                elif name == 'text-decoration' and value == 'underline':
                    styles['underline'] = True
            self.push_styles(**styles)

    def normalize_url(self, url):
        """
        Normalize a URL to enable string equality comparison.

        :param url: The URL to normalize (a string).
        :returns: The normalized URL (a string).
        """
        return re.sub('^mailto:', '', url)

    def parse_color(self, value):
        """
        Convert a CSS color to something that :func:`.ansi_style()` understands.

        :param value: A string like ``rgb(1,2,3)``, ``#AABBCC`` or ``yellow``.
        :returns: A color value supported by :func:`.ansi_style()` or :data:`None`.
        """
        # Parse an 'rgb(N,N,N)' expression.
        if value.startswith('rgb'):
            tokens = re.findall(r'\d+', value)
            if len(tokens) == 3:
                return tuple(map(int, tokens))
        # Parse an '#XXXXXX' expression.
        elif value.startswith('#'):
            value = value[1:]
            length = len(value)
            if length == 6:
                # Six hex digits (proper notation).
                return (
                    int(value[:2], 16),
                    int(value[2:4], 16),
                    int(value[4:6], 16),
                )
            elif length == 3:
                # Three hex digits (shorthand).
                return (
                    int(value[0], 16),
                    int(value[1], 16),
                    int(value[2], 16),
                )
        # Try to recognize a named color.
        value = value.lower()
        if value in ANSI_COLOR_CODES:
            return value

    def push_styles(self, **changes):
        """
        Push new style information onto the stack.

        :param changes: Any keyword arguments are passed on to :func:`.ansi_style()`.

        This method is a helper for :func:`handle_starttag()`
        that does the following:

        1. Make a copy of the current styles (from the top of the stack),
        2. Apply the given `changes` to the copy of the current styles,
        3. Add the new styles to the stack,
        4. Emit the appropriate ANSI escape sequence to the output stream.
        """
        prototype = self.current_style
        if prototype:
            new_style = dict(prototype)
            new_style.update(changes)
        else:
            new_style = changes
        self.stack.append(new_style)
        self.emit_style(new_style)

    def render_url(self, url):
        """
        Prepare a URL for rendering on the terminal.

        :param url: The URL to simplify (a string).
        :returns: The simplified URL (a string).

        This method pre-processes a URL before rendering on the terminal. The
        following modifications are made:

        - The ``mailto:`` prefix is stripped.
        - Spaces are converted to ``%20``.
        - A trailing parenthesis is converted to ``%29``.
        """
        url = re.sub('^mailto:', '', url)
        url = re.sub(' ', '%20', url)
        url = re.sub(r'\)$', '%29', url)
        return url

    def reset(self):
        """
        Reset the state of the HTML parser and ANSI converter.

        When `output` is a :class:`~python3:io.StringIO` object a new
        instance will be created (and the old one garbage collected).
        """
        # Reset the state of the superclass.
        HTMLParser.reset(self)
        # Reset our instance variables.
        self.link_text = None
        self.link_url = None
        self.preformatted_text_level = 0
        if self.output is None or isinstance(self.output, StringIO):
            # If the caller specified something like output=sys.stdout then it
            # doesn't make much sense to negate that choice here in reset().
            self.output = StringIO()
        self.stack = []

    def urls_match(self, a, b):
        """
        Compare two URLs for equality using :func:`normalize_url()`.

        :param a: A string containing a URL.
        :param b: A string containing a URL.
        :returns: :data:`True` if the URLs are the same, :data:`False` otherwise.

        This method is used by :func:`handle_endtag()` to omit the URL of a
        hyperlink (``<a href="...">``) when the link text is that same URL.
        """
        return self.normalize_url(a) == self.normalize_url(b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\config.py ===
# testing/config.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

from argparse import Namespace
import collections
import inspect
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from . import mock
from . import requirements as _requirements
from .util import fail
from .. import util

# default requirements; this is replaced by plugin_base when pytest
# is run
requirements = _requirements.SuiteRequirements()

db = None
db_url = None
db_opts = None
file_config = None
test_schema = None
test_schema_2 = None
any_async = False
_current = None
ident = "main"
options: Namespace = None  # type: ignore

if typing.TYPE_CHECKING:
    from .plugin.plugin_base import FixtureFunctions

    _fixture_functions: FixtureFunctions
else:

    class _NullFixtureFunctions:
        def _null_decorator(self):
            def go(fn):
                return fn

            return go

        def skip_test_exception(self, *arg, **kw):
            return Exception()

        @property
        def add_to_marker(self):
            return mock.Mock()

        def mark_base_test_class(self):
            return self._null_decorator()

        def combinations(self, *arg_sets, **kw):
            return self._null_decorator()

        def param_ident(self, *parameters):
            return self._null_decorator()

        def fixture(self, *arg, **kw):
            return self._null_decorator()

        def get_current_test_name(self):
            return None

        def async_test(self, fn):
            return fn

    # default fixture functions; these are replaced by plugin_base when
    # pytest runs
    _fixture_functions = _NullFixtureFunctions()


_FN = TypeVar("_FN", bound=Callable[..., Any])


def combinations(
    *comb: Union[Any, Tuple[Any, ...]],
    argnames: Optional[str] = None,
    id_: Optional[str] = None,
    **kw: str,
) -> Callable[[_FN], _FN]:
    r"""Deliver multiple versions of a test based on positional combinations.

    This is a facade over pytest.mark.parametrize.


    :param \*comb: argument combinations.  These are tuples that will be passed
     positionally to the decorated function.

    :param argnames: optional list of argument names.   These are the names
     of the arguments in the test function that correspond to the entries
     in each argument tuple.   pytest.mark.parametrize requires this, however
     the combinations function will derive it automatically if not present
     by using ``inspect.getfullargspec(fn).args[1:]``.  Note this assumes the
     first argument is "self" which is discarded.

    :param id\_: optional id template.  This is a string template that
     describes how the "id" for each parameter set should be defined, if any.
     The number of characters in the template should match the number of
     entries in each argument tuple.   Each character describes how the
     corresponding entry in the argument tuple should be handled, as far as
     whether or not it is included in the arguments passed to the function, as
     well as if it is included in the tokens used to create the id of the
     parameter set.

     If omitted, the argument combinations are passed to parametrize as is.  If
     passed, each argument combination is turned into a pytest.param() object,
     mapping the elements of the argument tuple to produce an id based on a
     character value in the same position within the string template using the
     following scheme:

     .. sourcecode:: text

        i - the given argument is a string that is part of the id only, don't
            pass it as an argument

        n - the given argument should be passed and it should be added to the
            id by calling the .__name__ attribute

        r - the given argument should be passed and it should be added to the
            id by calling repr()

        s - the given argument should be passed and it should be added to the
            id by calling str()

        a - (argument) the given argument should be passed and it should not
            be used to generated the id

     e.g.::

        @testing.combinations(
            (operator.eq, "eq"),
            (operator.ne, "ne"),
            (operator.gt, "gt"),
            (operator.lt, "lt"),
            id_="na",
        )
        def test_operator(self, opfunc, name):
            pass

    The above combination will call ``.__name__`` on the first member of
    each tuple and use that as the "id" to pytest.param().


    """
    return _fixture_functions.combinations(
        *comb, id_=id_, argnames=argnames, **kw
    )


def combinations_list(arg_iterable: Iterable[Tuple[Any, ...]], **kw):
    "As combination, but takes a single iterable"
    return combinations(*arg_iterable, **kw)


class Variation:
    __slots__ = ("_name", "_argname")

    def __init__(self, case, argname, case_names):
        self._name = case
        self._argname = argname
        for casename in case_names:
            setattr(self, casename, casename == case)

    if typing.TYPE_CHECKING:

        def __getattr__(self, key: str) -> bool: ...

    @property
    def name(self):
        return self._name

    def __bool__(self):
        return self._name == self._argname

    def __nonzero__(self):
        return not self.__bool__()

    def __str__(self):
        return f"{self._argname}={self._name!r}"

    def __repr__(self):
        return str(self)

    def fail(self) -> NoReturn:
        fail(f"Unknown {self}")

    @classmethod
    def idfn(cls, variation):
        return variation.name

    @classmethod
    def generate_cases(cls, argname, cases):
        case_names = [
            argname if c is True else "not_" + argname if c is False else c
            for c in cases
        ]

        typ = type(
            argname,
            (Variation,),
            {
                "__slots__": tuple(case_names),
            },
        )

        return [typ(casename, argname, case_names) for casename in case_names]


def variation(argname_or_fn, cases=None):
    """a helper around testing.combinations that provides a single namespace
    that can be used as a switch.

    e.g.::

        @testing.variation("querytyp", ["select", "subquery", "legacy_query"])
        @testing.variation("lazy", ["select", "raise", "raise_on_sql"])
        def test_thing(self, querytyp, lazy, decl_base):
            class Thing(decl_base):
                __tablename__ = "thing"

                # use name directly
                rel = relationship("Rel", lazy=lazy.name)

            # use as a switch
            if querytyp.select:
                stmt = select(Thing)
            elif querytyp.subquery:
                stmt = select(Thing).subquery()
            elif querytyp.legacy_query:
                stmt = Session.query(Thing)
            else:
                querytyp.fail()

    The variable provided is a slots object of boolean variables, as well
    as the name of the case itself under the attribute ".name"

    """

    if inspect.isfunction(argname_or_fn):
        argname = argname_or_fn.__name__
        cases = argname_or_fn(None)

        @variation_fixture(argname, cases)
        def go(self, request):
            yield request.param

        return go
    else:
        argname = argname_or_fn
    cases_plus_limitations = [
        (
            entry
            if (isinstance(entry, tuple) and len(entry) == 2)
            else (entry, None)
        )
        for entry in cases
    ]

    variations = Variation.generate_cases(
        argname, [c for c, l in cases_plus_limitations]
    )
    return combinations(
        *[
            (
                (variation._name, variation, limitation)
                if limitation is not None
                else (variation._name, variation)
            )
            for variation, (case, limitation) in zip(
                variations, cases_plus_limitations
            )
        ],
        id_="ia",
        argnames=argname,
    )


def variation_fixture(argname, cases, scope="function"):
    return fixture(
        params=Variation.generate_cases(argname, cases),
        ids=Variation.idfn,
        scope=scope,
    )


def fixture(*arg: Any, **kw: Any) -> Any:
    return _fixture_functions.fixture(*arg, **kw)


def get_current_test_name() -> str:
    return _fixture_functions.get_current_test_name()


def mark_base_test_class() -> Any:
    return _fixture_functions.mark_base_test_class()


class _AddToMarker:
    def __getattr__(self, attr: str) -> Any:
        return getattr(_fixture_functions.add_to_marker, attr)


add_to_marker = _AddToMarker()


class Config:
    def __init__(self, db, db_opts, options, file_config):
        self._set_name(db)
        self.db = db
        self.db_opts = db_opts
        self.options = options
        self.file_config = file_config
        self.test_schema = "test_schema"
        self.test_schema_2 = "test_schema_2"

        self.is_async = db.dialect.is_async and not util.asbool(
            db.url.query.get("async_fallback", False)
        )

    _stack = collections.deque()
    _configs = set()

    def _set_name(self, db):
        suffix = "_async" if db.dialect.is_async else ""
        if db.dialect.server_version_info:
            svi = ".".join(str(tok) for tok in db.dialect.server_version_info)
            self.name = "%s+%s%s_[%s]" % (db.name, db.driver, suffix, svi)
        else:
            self.name = "%s+%s%s" % (db.name, db.driver, suffix)

    @classmethod
    def register(cls, db, db_opts, options, file_config):
        """add a config as one of the global configs.

        If there are no configs set up yet, this config also
        gets set as the "_current".
        """
        global any_async

        cfg = Config(db, db_opts, options, file_config)

        # if any backends include an async driver, then ensure
        # all setup/teardown and tests are wrapped in the maybe_async()
        # decorator that will set up a greenlet context for async drivers.
        any_async = any_async or cfg.is_async

        cls._configs.add(cfg)
        return cfg

    @classmethod
    def set_as_current(cls, config, namespace):
        global db, _current, db_url, test_schema, test_schema_2, db_opts
        _current = config
        db_url = config.db.url
        db_opts = config.db_opts
        test_schema = config.test_schema
        test_schema_2 = config.test_schema_2
        namespace.db = db = config.db

    @classmethod
    def push_engine(cls, db, namespace):
        assert _current, "Can't push without a default Config set up"
        cls.push(
            Config(
                db, _current.db_opts, _current.options, _current.file_config
            ),
            namespace,
        )

    @classmethod
    def push(cls, config, namespace):
        cls._stack.append(_current)
        cls.set_as_current(config, namespace)

    @classmethod
    def pop(cls, namespace):
        if cls._stack:
            # a failed test w/ -x option can call reset() ahead of time
            _current = cls._stack[-1]
            del cls._stack[-1]
            cls.set_as_current(_current, namespace)

    @classmethod
    def reset(cls, namespace):
        if cls._stack:
            cls.set_as_current(cls._stack[0], namespace)
            cls._stack.clear()

    @classmethod
    def all_configs(cls):
        return cls._configs

    @classmethod
    def all_dbs(cls):
        for cfg in cls.all_configs():
            yield cfg.db

    def skip_test(self, msg):
        skip_test(msg)


def skip_test(msg):
    raise _fixture_functions.skip_test_exception(msg)


def async_test(fn):
    return _fixture_functions.async_test(fn)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\parabola.py ===
"""Parabolic geometrical entity.

Contains
* Parabola

"""

from sympy.core import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, symbols
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point, Point2D
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D
from sympy.geometry.ellipse import Ellipse
from sympy.functions import sign
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve


class Parabola(GeometrySet):
    """A parabolic GeometryEntity.

    A parabola is declared with a point, that is called 'focus', and
    a line, that is called 'directrix'.
    Only vertical or horizontal parabolas are currently supported.

    Parameters
    ==========

    focus : Point
        Default value is Point(0, 0)
    directrix : Line

    Attributes
    ==========

    focus
    directrix
    axis of symmetry
    focal length
    p parameter
    vertex
    eccentricity

    Raises
    ======
    ValueError
        When `focus` is not a two dimensional point.
        When `focus` is a point of directrix.
    NotImplementedError
        When `directrix` is neither horizontal nor vertical.

    Examples
    ========

    >>> from sympy import Parabola, Point, Line
    >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7,8)))
    >>> p1.focus
    Point2D(0, 0)
    >>> p1.directrix
    Line2D(Point2D(5, 8), Point2D(7, 8))

    """

    def __new__(cls, focus=None, directrix=None, **kwargs):

        if focus:
            focus = Point(focus, dim=2)
        else:
            focus = Point(0, 0)

        directrix = Line(directrix)

        if directrix.contains(focus):
            raise ValueError('The focus must not be a point of directrix')

        return GeometryEntity.__new__(cls, focus, directrix, **kwargs)

    @property
    def ambient_dimension(self):
        """Returns the ambient dimension of parabola.

        Returns
        =======

        ambient_dimension : integer

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> f1 = Point(0, 0)
        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))
        >>> p1.ambient_dimension
        2

        """
        return 2

    @property
    def axis_of_symmetry(self):
        """Return the axis of symmetry of the parabola: a line
        perpendicular to the directrix passing through the focus.

        Returns
        =======

        axis_of_symmetry : Line

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.axis_of_symmetry
        Line2D(Point2D(0, 0), Point2D(0, 1))

        """
        return self.directrix.perpendicular_line(self.focus)

    @property
    def directrix(self):
        """The directrix of the parabola.

        Returns
        =======

        directrix : Line

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> l1 = Line(Point(5, 8), Point(7, 8))
        >>> p1 = Parabola(Point(0, 0), l1)
        >>> p1.directrix
        Line2D(Point2D(5, 8), Point2D(7, 8))

        """
        return self.args[1]

    @property
    def eccentricity(self):
        """The eccentricity of the parabola.

        Returns
        =======

        eccentricity : number

        A parabola may also be characterized as a conic section with an
        eccentricity of 1. As a consequence of this, all parabolas are
        similar, meaning that while they can be different sizes,
        they are all the same shape.

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola


        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.eccentricity
        1

        Notes
        -----
        The eccentricity for every Parabola is 1 by definition.

        """
        return S.One

    def equation(self, x='x', y='y'):
        """The equation of the parabola.

        Parameters
        ==========
        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.

        Returns
        =======
        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.equation()
        -x**2 - 16*y + 64
        >>> p1.equation('f')
        -f**2 - 16*y + 64
        >>> p1.equation(y='z')
        -x**2 - 16*z + 64

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        m = self.directrix.slope
        if m is S.Infinity:
            t1 = 4 * (self.p_parameter) * (x - self.vertex.x)
            t2 = (y - self.vertex.y)**2
        elif m == 0:
            t1 = 4 * (self.p_parameter) * (y - self.vertex.y)
            t2 = (x - self.vertex.x)**2
        else:
            a, b = self.focus
            c, d = self.directrix.coefficients[:2]
            t1 = (x - a)**2 + (y - b)**2
            t2 = self.directrix.equation(x, y)**2/(c**2 + d**2)
        return t1 - t2

    @property
    def focal_length(self):
        """The focal length of the parabola.

        Returns
        =======

        focal_lenght : number or symbolic expression

        Notes
        =====

        The distance between the vertex and the focus
        (or the vertex and directrix), measured along the axis
        of symmetry, is the "focal length".

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.focal_length
        4

        """
        distance = self.directrix.distance(self.focus)
        focal_length = distance/2

        return focal_length

    @property
    def focus(self):
        """The focus of the parabola.

        Returns
        =======

        focus : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> f1 = Point(0, 0)
        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))
        >>> p1.focus
        Point2D(0, 0)

        """
        return self.args[0]

    def intersection(self, o):
        """The intersection of the parabola and another geometrical entity `o`.

        Parameters
        ==========

        o : GeometryEntity, LinearEntity

        Returns
        =======

        intersection : list of GeometryEntity objects

        Examples
        ========

        >>> from sympy import Parabola, Point, Ellipse, Line, Segment
        >>> p1 = Point(0,0)
        >>> l1 = Line(Point(1, -2), Point(-1,-2))
        >>> parabola1 = Parabola(p1, l1)
        >>> parabola1.intersection(Ellipse(Point(0, 0), 2, 5))
        [Point2D(-2, 0), Point2D(2, 0)]
        >>> parabola1.intersection(Line(Point(-7, 3), Point(12, 3)))
        [Point2D(-4, 3), Point2D(4, 3)]
        >>> parabola1.intersection(Segment((-12, -65), (14, -68)))
        []

        """
        x, y = symbols('x y', real=True)
        parabola_eq = self.equation()
        if isinstance(o, Parabola):
            if o in self:
                return [o]
            else:
                return list(ordered([Point(i) for i in solve(
                    [parabola_eq, o.equation()], [x, y], set=True)[1]]))
        elif isinstance(o, Point2D):
            if simplify(parabola_eq.subs([(x, o._args[0]), (y, o._args[1])])) == 0:
                return [o]
            else:
                return []
        elif isinstance(o, (Segment2D, Ray2D)):
            result = solve([parabola_eq,
                Line2D(o.points[0], o.points[1]).equation()],
                [x, y], set=True)[1]
            return list(ordered([Point2D(i) for i in result if i in o]))
        elif isinstance(o, (Line2D, Ellipse)):
            return list(ordered([Point2D(i) for i in solve(
                [parabola_eq, o.equation()], [x, y], set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Wrong type of argument were put')

    @property
    def p_parameter(self):
        """P is a parameter of parabola.

        Returns
        =======

        p : number or symbolic expression

        Notes
        =====

        The absolute value of p is the focal length. The sign on p tells
        which way the parabola faces. Vertical parabolas that open up
        and horizontal that open right, give a positive value for p.
        Vertical parabolas that open down and horizontal that open left,
        give a negative value for p.


        See Also
        ========

        https://www.sparknotes.com/math/precalc/conicsections/section2/

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.p_parameter
        -4

        """
        m = self.directrix.slope
        if m is S.Infinity:
            x = self.directrix.coefficients[2]
            p = sign(self.focus.args[0] + x)
        elif m == 0:
            y = self.directrix.coefficients[2]
            p = sign(self.focus.args[1] + y)
        else:
            d = self.directrix.projection(self.focus)
            p = sign(self.focus.x - d.x)
        return p * self.focal_length

    @property
    def vertex(self):
        """The vertex of the parabola.

        Returns
        =======

        vertex : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.vertex
        Point2D(0, 4)

        """
        focus = self.focus
        m = self.directrix.slope
        if m is S.Infinity:
            vertex = Point(focus.args[0] - self.p_parameter, focus.args[1])
        elif m == 0:
            vertex = Point(focus.args[0], focus.args[1] - self.p_parameter)
        else:
            vertex = self.axis_of_symmetry.intersection(self)[0]
        return vertex

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\rref.py ===
# Algorithms for computing the reduced row echelon form of a matrix.
#
# We need to choose carefully which algorithms to use depending on the domain,
# shape, and sparsity of the matrix as well as things like the bit count in the
# case of ZZ or QQ. This is important because the algorithms have different
# performance characteristics in the extremes of dense vs sparse.
#
# In all cases we use the sparse implementations but we need to choose between
# Gauss-Jordan elimination with division and fraction-free Gauss-Jordan
# elimination. For very sparse matrices over ZZ with low bit counts it is
# asymptotically faster to use Gauss-Jordan elimination with division. For
# dense matrices with high bit counts it is asymptotically faster to use
# fraction-free Gauss-Jordan.
#
# The most important thing is to get the extreme cases right because it can
# make a big difference. In between the extremes though we have to make a
# choice and here we use empirically determined thresholds based on timings
# with random sparse matrices.
#
# In the case of QQ we have to consider the denominators as well. If the
# denominators are small then it is faster to clear them and use fraction-free
# Gauss-Jordan over ZZ. If the denominators are large then it is faster to use
# Gauss-Jordan elimination with division over QQ.
#
# Timings for the various algorithms can be found at
#
#   https://github.com/sympy/sympy/issues/25410
#   https://github.com/sympy/sympy/pull/25443

from sympy.polys.domains import ZZ

from sympy.polys.matrices.sdm import SDM, sdm_irref, sdm_rref_den
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import ddm_irref, ddm_irref_den


def _dm_rref(M, *, method='auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix``.

    This function is the implementation of :meth:`DomainMatrix.rref`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the field associated with the domain
    of the Matrix.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref_den
        Alternative function for computing RREF with denominator.
    """
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=False)

    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'GJ':
        # Use Gauss-Jordan with division over the associated field.
        Mf = _to_field(M)
        M_rref, pivots = _dm_rref_GJ(Mf)

    elif method == 'FF':
        # Use fraction-free GJ over the current domain.
        M_rref_f, den, pivots = _dm_rref_den_FF(M)
        M_rref = _to_field(M_rref_f) / den

    elif method == 'CD':
        # Clear denominators and use fraction-free GJ in the associated ring.
        _, Mr = M.clear_denoms_rowwise(convert=True)
        M_rref_f, den, pivots = _dm_rref_den_FF(Mr)
        M_rref = _to_field(M_rref_f) / den

    else:
        raise ValueError(f"Unknown method for rref: {method}")

    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # Invariants:
    #   - M_rref is in the same format (sparse or dense) as the input matrix.
    #   - M_rref is in the associated field domain and any denominator was
    #     divided in (so is implicitly 1 now).

    return M_rref, pivots


def _dm_rref_den(M, *, keep_domain=True, method='auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix`` with denominator.

    This function is the implementation of :meth:`DomainMatrix.rref_den`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the same domain as the input matrix
    unless ``keep_domain=False`` in which case the result might be over an
    associated ring or field domain.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref
        Alternative function for computing RREF without denominator.
    """
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=True)

    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'FF':
        # Use fraction-free GJ over the current domain.
        M_rref, den, pivots = _dm_rref_den_FF(M)

    elif method == 'GJ':
        # Use Gauss-Jordan with division over the associated field.
        M_rref_f, pivots = _dm_rref_GJ(_to_field(M))

        # Convert back to the ring?
        if keep_domain and M_rref_f.domain != M.domain:
            _, M_rref = M_rref_f.clear_denoms(convert=True)

            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
        else:
            # Possibly an associated field
            M_rref = M_rref_f
            den = M_rref.domain.one

    elif method == 'CD':
        # Clear denominators and use fraction-free GJ in the associated ring.
        _, Mr = M.clear_denoms_rowwise(convert=True)

        M_rref_r, den, pivots = _dm_rref_den_FF(Mr)

        if keep_domain and M_rref_r.domain != M.domain:
            # Convert back to the field
            M_rref = _to_field(M_rref_r) / den
            den = M.domain.one
        else:
            # Possibly an associated ring
            M_rref = M_rref_r

            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
    else:
        raise ValueError(f"Unknown method for rref: {method}")

    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # Invariants:
    #   - M_rref is in the same format (sparse or dense) as the input matrix.
    #   - If keep_domain=True then M_rref and den are in the same domain as the
    #     input matrix
    #   - If keep_domain=False then M_rref might be in an associated ring or
    #     field domain but den is always in the same domain as M_rref.

    return M_rref, den, pivots


def _dm_to_fmt(M, fmt):
    """Convert a matrix to the given format and return the old format."""
    old_fmt = M.rep.fmt
    if old_fmt == fmt:
        pass
    elif fmt == 'dense':
        M = M.to_dense()
    elif fmt == 'sparse':
        M = M.to_sparse()
    else:
        raise ValueError(f'Unknown format: {fmt}') # pragma: no cover
    return M, old_fmt


# These are the four basic implementations that we want to choose between:


def _dm_rref_GJ(M):
    """Compute RREF using Gauss-Jordan elimination with division."""
    if M.rep.fmt == 'sparse':
        return _dm_rref_GJ_sparse(M)
    else:
        return _dm_rref_GJ_dense(M)


def _dm_rref_den_FF(M):
    """Compute RREF using fraction-free Gauss-Jordan elimination."""
    if M.rep.fmt == 'sparse':
        return _dm_rref_den_FF_sparse(M)
    else:
        return _dm_rref_den_FF_dense(M)


def _dm_rref_GJ_sparse(M):
    """Compute RREF using sparse Gauss-Jordan elimination with division."""
    M_rref_d, pivots, _ = sdm_irref(M.rep)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_sdm), pivots


def _dm_rref_GJ_dense(M):
    """Compute RREF using dense Gauss-Jordan elimination with division."""
    partial_pivot = M.domain.is_RR or M.domain.is_CC
    ddm = M.rep.to_ddm().copy()
    pivots = ddm_irref(ddm, _partial_pivot=partial_pivot)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_ddm.to_dfm_or_ddm()), pivots


def _dm_rref_den_FF_sparse(M):
    """Compute RREF using sparse fraction-free Gauss-Jordan elimination."""
    M_rref_d, den, pivots = sdm_rref_den(M.rep, M.domain)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_sdm), den, pivots


def _dm_rref_den_FF_dense(M):
    """Compute RREF using sparse fraction-free Gauss-Jordan elimination."""
    ddm = M.rep.to_ddm().copy()
    den, pivots = ddm_irref_den(ddm, M.domain)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_ddm.to_dfm_or_ddm()), den, pivots


def _dm_rref_choose_method(M, method, *, denominator=False):
    """Choose the fastest method for computing RREF for M."""

    if method != 'auto':
        if method.endswith('_dense'):
            method = method[:-len('_dense')]
            use_fmt = 'dense'
        else:
            use_fmt = 'sparse'

    else:
        # The sparse implementations are always faster
        use_fmt = 'sparse'

        K = M.domain

        if K.is_ZZ:
            method = _dm_rref_choose_method_ZZ(M, denominator=denominator)
        elif K.is_QQ:
            method = _dm_rref_choose_method_QQ(M, denominator=denominator)
        elif K.is_RR or K.is_CC:
            # TODO: Add partial pivot support to the sparse implementations.
            method = 'GJ'
            use_fmt = 'dense'
        elif K.is_EX and M.rep.fmt == 'dense' and not denominator:
            # Do not switch to the sparse implementation for EX because the
            # domain does not have proper canonicalization and the sparse
            # implementation gives equivalent but non-identical results over EX
            # from performing arithmetic in a different order. Specifically
            # test_issue_23718 ends up getting a more complicated expression
            # when using the sparse implementation. Probably the best fix for
            # this is something else but for now we stick with the dense
            # implementation for EX if the matrix is already dense.
            method = 'GJ'
            use_fmt = 'dense'
        else:
            # This is definitely suboptimal. More work is needed to determine
            # the best method for computing RREF over different domains.
            if denominator:
                method = 'FF'
            else:
                method = 'GJ'

    return method, use_fmt


def _dm_rref_choose_method_QQ(M, *, denominator=False):
    """Choose the fastest method for computing RREF over QQ."""
    # The same sorts of considerations apply here as in the case of ZZ. Here
    # though a new more significant consideration is what sort of denominators
    # we have and what to do with them so we focus on that.

    # First compute the density. This is the average number of non-zero entries
    # per row but only counting rows that have at least one non-zero entry
    # since RREF can ignore fully zero rows.
    density, _, ncols = _dm_row_density(M)

    # For sparse matrices use Gauss-Jordan elimination over QQ regardless.
    if density < min(5, ncols/2):
        return 'GJ'

    # Compare the bit-length of the lcm of the denominators to the bit length
    # of the numerators.
    #
    # The threshold here is empirical: we prefer rref over QQ if clearing
    # denominators would result in a numerator matrix having 5x the bit size of
    # the current numerators.
    numers, denoms = _dm_QQ_numers_denoms(M)
    numer_bits = max([n.bit_length() for n in numers], default=1)

    denom_lcm = ZZ.one
    for d in denoms:
        denom_lcm = ZZ.lcm(denom_lcm, d)
        if denom_lcm.bit_length() > 5*numer_bits:
            return 'GJ'

    # If we get here then the matrix is dense and the lcm of the denominators
    # is not too large compared to the numerators. For particularly small
    # denominators it is fastest just to clear them and use fraction-free
    # Gauss-Jordan over ZZ. With very small denominators this is a little
    # faster than using rref_den over QQ but there is an intermediate regime
    # where rref_den over QQ is significantly faster. The small denominator
    # case is probably very common because small fractions like 1/2 or 1/3 are
    # often seen in user inputs.

    if denom_lcm.bit_length() < 50:
        return 'CD'
    else:
        return 'FF'


def _dm_rref_choose_method_ZZ(M, *, denominator=False):
    """Choose the fastest method for computing RREF over ZZ."""
    # In the extreme of very sparse matrices and low bit counts it is faster to
    # use Gauss-Jordan elimination over QQ rather than fraction-free
    # Gauss-Jordan over ZZ. In the opposite extreme of dense matrices and high
    # bit counts it is faster to use fraction-free Gauss-Jordan over ZZ. These
    # two extreme cases need to be handled differently because they lead to
    # different asymptotic complexities. In between these two extremes we need
    # a threshold for deciding which method to use. This threshold is
    # determined empirically by timing the two methods with random matrices.

    # The disadvantage of using empirical timings is that future optimisations
    # might change the relative speeds so this can easily become out of date.
    # The main thing is to get the asymptotic complexity right for the extreme
    # cases though so the precise value of the threshold is hopefully not too
    # important.

    # Empirically determined parameter.
    PARAM = 10000

    # First compute the density. This is the average number of non-zero entries
    # per row but only counting rows that have at least one non-zero entry
    # since RREF can ignore fully zero rows.
    density, nrows_nz, ncols = _dm_row_density(M)

    # For small matrices use QQ if more than half the entries are zero.
    if nrows_nz < 10:
        if density < ncols/2:
            return 'GJ'
        else:
            return 'FF'

    # These are just shortcuts for the formula below.
    if density < 5:
        return 'GJ'
    elif density > 5 + PARAM/nrows_nz:
        return 'FF'  # pragma: no cover

    # Maximum bitsize of any entry.
    elements = _dm_elements(M)
    bits = max([e.bit_length() for e in elements], default=1)

    # Wideness parameter. This is 1 for square or tall matrices but >1 for wide
    # matrices.
    wideness = max(1, 2/3*ncols/nrows_nz)

    max_density = (5 + PARAM/(nrows_nz*bits**2)) * wideness

    if density < max_density:
        return 'GJ'
    else:
        return 'FF'


def _dm_row_density(M):
    """Density measure for sparse matrices.

    Defines the "density", ``d`` as the average number of non-zero entries per
    row except ignoring rows that are fully zero. RREF can ignore fully zero
    rows so they are excluded. By definition ``d >= 1`` except that we define
    ``d = 0`` for the zero matrix.

    Returns ``(density, nrows_nz, ncols)`` where ``nrows_nz`` counts the number
    of nonzero rows and ``ncols`` is the number of columns.
    """
    # Uses the SDM dict-of-dicts representation.
    ncols = M.shape[1]
    rows_nz = M.rep.to_sdm().values()
    if not rows_nz:
        return 0, 0, ncols
    else:
        nrows_nz = len(rows_nz)
        density = sum(map(len, rows_nz)) / nrows_nz
        return density, nrows_nz, ncols


def _dm_elements(M):
    """Return nonzero elements of a DomainMatrix."""
    elements, _ = M.to_flat_nz()
    return elements


def _dm_QQ_numers_denoms(Mq):
    """Returns the numerators and denominators of a DomainMatrix over QQ."""
    elements = _dm_elements(Mq)
    numers = [e.numerator for e in elements]
    denoms = [e.denominator for e in elements]
    return numers, denoms


def _to_field(M):
    """Convert a DomainMatrix to a field if possible."""
    K = M.domain
    if K.has_assoc_Field:
        return M.to_field()
    else:
        return M

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_util.py ===
# Little utilities we use internally
from __future__ import annotations

import collections.abc
import inspect
import signal
from abc import ABCMeta
from collections.abc import Awaitable, Callable, Sequence
from functools import update_wrapper
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NoReturn,
    TypeVar,
    final as std_final,
)

from sniffio import thread_local as sniffio_loop

import trio

# Explicit "Any" is not allowed
CallT = TypeVar("CallT", bound=Callable[..., Any])  # type: ignore[explicit-any]
T = TypeVar("T")
RetT = TypeVar("RetT")

if TYPE_CHECKING:
    import sys
    from types import AsyncGeneratorType, TracebackType

    from typing_extensions import ParamSpec, Self, TypeVarTuple, Unpack

    if sys.version_info < (3, 11):
        from exceptiongroup import BaseExceptionGroup

    ArgsT = ParamSpec("ArgsT")
    PosArgsT = TypeVarTuple("PosArgsT")


# See: #461 as to why this is needed.
# The gist is that threading.main_thread() has the capability to lie to us
# if somebody else edits the threading ident cache to replace the main
# thread; causing threading.current_thread() to return a _DummyThread,
# causing the C-c check to fail, and so on.
# Trying to use signal out of the main thread will fail, so we can then
# reliably check if this is the main thread without relying on a
# potentially modified threading.
def is_main_thread() -> bool:
    """Attempt to reliably check if we are in the main thread."""
    try:
        signal.signal(signal.SIGINT, signal.getsignal(signal.SIGINT))
        return True
    except (TypeError, ValueError):
        return False


######
# Call the function and get the coroutine object, while giving helpful
# errors for common mistakes. Returns coroutine object.
######
def coroutine_or_error(
    async_fn: Callable[[Unpack[PosArgsT]], Awaitable[RetT]],
    *args: Unpack[PosArgsT],
) -> collections.abc.Coroutine[object, NoReturn, RetT]:
    def _return_value_looks_like_wrong_library(value: object) -> bool:
        # Returned by legacy @asyncio.coroutine functions, which includes
        # a surprising proportion of asyncio builtins.
        if isinstance(value, collections.abc.Generator):
            return True
        # The protocol for detecting an asyncio Future-like object
        if getattr(value, "_asyncio_future_blocking", None) is not None:
            return True
        # This janky check catches tornado Futures and twisted Deferreds.
        # By the time we're calling this function, we already know
        # something has gone wrong, so a heuristic is pretty safe.
        return value.__class__.__name__ in ("Future", "Deferred")

    # Make sure a sync-fn-that-returns-coroutine still sees itself as being
    # in trio context
    prev_loop, sniffio_loop.name = sniffio_loop.name, "trio"

    try:
        coro = async_fn(*args)

    except TypeError:
        # Give good error for: nursery.start_soon(trio.sleep(1))
        if isinstance(async_fn, collections.abc.Coroutine):
            # explicitly close coroutine to avoid RuntimeWarning
            async_fn.close()

            raise TypeError(
                "Trio was expecting an async function, but instead it got "
                f"a coroutine object {async_fn!r}\n"
                "\n"
                "Probably you did something like:\n"
                "\n"
                f"  trio.run({async_fn.__name__}(...))            # incorrect!\n"
                f"  nursery.start_soon({async_fn.__name__}(...))  # incorrect!\n"
                "\n"
                "Instead, you want (notice the parentheses!):\n"
                "\n"
                f"  trio.run({async_fn.__name__}, ...)            # correct!\n"
                f"  nursery.start_soon({async_fn.__name__}, ...)  # correct!",
            ) from None

        # Give good error for: nursery.start_soon(future)
        if _return_value_looks_like_wrong_library(async_fn):
            raise TypeError(
                "Trio was expecting an async function, but instead it got "
                f"{async_fn!r} – are you trying to use a library written for "
                "asyncio/twisted/tornado or similar? That won't work "
                "without some sort of compatibility shim.",
            ) from None

        raise

    finally:
        sniffio_loop.name = prev_loop

    # We can't check iscoroutinefunction(async_fn), because that will fail
    # for things like functools.partial objects wrapping an async
    # function. So we have to just call it and then check whether the
    # return value is a coroutine object.
    # Note: will not be necessary on python>=3.8, see https://bugs.python.org/issue34890
    # TODO: python3.7 support is now dropped, so the above can be addressed.
    if not isinstance(coro, collections.abc.Coroutine):
        # Give good error for: nursery.start_soon(func_returning_future)
        if _return_value_looks_like_wrong_library(coro):
            raise TypeError(
                f"Trio got unexpected {coro!r} – are you trying to use a "
                "library written for asyncio/twisted/tornado or similar? "
                "That won't work without some sort of compatibility shim.",
            )

        if inspect.isasyncgen(coro):
            raise TypeError(
                "start_soon expected an async function but got an async "
                f"generator {coro!r}",
            )

        # Give good error for: nursery.start_soon(some_sync_fn)
        raise TypeError(
            "Trio expected an async function, but {!r} appears to be "
            "synchronous".format(getattr(async_fn, "__qualname__", async_fn)),
        )

    return coro


class ConflictDetector:
    """Detect when two tasks are about to perform operations that would
    conflict.

    Use as a synchronous context manager; if two tasks enter it at the same
    time then the second one raises an error. You can use it when there are
    two pieces of code that *would* collide and need a lock if they ever were
    called at the same time, but that should never happen.

    We use this in particular for things like, making sure that two different
    tasks don't call sendall simultaneously on the same stream.

    """

    def __init__(self, msg: str) -> None:
        self._msg = msg
        self._held = False

    def __enter__(self) -> None:
        if self._held:
            raise trio.BusyResourceError(self._msg)
        else:
            self._held = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._held = False


def async_wraps(  # type: ignore[explicit-any]
    cls: type[object],
    wrapped_cls: type[object],
    attr_name: str,
) -> Callable[[CallT], CallT]:
    """Similar to wraps, but for async wrappers of non-async functions."""

    def decorator(func: CallT) -> CallT:  # type: ignore[explicit-any]
        func.__name__ = attr_name
        func.__qualname__ = f"{cls.__qualname__}.{attr_name}"

        func.__doc__ = f"Like :meth:`~{wrapped_cls.__module__}.{wrapped_cls.__qualname__}.{attr_name}`, but async."

        return func

    return decorator


def fixup_module_metadata(
    module_name: str,
    namespace: collections.abc.Mapping[str, object],
) -> None:
    seen_ids: set[int] = set()

    def fix_one(qualname: str, name: str, obj: object) -> None:
        # avoid infinite recursion (relevant when using
        # typing.Generic, for example)
        if id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))

        mod = getattr(obj, "__module__", None)
        if mod is not None and mod.startswith("trio."):
            obj.__module__ = module_name
            # Modules, unlike everything else in Python, put fully-qualified
            # names into their __name__ attribute. We check for "." to avoid
            # rewriting these.
            if hasattr(obj, "__name__") and "." not in obj.__name__:
                obj.__name__ = name
                if hasattr(obj, "__qualname__"):
                    obj.__qualname__ = qualname
            if isinstance(obj, type):
                for attr_name, attr_value in obj.__dict__.items():
                    fix_one(objname + "." + attr_name, attr_name, attr_value)

    for objname, obj in namespace.items():
        if not objname.startswith("_"):  # ignore private attributes
            fix_one(objname, objname, obj)


# We need ParamSpec to type this "properly", but that requires a runtime typing_extensions import
# to use as a class base. This is only used at runtime and isn't correct for type checkers anyway,
# so don't bother.
class generic_function(Generic[RetT]):
    """Decorator that makes a function indexable, to communicate
    non-inferable generic type parameters to a static type checker.

    If you write::

        @generic_function
        def open_memory_channel(max_buffer_size: int) -> Tuple[
            SendChannel[T], ReceiveChannel[T]
        ]: ...

    it is valid at runtime to say ``open_memory_channel[bytes](5)``.
    This behaves identically to ``open_memory_channel(5)`` at runtime,
    and currently won't type-check without a mypy plugin or clever stubs,
    but at least it becomes possible to write those.
    """

    def __init__(  # type: ignore[explicit-any]
        self,
        fn: Callable[..., RetT],
    ) -> None:
        update_wrapper(self, fn)
        self._fn = fn

    def __call__(self, *args: object, **kwargs: object) -> RetT:
        return self._fn(*args, **kwargs)

    def __getitem__(self, subscript: object) -> Self:
        return self


def _init_final_cls(cls: type[object]) -> NoReturn:
    """Raises an exception when a final class is subclassed."""
    raise TypeError(f"{cls.__module__}.{cls.__qualname__} does not support subclassing")


def _final_impl(decorated: type[T]) -> type[T]:
    """Decorator that enforces a class to be final (i.e., subclass not allowed).

    If a class uses this metaclass like this::

        @final
        class SomeClass:
            pass

    The metaclass will ensure that no subclass can be created.

    Raises
    ------
    - TypeError if a subclass is created
    """
    # Override the method blindly. We're always going to raise, so it doesn't
    # matter what the original did (if anything).
    decorated.__init_subclass__ = classmethod(_init_final_cls)  # type: ignore[assignment]
    # Apply the typing decorator, in 3.11+ it adds a __final__ marker attribute.
    return std_final(decorated)


if TYPE_CHECKING:
    from typing import final
else:
    final = _final_impl


@final  # No subclassing of NoPublicConstructor itself.
class NoPublicConstructor(ABCMeta):
    """Metaclass that ensures a private constructor.

    If a class uses this metaclass like this::

        @final
        class SomeClass(metaclass=NoPublicConstructor):
            pass

    The metaclass will ensure that no instance can be initialized. This should always be
    used with @final.

    If you try to instantiate your class (SomeClass()), a TypeError will be thrown. Use
    _create() instead in the class's implementation.

    Raises
    ------
    - TypeError if an instance is created.
    """

    def __call__(cls, *args: object, **kwargs: object) -> None:
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor",
        )

    def _create(cls: type[T], *args: object, **kwargs: object) -> T:
        return super().__call__(*args, **kwargs)  # type: ignore


def name_asyncgen(agen: AsyncGeneratorType[object, NoReturn]) -> str:
    """Return the fully-qualified name of the async generator function
    that produced the async generator iterator *agen*.
    """
    if not hasattr(agen, "ag_code"):  # pragma: no cover
        return repr(agen)
    try:
        module = agen.ag_frame.f_globals["__name__"]
    except (AttributeError, KeyError):
        module = f"<{agen.ag_code.co_filename}>"
    try:
        qualname = agen.__qualname__
    except AttributeError:
        qualname = agen.ag_code.co_name
    return f"{module}.{qualname}"


# work around a pyright error
if TYPE_CHECKING:
    Fn = TypeVar("Fn", bound=Callable[..., object])  # type: ignore[explicit-any]

    def wraps(  # type: ignore[explicit-any]
        wrapped: Callable[..., object],
        assigned: Sequence[str] = ...,
        updated: Sequence[str] = ...,
    ) -> Callable[[Fn], Fn]: ...

else:
    from functools import wraps  # noqa: F401  # this is re-exported


def raise_saving_context(exc: BaseException) -> NoReturn:
    """This helper allows re-raising an exception without __context__ being set."""
    # cause does not need special handling, we simply avoid using `raise .. from ..`
    # __suppress_context__ also does not need handling, it's only set if modifying cause
    __tracebackhide__ = True
    context = exc.__context__
    try:
        raise exc
    finally:
        exc.__context__ = context
        del exc, context


class MultipleExceptionError(Exception):
    """Raised by raise_single_exception_from_group if encountering multiple
    non-cancelled exceptions."""


def raise_single_exception_from_group(
    eg: BaseExceptionGroup[BaseException],
) -> NoReturn:
    """This function takes an exception group that is assumed to have at most
    one non-cancelled exception, which it reraises as a standalone exception.

    This exception may be an exceptiongroup itself, in which case it will not be unwrapped.

    If a :exc:`KeyboardInterrupt` is encountered, a new KeyboardInterrupt is immediately
    raised with the entire group as cause.

    If the group only contains :exc:`Cancelled` it reraises the first one encountered.

    It will retain context and cause of the contained exception, and entirely discard
    the cause/context of the group(s).

    If multiple non-cancelled exceptions are encountered, it raises
    :exc:`AssertionError`.
    """
    # immediately bail out if there's any KI or SystemExit
    for e in eg.exceptions:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise type(e) from eg

    cancelled_exception: trio.Cancelled | None = None
    noncancelled_exception: BaseException | None = None

    for e in eg.exceptions:
        if isinstance(e, trio.Cancelled):
            if cancelled_exception is None:
                cancelled_exception = e
        elif noncancelled_exception is None:
            noncancelled_exception = e
        else:
            raise MultipleExceptionError(
                "Attempted to unwrap exceptiongroup with multiple non-cancelled exceptions. This is often caused by a bug in the caller."
            ) from eg

    if noncancelled_exception is not None:
        raise_saving_context(noncancelled_exception)

    assert cancelled_exception is not None, "group can't be empty"
    raise_saving_context(cancelled_exception)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\eval.py ===
"""
Top level ``eval`` module.
"""
from __future__ import annotations

import tokenize
from typing import TYPE_CHECKING
import warnings

from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.common import (
    is_extension_array_dtype,
    is_string_dtype,
)

from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
    PARSERS,
    Expr,
)
from pandas.core.computation.parsing import tokenize_string
from pandas.core.computation.scope import ensure_scope
from pandas.core.generic import NDFrame

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas.core.computation.ops import BinOp


def _check_engine(engine: str | None) -> str:
    """
    Make sure a valid engine is passed.

    Parameters
    ----------
    engine : str
        String to validate.

    Raises
    ------
    KeyError
      * If an invalid engine is passed.
    ImportError
      * If numexpr was requested but doesn't exist.

    Returns
    -------
    str
        Engine name.
    """
    from pandas.core.computation.check import NUMEXPR_INSTALLED
    from pandas.core.computation.expressions import USE_NUMEXPR

    if engine is None:
        engine = "numexpr" if USE_NUMEXPR else "python"

    if engine not in ENGINES:
        valid_engines = list(ENGINES.keys())
        raise KeyError(
            f"Invalid engine '{engine}' passed, valid engines are {valid_engines}"
        )

    # TODO: validate this in a more general way (thinking of future engines
    # that won't necessarily be import-able)
    # Could potentially be done on engine instantiation
    if engine == "numexpr" and not NUMEXPR_INSTALLED:
        raise ImportError(
            "'numexpr' is not installed or an unsupported version. Cannot use "
            "engine='numexpr' for query/eval if 'numexpr' is not installed"
        )

    return engine


def _check_parser(parser: str):
    """
    Make sure a valid parser is passed.

    Parameters
    ----------
    parser : str

    Raises
    ------
    KeyError
      * If an invalid parser is passed
    """
    if parser not in PARSERS:
        raise KeyError(
            f"Invalid parser '{parser}' passed, valid parsers are {PARSERS.keys()}"
        )


def _check_resolvers(resolvers):
    if resolvers is not None:
        for resolver in resolvers:
            if not hasattr(resolver, "__getitem__"):
                name = type(resolver).__name__
                raise TypeError(
                    f"Resolver of type '{name}' does not "
                    "implement the __getitem__ method"
                )


def _check_expression(expr):
    """
    Make sure an expression is not an empty string

    Parameters
    ----------
    expr : object
        An object that can be converted to a string

    Raises
    ------
    ValueError
      * If expr is an empty string
    """
    if not expr:
        raise ValueError("expr cannot be an empty string")


def _convert_expression(expr) -> str:
    """
    Convert an object to an expression.

    This function converts an object to an expression (a unicode string) and
    checks to make sure it isn't empty after conversion. This is used to
    convert operators to their string representation for recursive calls to
    :func:`~pandas.eval`.

    Parameters
    ----------
    expr : object
        The object to be converted to a string.

    Returns
    -------
    str
        The string representation of an object.

    Raises
    ------
    ValueError
      * If the expression is empty.
    """
    s = pprint_thing(expr)
    _check_expression(s)
    return s


def _check_for_locals(expr: str, stack_level: int, parser: str):
    at_top_of_stack = stack_level == 0
    not_pandas_parser = parser != "pandas"

    if not_pandas_parser:
        msg = "The '@' prefix is only supported by the pandas parser"
    elif at_top_of_stack:
        msg = (
            "The '@' prefix is not allowed in top-level eval calls.\n"
            "please refer to your variables by name without the '@' prefix."
        )

    if at_top_of_stack or not_pandas_parser:
        for toknum, tokval in tokenize_string(expr):
            if toknum == tokenize.OP and tokval == "@":
                raise SyntaxError(msg)


def eval(
    expr: str | BinOp,  # we leave BinOp out of the docstr bc it isn't for users
    parser: str = "pandas",
    engine: str | None = None,
    local_dict=None,
    global_dict=None,
    resolvers=(),
    level: int = 0,
    target=None,
    inplace: bool = False,
):
    """
    Evaluate a Python expression as a string using various backends.

    The following arithmetic operations are supported: ``+``, ``-``, ``*``,
    ``/``, ``**``, ``%``, ``//`` (python engine only) along with the following
    boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not).
    Additionally, the ``'pandas'`` parser allows the use of :keyword:`and`,
    :keyword:`or`, and :keyword:`not` with the same semantics as the
    corresponding bitwise operators.  :class:`~pandas.Series` and
    :class:`~pandas.DataFrame` objects are supported and behave as they would
    with plain ol' Python evaluation.

    Parameters
    ----------
    expr : str
        The expression to evaluate. This string cannot contain any Python
        `statements
        <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
        only Python `expressions
        <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
    parser : {'pandas', 'python'}, default 'pandas'
        The parser to use to construct the syntax tree from the expression. The
        default of ``'pandas'`` parses code slightly different than standard
        Python. Alternatively, you can parse an expression using the
        ``'python'`` parser to retain strict Python semantics.  See the
        :ref:`enhancing performance <enhancingperf.eval>` documentation for
        more details.
    engine : {'python', 'numexpr'}, default 'numexpr'

        The engine used to evaluate the expression. Supported engines are

        - None : tries to use ``numexpr``, falls back to ``python``
        - ``'numexpr'`` : This default engine evaluates pandas objects using
          numexpr for large speed ups in complex expressions with large frames.
        - ``'python'`` : Performs operations as if you had ``eval``'d in top
          level python. This engine is generally not that useful.

        More backends may be available in the future.
    local_dict : dict or None, optional
        A dictionary of local variables, taken from locals() by default.
    global_dict : dict or None, optional
        A dictionary of global variables, taken from globals() by default.
    resolvers : list of dict-like or None, optional
        A list of objects implementing the ``__getitem__`` special method that
        you can use to inject an additional collection of namespaces to use for
        variable lookup. For example, this is used in the
        :meth:`~DataFrame.query` method to inject the
        ``DataFrame.index`` and ``DataFrame.columns``
        variables that refer to their respective :class:`~pandas.DataFrame`
        instance attributes.
    level : int, optional
        The number of prior stack frames to traverse and add to the current
        scope. Most users will **not** need to change this parameter.
    target : object, optional, default None
        This is the target object for assignment. It is used when there is
        variable assignment in the expression. If so, then `target` must
        support item assignment with string keys, and if a copy is being
        returned, it must also support `.copy()`.
    inplace : bool, default False
        If `target` is provided, and the expression mutates `target`, whether
        to modify `target` inplace. Otherwise, return a copy of `target` with
        the mutation.

    Returns
    -------
    ndarray, numeric scalar, DataFrame, Series, or None
        The completion value of evaluating the given code or None if ``inplace=True``.

    Raises
    ------
    ValueError
        There are many instances where such an error can be raised:

        - `target=None`, but the expression is multiline.
        - The expression is multiline, but not all them have item assignment.
          An example of such an arrangement is this:

          a = b + 1
          a + 2

          Here, there are expressions on different lines, making it multiline,
          but the last line has no variable assigned to the output of `a + 2`.
        - `inplace=True`, but the expression is missing item assignment.
        - Item assignment is provided, but the `target` does not support
          string item assignment.
        - Item assignment is provided and `inplace=False`, but the `target`
          does not support the `.copy()` method

    See Also
    --------
    DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
    DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

    Notes
    -----
    The ``dtype`` of any objects involved in an arithmetic ``%`` operation are
    recursively cast to ``float64``.

    See the :ref:`enhancing performance <enhancingperf.eval>` documentation for
    more details.

    Examples
    --------
    >>> df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    >>> df
      animal  age
    0    dog   10
    1    pig   20

    We can add a new column using ``pd.eval``:

    >>> pd.eval("double_age = df.age * 2", target=df)
      animal  age  double_age
    0    dog   10          20
    1    pig   20          40
    """
    inplace = validate_bool_kwarg(inplace, "inplace")

    exprs: list[str | BinOp]
    if isinstance(expr, str):
        _check_expression(expr)
        exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
    else:
        # ops.BinOp; for internal compat, not intended to be passed by users
        exprs = [expr]
    multi_line = len(exprs) > 1

    if multi_line and target is None:
        raise ValueError(
            "multi-line expressions are only valid in the "
            "context of data, use DataFrame.eval"
        )
    engine = _check_engine(engine)
    _check_parser(parser)
    _check_resolvers(resolvers)

    ret = None
    first_expr = True
    target_modified = False

    for expr in exprs:
        expr = _convert_expression(expr)
        _check_for_locals(expr, level, parser)

        # get our (possibly passed-in) scope
        env = ensure_scope(
            level + 1,
            global_dict=global_dict,
            local_dict=local_dict,
            resolvers=resolvers,
            target=target,
        )

        parsed_expr = Expr(expr, engine=engine, parser=parser, env=env)

        if engine == "numexpr" and (
            (
                is_extension_array_dtype(parsed_expr.terms.return_type)
                and not is_string_dtype(parsed_expr.terms.return_type)
            )
            or getattr(parsed_expr.terms, "operand_types", None) is not None
            and any(
                (is_extension_array_dtype(elem) and not is_string_dtype(elem))
                for elem in parsed_expr.terms.operand_types
            )
        ):
            warnings.warn(
                "Engine has switched to 'python' because numexpr does not support "
                "extension array dtypes. Please set your engine to python manually.",
                RuntimeWarning,
                stacklevel=find_stack_level(),
            )
            engine = "python"

        # construct the engine and evaluate the parsed expression
        eng = ENGINES[engine]
        eng_inst = eng(parsed_expr)
        ret = eng_inst.evaluate()

        if parsed_expr.assigner is None:
            if multi_line:
                raise ValueError(
                    "Multi-line expressions are only valid "
                    "if all expressions contain an assignment"
                )
            if inplace:
                raise ValueError("Cannot operate inplace if there is no assignment")

        # assign if needed
        assigner = parsed_expr.assigner
        if env.target is not None and assigner is not None:
            target_modified = True

            # if returning a copy, copy only on the first assignment
            if not inplace and first_expr:
                try:
                    target = env.target
                    if isinstance(target, NDFrame):
                        target = target.copy(deep=None)
                    else:
                        target = target.copy()
                except AttributeError as err:
                    raise ValueError("Cannot return a copy of the target") from err
            else:
                target = env.target

            # TypeError is most commonly raised (e.g. int, list), but you
            # get IndexError if you try to do this assignment on np.ndarray.
            # we will ignore numpy warnings here; e.g. if trying
            # to use a non-numeric indexer
            try:
                if inplace and isinstance(target, NDFrame):
                    target.loc[:, assigner] = ret
                else:
                    target[assigner] = ret  # pyright: ignore[reportGeneralTypeIssues]
            except (TypeError, IndexError) as err:
                raise ValueError("Cannot assign expression output to target") from err

            if not resolvers:
                resolvers = ({assigner: ret},)
            else:
                # existing resolver needs updated to handle
                # case of mutating existing column in copy
                for resolver in resolvers:
                    if assigner in resolver:
                        resolver[assigner] = ret
                        break
                else:
                    resolvers += ({assigner: ret},)

            ret = None
            first_expr = False

    # We want to exclude `inplace=None` as being False.
    if inplace is False:
        return target if target_modified else ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\css.py ===
"""
Utilities for interpreting CSS from Stylers for formatting non-HTML outputs.
"""
from __future__ import annotations

import re
from typing import (
    TYPE_CHECKING,
    Callable,
)
import warnings

from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
        Iterator,
    )


def _side_expander(prop_fmt: str) -> Callable:
    """
    Wrapper to expand shorthand property into top, right, bottom, left properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """

    def expand(self, prop, value: str) -> Generator[tuple[str, str], None, None]:
        """
        Expand shorthand property into side-specific property (top, right, bottom, left)

        Parameters
        ----------
            prop (str): CSS property name
            value (str): String token for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        try:
            mapping = self.SIDE_SHORTHANDS[len(tokens)]
        except KeyError:
            warnings.warn(
                f'Could not expand "{prop}: {value}"',
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return
        for key, idx in zip(self.SIDES, mapping):
            yield prop_fmt.format(key), tokens[idx]

    return expand


def _border_expander(side: str = "") -> Callable:
    """
    Wrapper to expand 'border' property into border color, style, and width properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """
    if side != "":
        side = f"-{side}"

    def expand(self, prop, value: str) -> Generator[tuple[str, str], None, None]:
        """
        Expand border into color, style, and width tuples

        Parameters
        ----------
            prop : str
                CSS property name passed to styler
            value : str
                Value passed to styler for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        if len(tokens) == 0 or len(tokens) > 3:
            warnings.warn(
                f'Too many tokens provided to "{prop}" (expected 1-3)',
                CSSWarning,
                stacklevel=find_stack_level(),
            )

        # TODO: Can we use current color as initial value to comply with CSS standards?
        border_declarations = {
            f"border{side}-color": "black",
            f"border{side}-style": "none",
            f"border{side}-width": "medium",
        }
        for token in tokens:
            if token.lower() in self.BORDER_STYLES:
                border_declarations[f"border{side}-style"] = token
            elif any(ratio in token.lower() for ratio in self.BORDER_WIDTH_RATIOS):
                border_declarations[f"border{side}-width"] = token
            else:
                border_declarations[f"border{side}-color"] = token
            # TODO: Warn user if item entered more than once (e.g. "border: red green")

        # Per CSS, "border" will reset previous "border-*" definitions
        yield from self.atomize(border_declarations.items())

    return expand


class CSSResolver:
    """
    A callable for parsing and resolving CSS to atomic properties.
    """

    UNIT_RATIOS = {
        "pt": ("pt", 1),
        "em": ("em", 1),
        "rem": ("pt", 12),
        "ex": ("em", 0.5),
        # 'ch':
        "px": ("pt", 0.75),
        "pc": ("pt", 12),
        "in": ("pt", 72),
        "cm": ("in", 1 / 2.54),
        "mm": ("in", 1 / 25.4),
        "q": ("mm", 0.25),
        "!!default": ("em", 0),
    }

    FONT_SIZE_RATIOS = UNIT_RATIOS.copy()
    FONT_SIZE_RATIOS.update(
        {
            "%": ("em", 0.01),
            "xx-small": ("rem", 0.5),
            "x-small": ("rem", 0.625),
            "small": ("rem", 0.8),
            "medium": ("rem", 1),
            "large": ("rem", 1.125),
            "x-large": ("rem", 1.5),
            "xx-large": ("rem", 2),
            "smaller": ("em", 1 / 1.2),
            "larger": ("em", 1.2),
            "!!default": ("em", 1),
        }
    )

    MARGIN_RATIOS = UNIT_RATIOS.copy()
    MARGIN_RATIOS.update({"none": ("pt", 0)})

    BORDER_WIDTH_RATIOS = UNIT_RATIOS.copy()
    BORDER_WIDTH_RATIOS.update(
        {
            "none": ("pt", 0),
            "thick": ("px", 4),
            "medium": ("px", 2),
            "thin": ("px", 1),
            # Default: medium only if solid
        }
    )

    BORDER_STYLES = [
        "none",
        "hidden",
        "dotted",
        "dashed",
        "solid",
        "double",
        "groove",
        "ridge",
        "inset",
        "outset",
        "mediumdashdot",
        "dashdotdot",
        "hair",
        "mediumdashdotdot",
        "dashdot",
        "slantdashdot",
        "mediumdashed",
    ]

    SIDE_SHORTHANDS = {
        1: [0, 0, 0, 0],
        2: [0, 1, 0, 1],
        3: [0, 1, 2, 1],
        4: [0, 1, 2, 3],
    }

    SIDES = ("top", "right", "bottom", "left")

    CSS_EXPANSIONS = {
        **{
            (f"border-{prop}" if prop else "border"): _border_expander(prop)
            for prop in ["", "top", "right", "bottom", "left"]
        },
        **{
            f"border-{prop}": _side_expander(f"border-{{:s}}-{prop}")
            for prop in ["color", "style", "width"]
        },
        "margin": _side_expander("margin-{:s}"),
        "padding": _side_expander("padding-{:s}"),
    }

    def __call__(
        self,
        declarations: str | Iterable[tuple[str, str]],
        inherited: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        The given declarations to atomic properties.

        Parameters
        ----------
        declarations_str : str | Iterable[tuple[str, str]]
            A CSS string or set of CSS declaration tuples
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}
        inherited : dict, optional
            Atomic properties indicating the inherited style context in which
            declarations_str is to be resolved. ``inherited`` should already
            be resolved, i.e. valid output of this method.

        Returns
        -------
        dict
            Atomic CSS 2.2 properties.

        Examples
        --------
        >>> resolve = CSSResolver()
        >>> inherited = {'font-family': 'serif', 'font-weight': 'bold'}
        >>> out = resolve('''
        ...               border-color: BLUE RED;
        ...               font-size: 1em;
        ...               font-size: 2em;
        ...               font-weight: normal;
        ...               font-weight: inherit;
        ...               ''', inherited)
        >>> sorted(out.items())  # doctest: +NORMALIZE_WHITESPACE
        [('border-bottom-color', 'blue'),
         ('border-left-color', 'red'),
         ('border-right-color', 'red'),
         ('border-top-color', 'blue'),
         ('font-family', 'serif'),
         ('font-size', '24pt'),
         ('font-weight', 'bold')]
        """
        if isinstance(declarations, str):
            declarations = self.parse(declarations)
        props = dict(self.atomize(declarations))
        if inherited is None:
            inherited = {}

        props = self._update_initial(props, inherited)
        props = self._update_font_size(props, inherited)
        return self._update_other_units(props)

    def _update_initial(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        # 1. resolve inherited, initial
        for prop, val in inherited.items():
            if prop not in props:
                props[prop] = val

        new_props = props.copy()
        for prop, val in props.items():
            if val == "inherit":
                val = inherited.get(prop, "initial")

            if val in ("initial", None):
                # we do not define a complete initial stylesheet
                del new_props[prop]
            else:
                new_props[prop] = val
        return new_props

    def _update_font_size(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        # 2. resolve relative font size
        if props.get("font-size"):
            props["font-size"] = self.size_to_pt(
                props["font-size"],
                self._get_font_size(inherited),
                conversions=self.FONT_SIZE_RATIOS,
            )
        return props

    def _get_font_size(self, props: dict[str, str]) -> float | None:
        if props.get("font-size"):
            font_size_string = props["font-size"]
            return self._get_float_font_size_from_pt(font_size_string)
        return None

    def _get_float_font_size_from_pt(self, font_size_string: str) -> float:
        assert font_size_string.endswith("pt")
        return float(font_size_string.rstrip("pt"))

    def _update_other_units(self, props: dict[str, str]) -> dict[str, str]:
        font_size = self._get_font_size(props)
        # 3. TODO: resolve other font-relative units
        for side in self.SIDES:
            prop = f"border-{side}-width"
            if prop in props:
                props[prop] = self.size_to_pt(
                    props[prop],
                    em_pt=font_size,
                    conversions=self.BORDER_WIDTH_RATIOS,
                )

            for prop in [f"margin-{side}", f"padding-{side}"]:
                if prop in props:
                    # TODO: support %
                    props[prop] = self.size_to_pt(
                        props[prop],
                        em_pt=font_size,
                        conversions=self.MARGIN_RATIOS,
                    )
        return props

    def size_to_pt(self, in_val, em_pt=None, conversions=UNIT_RATIOS) -> str:
        def _error():
            warnings.warn(
                f"Unhandled size: {repr(in_val)}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return self.size_to_pt("1!!default", conversions=conversions)

        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
        if match is None:
            return _error()

        val, unit = match.groups()
        if val == "":
            # hack for 'large' etc.
            val = 1
        else:
            try:
                val = float(val)
            except ValueError:
                return _error()

        while unit != "pt":
            if unit == "em":
                if em_pt is None:
                    unit = "rem"
                else:
                    val *= em_pt
                    unit = "pt"
                continue

            try:
                unit, mul = conversions[unit]
            except KeyError:
                return _error()
            val *= mul

        val = round(val, 5)
        if int(val) == val:
            size_fmt = f"{int(val):d}pt"
        else:
            size_fmt = f"{val:f}pt"
        return size_fmt

    def atomize(self, declarations: Iterable) -> Generator[tuple[str, str], None, None]:
        for prop, value in declarations:
            prop = prop.lower()
            value = value.lower()
            if prop in self.CSS_EXPANSIONS:
                expand = self.CSS_EXPANSIONS[prop]
                yield from expand(self, prop, value)
            else:
                yield prop, value

    def parse(self, declarations_str: str) -> Iterator[tuple[str, str]]:
        """
        Generates (prop, value) pairs from declarations.

        In a future version may generate parsed tokens from tinycss/tinycss2

        Parameters
        ----------
        declarations_str : str
        """
        for decl in declarations_str.split(";"):
            if not decl.strip():
                continue
            prop, sep, val = decl.partition(":")
            prop = prop.strip().lower()
            # TODO: don't lowercase case sensitive parts of values (strings)
            val = val.strip().lower()
            if sep:
                yield prop, val
            else:
                warnings.warn(
                    f"Ill-formatted attribute: expected a colon in {repr(decl)}",
                    CSSWarning,
                    stacklevel=find_stack_level(),
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_antlr\autolevlistener.py ===
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .autolevparser import AutolevParser
else:
    from autolevparser import AutolevParser

# This class defines a complete listener for a parse tree produced by AutolevParser.
class AutolevListener(ParseTreeListener):

    # Enter a parse tree produced by AutolevParser#prog.
    def enterProg(self, ctx:AutolevParser.ProgContext):
        pass

    # Exit a parse tree produced by AutolevParser#prog.
    def exitProg(self, ctx:AutolevParser.ProgContext):
        pass


    # Enter a parse tree produced by AutolevParser#stat.
    def enterStat(self, ctx:AutolevParser.StatContext):
        pass

    # Exit a parse tree produced by AutolevParser#stat.
    def exitStat(self, ctx:AutolevParser.StatContext):
        pass


    # Enter a parse tree produced by AutolevParser#vecAssign.
    def enterVecAssign(self, ctx:AutolevParser.VecAssignContext):
        pass

    # Exit a parse tree produced by AutolevParser#vecAssign.
    def exitVecAssign(self, ctx:AutolevParser.VecAssignContext):
        pass


    # Enter a parse tree produced by AutolevParser#indexAssign.
    def enterIndexAssign(self, ctx:AutolevParser.IndexAssignContext):
        pass

    # Exit a parse tree produced by AutolevParser#indexAssign.
    def exitIndexAssign(self, ctx:AutolevParser.IndexAssignContext):
        pass


    # Enter a parse tree produced by AutolevParser#regularAssign.
    def enterRegularAssign(self, ctx:AutolevParser.RegularAssignContext):
        pass

    # Exit a parse tree produced by AutolevParser#regularAssign.
    def exitRegularAssign(self, ctx:AutolevParser.RegularAssignContext):
        pass


    # Enter a parse tree produced by AutolevParser#equals.
    def enterEquals(self, ctx:AutolevParser.EqualsContext):
        pass

    # Exit a parse tree produced by AutolevParser#equals.
    def exitEquals(self, ctx:AutolevParser.EqualsContext):
        pass


    # Enter a parse tree produced by AutolevParser#index.
    def enterIndex(self, ctx:AutolevParser.IndexContext):
        pass

    # Exit a parse tree produced by AutolevParser#index.
    def exitIndex(self, ctx:AutolevParser.IndexContext):
        pass


    # Enter a parse tree produced by AutolevParser#diff.
    def enterDiff(self, ctx:AutolevParser.DiffContext):
        pass

    # Exit a parse tree produced by AutolevParser#diff.
    def exitDiff(self, ctx:AutolevParser.DiffContext):
        pass


    # Enter a parse tree produced by AutolevParser#functionCall.
    def enterFunctionCall(self, ctx:AutolevParser.FunctionCallContext):
        pass

    # Exit a parse tree produced by AutolevParser#functionCall.
    def exitFunctionCall(self, ctx:AutolevParser.FunctionCallContext):
        pass


    # Enter a parse tree produced by AutolevParser#varDecl.
    def enterVarDecl(self, ctx:AutolevParser.VarDeclContext):
        pass

    # Exit a parse tree produced by AutolevParser#varDecl.
    def exitVarDecl(self, ctx:AutolevParser.VarDeclContext):
        pass


    # Enter a parse tree produced by AutolevParser#varType.
    def enterVarType(self, ctx:AutolevParser.VarTypeContext):
        pass

    # Exit a parse tree produced by AutolevParser#varType.
    def exitVarType(self, ctx:AutolevParser.VarTypeContext):
        pass


    # Enter a parse tree produced by AutolevParser#varDecl2.
    def enterVarDecl2(self, ctx:AutolevParser.VarDecl2Context):
        pass

    # Exit a parse tree produced by AutolevParser#varDecl2.
    def exitVarDecl2(self, ctx:AutolevParser.VarDecl2Context):
        pass


    # Enter a parse tree produced by AutolevParser#ranges.
    def enterRanges(self, ctx:AutolevParser.RangesContext):
        pass

    # Exit a parse tree produced by AutolevParser#ranges.
    def exitRanges(self, ctx:AutolevParser.RangesContext):
        pass


    # Enter a parse tree produced by AutolevParser#massDecl.
    def enterMassDecl(self, ctx:AutolevParser.MassDeclContext):
        pass

    # Exit a parse tree produced by AutolevParser#massDecl.
    def exitMassDecl(self, ctx:AutolevParser.MassDeclContext):
        pass


    # Enter a parse tree produced by AutolevParser#massDecl2.
    def enterMassDecl2(self, ctx:AutolevParser.MassDecl2Context):
        pass

    # Exit a parse tree produced by AutolevParser#massDecl2.
    def exitMassDecl2(self, ctx:AutolevParser.MassDecl2Context):
        pass


    # Enter a parse tree produced by AutolevParser#inertiaDecl.
    def enterInertiaDecl(self, ctx:AutolevParser.InertiaDeclContext):
        pass

    # Exit a parse tree produced by AutolevParser#inertiaDecl.
    def exitInertiaDecl(self, ctx:AutolevParser.InertiaDeclContext):
        pass


    # Enter a parse tree produced by AutolevParser#matrix.
    def enterMatrix(self, ctx:AutolevParser.MatrixContext):
        pass

    # Exit a parse tree produced by AutolevParser#matrix.
    def exitMatrix(self, ctx:AutolevParser.MatrixContext):
        pass


    # Enter a parse tree produced by AutolevParser#matrixInOutput.
    def enterMatrixInOutput(self, ctx:AutolevParser.MatrixInOutputContext):
        pass

    # Exit a parse tree produced by AutolevParser#matrixInOutput.
    def exitMatrixInOutput(self, ctx:AutolevParser.MatrixInOutputContext):
        pass


    # Enter a parse tree produced by AutolevParser#codeCommands.
    def enterCodeCommands(self, ctx:AutolevParser.CodeCommandsContext):
        pass

    # Exit a parse tree produced by AutolevParser#codeCommands.
    def exitCodeCommands(self, ctx:AutolevParser.CodeCommandsContext):
        pass


    # Enter a parse tree produced by AutolevParser#settings.
    def enterSettings(self, ctx:AutolevParser.SettingsContext):
        pass

    # Exit a parse tree produced by AutolevParser#settings.
    def exitSettings(self, ctx:AutolevParser.SettingsContext):
        pass


    # Enter a parse tree produced by AutolevParser#units.
    def enterUnits(self, ctx:AutolevParser.UnitsContext):
        pass

    # Exit a parse tree produced by AutolevParser#units.
    def exitUnits(self, ctx:AutolevParser.UnitsContext):
        pass


    # Enter a parse tree produced by AutolevParser#inputs.
    def enterInputs(self, ctx:AutolevParser.InputsContext):
        pass

    # Exit a parse tree produced by AutolevParser#inputs.
    def exitInputs(self, ctx:AutolevParser.InputsContext):
        pass


    # Enter a parse tree produced by AutolevParser#id_diff.
    def enterId_diff(self, ctx:AutolevParser.Id_diffContext):
        pass

    # Exit a parse tree produced by AutolevParser#id_diff.
    def exitId_diff(self, ctx:AutolevParser.Id_diffContext):
        pass


    # Enter a parse tree produced by AutolevParser#inputs2.
    def enterInputs2(self, ctx:AutolevParser.Inputs2Context):
        pass

    # Exit a parse tree produced by AutolevParser#inputs2.
    def exitInputs2(self, ctx:AutolevParser.Inputs2Context):
        pass


    # Enter a parse tree produced by AutolevParser#outputs.
    def enterOutputs(self, ctx:AutolevParser.OutputsContext):
        pass

    # Exit a parse tree produced by AutolevParser#outputs.
    def exitOutputs(self, ctx:AutolevParser.OutputsContext):
        pass


    # Enter a parse tree produced by AutolevParser#outputs2.
    def enterOutputs2(self, ctx:AutolevParser.Outputs2Context):
        pass

    # Exit a parse tree produced by AutolevParser#outputs2.
    def exitOutputs2(self, ctx:AutolevParser.Outputs2Context):
        pass


    # Enter a parse tree produced by AutolevParser#codegen.
    def enterCodegen(self, ctx:AutolevParser.CodegenContext):
        pass

    # Exit a parse tree produced by AutolevParser#codegen.
    def exitCodegen(self, ctx:AutolevParser.CodegenContext):
        pass


    # Enter a parse tree produced by AutolevParser#commands.
    def enterCommands(self, ctx:AutolevParser.CommandsContext):
        pass

    # Exit a parse tree produced by AutolevParser#commands.
    def exitCommands(self, ctx:AutolevParser.CommandsContext):
        pass


    # Enter a parse tree produced by AutolevParser#vec.
    def enterVec(self, ctx:AutolevParser.VecContext):
        pass

    # Exit a parse tree produced by AutolevParser#vec.
    def exitVec(self, ctx:AutolevParser.VecContext):
        pass


    # Enter a parse tree produced by AutolevParser#parens.
    def enterParens(self, ctx:AutolevParser.ParensContext):
        pass

    # Exit a parse tree produced by AutolevParser#parens.
    def exitParens(self, ctx:AutolevParser.ParensContext):
        pass


    # Enter a parse tree produced by AutolevParser#VectorOrDyadic.
    def enterVectorOrDyadic(self, ctx:AutolevParser.VectorOrDyadicContext):
        pass

    # Exit a parse tree produced by AutolevParser#VectorOrDyadic.
    def exitVectorOrDyadic(self, ctx:AutolevParser.VectorOrDyadicContext):
        pass


    # Enter a parse tree produced by AutolevParser#Exponent.
    def enterExponent(self, ctx:AutolevParser.ExponentContext):
        pass

    # Exit a parse tree produced by AutolevParser#Exponent.
    def exitExponent(self, ctx:AutolevParser.ExponentContext):
        pass


    # Enter a parse tree produced by AutolevParser#MulDiv.
    def enterMulDiv(self, ctx:AutolevParser.MulDivContext):
        pass

    # Exit a parse tree produced by AutolevParser#MulDiv.
    def exitMulDiv(self, ctx:AutolevParser.MulDivContext):
        pass


    # Enter a parse tree produced by AutolevParser#AddSub.
    def enterAddSub(self, ctx:AutolevParser.AddSubContext):
        pass

    # Exit a parse tree produced by AutolevParser#AddSub.
    def exitAddSub(self, ctx:AutolevParser.AddSubContext):
        pass


    # Enter a parse tree produced by AutolevParser#float.
    def enterFloat(self, ctx:AutolevParser.FloatContext):
        pass

    # Exit a parse tree produced by AutolevParser#float.
    def exitFloat(self, ctx:AutolevParser.FloatContext):
        pass


    # Enter a parse tree produced by AutolevParser#int.
    def enterInt(self, ctx:AutolevParser.IntContext):
        pass

    # Exit a parse tree produced by AutolevParser#int.
    def exitInt(self, ctx:AutolevParser.IntContext):
        pass


    # Enter a parse tree produced by AutolevParser#idEqualsExpr.
    def enterIdEqualsExpr(self, ctx:AutolevParser.IdEqualsExprContext):
        pass

    # Exit a parse tree produced by AutolevParser#idEqualsExpr.
    def exitIdEqualsExpr(self, ctx:AutolevParser.IdEqualsExprContext):
        pass


    # Enter a parse tree produced by AutolevParser#negativeOne.
    def enterNegativeOne(self, ctx:AutolevParser.NegativeOneContext):
        pass

    # Exit a parse tree produced by AutolevParser#negativeOne.
    def exitNegativeOne(self, ctx:AutolevParser.NegativeOneContext):
        pass


    # Enter a parse tree produced by AutolevParser#function.
    def enterFunction(self, ctx:AutolevParser.FunctionContext):
        pass

    # Exit a parse tree produced by AutolevParser#function.
    def exitFunction(self, ctx:AutolevParser.FunctionContext):
        pass


    # Enter a parse tree produced by AutolevParser#rangess.
    def enterRangess(self, ctx:AutolevParser.RangessContext):
        pass

    # Exit a parse tree produced by AutolevParser#rangess.
    def exitRangess(self, ctx:AutolevParser.RangessContext):
        pass


    # Enter a parse tree produced by AutolevParser#colon.
    def enterColon(self, ctx:AutolevParser.ColonContext):
        pass

    # Exit a parse tree produced by AutolevParser#colon.
    def exitColon(self, ctx:AutolevParser.ColonContext):
        pass


    # Enter a parse tree produced by AutolevParser#id.
    def enterId(self, ctx:AutolevParser.IdContext):
        pass

    # Exit a parse tree produced by AutolevParser#id.
    def exitId(self, ctx:AutolevParser.IdContext):
        pass


    # Enter a parse tree produced by AutolevParser#exp.
    def enterExp(self, ctx:AutolevParser.ExpContext):
        pass

    # Exit a parse tree produced by AutolevParser#exp.
    def exitExp(self, ctx:AutolevParser.ExpContext):
        pass


    # Enter a parse tree produced by AutolevParser#matrices.
    def enterMatrices(self, ctx:AutolevParser.MatricesContext):
        pass

    # Exit a parse tree produced by AutolevParser#matrices.
    def exitMatrices(self, ctx:AutolevParser.MatricesContext):
        pass


    # Enter a parse tree produced by AutolevParser#Indexing.
    def enterIndexing(self, ctx:AutolevParser.IndexingContext):
        pass

    # Exit a parse tree produced by AutolevParser#Indexing.
    def exitIndexing(self, ctx:AutolevParser.IndexingContext):
        pass



del AutolevParser

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_qordered_attention.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

import numpy as np
from fusion_attention import AttentionMask
from fusion_base import Fusion
from fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionQOrderedAttention(Fusion):
    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask

        super().__init__(model, "QOrderedAttention", "QOrderedLayerNormalization")

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.
        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            logger.debug(f"{reshape_q.input[1]} is not initializer.")

            # Check if the second input to Reshape flows through a Constant node
            # TODO: Investigate why FusionAttention doesn't have such logic
            constant_node = self.model.match_parent_path(reshape_q, ["Constant"], [1])

            if constant_node is None:
                return self.num_heads, self.hidden_size  # Fall back to user specified value
            else:
                constant_node = constant_node[0]

                if len(constant_node.attribute) != 1:
                    return self.num_heads, self.hidden_size  # Fall back to user specified value

                # This is assuming it is a Tensor attribute (this is a safe assumption)
                q_shape = constant_node.attribute[0].t

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size].")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        add_before_layernorm = self.model.match_parent_path(
            normalize_node,
            ["QuantizeLinear", "Add"],
            [0, 0],
        )

        if add_before_layernorm is not None:
            start_node = add_before_layernorm[-1]
        else:
            return

        # Input QDQ nodes
        dequantize_input = self.model.match_parent_path(
            start_node,
            ["DequantizeLinear"],
            [None],
        )

        if dequantize_input is None:
            logger.debug("fuse_qordered_attention: failed to match input qdq nodes path")
            return

        dequantize_input = dequantize_input[-1]

        # QKV nodes
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", "DequantizeLinear", "QuantizeLinear", "MatMul"],
            [None, None, 0, 0, 0, 0, 0],
        )

        if qkv_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match qkv path")
            return

        (_, projection_matmul, reshape_qkv, transpose_qkv, dequantize_qkv, quantize_qkv, matmul_qkv) = qkv_nodes

        # Make sure the Q/DQ has the proper zero points and constant per-tensor scales
        if not FusionUtils.check_qdq_node_for_fusion(quantize_qkv, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(dequantize_qkv, self.model):
            return

        # Identify the root input to the Attention node
        other_inputs = []
        for _i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue

            other_inputs.append(input)

        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]

        # V nodes
        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "DequantizeLinear", "QuantizeLinear", "Add", "MatMul"],
            [1, 0, 0, 0, 0, None],
        )

        if v_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match v path")
            return

        (_, _, dequantize_v, quantize_v, add_v, matmul_v) = v_nodes

        # Make sure the Q/DQ has the proper zero points and constant per-tensor scales
        if not FusionUtils.check_qdq_node_for_fusion(quantize_v, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(dequantize_v, self.model):
            return

        # V MatMul weight
        dequantize_v_matmul_weight = self.model.match_parent_path(matmul_v, ["DequantizeLinear"], [1])

        if dequantize_v_matmul_weight is None:
            logger.debug("fuse_qordered_attention: failed to match v path")
            return

        dequantize_v_matmul_weight = dequantize_v_matmul_weight[0]

        if self.model.get_constant_value(dequantize_v_matmul_weight.input[0]) is None:
            return

        # Make sure the upstream DequantizeLinear-1 has the proper zero points and scales
        # Per-channel scales are supported for weights alone
        if not FusionUtils.check_qdq_node_for_fusion(dequantize_v_matmul_weight, self.model, False):
            return

        # QK nodes
        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            [
                "DequantizeLinear",
                "QuantizeLinear",
                "Softmax",
                "Add",
                "Div",
                "DequantizeLinear",
                "QuantizeLinear",
                "MatMul",
            ],
            [0, 0, 0, 0, None, 0, 0, 0],
        )

        if qk_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match qk path")
            return

        (
            dequantize_qk_softmax,
            quantize_qk_softmax,
            softmax_qk,
            add_qk,
            div_qk,
            dequantize_qk,
            quantize_qk,
            matmul_qk,
        ) = qk_nodes

        # Make sure the Q/DQ has the proper zero points and constant per-tensor scales
        if not FusionUtils.check_qdq_node_for_fusion(quantize_qk_softmax, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(dequantize_qk_softmax, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(quantize_qk, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(dequantize_qk, self.model):
            return

        # Q nodes
        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "DequantizeLinear", "QuantizeLinear", "Add", "MatMul"],
            [0, 0, 0, 0, 0, None],
        )

        if q_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match q path")
            return

        (_, reshape_q, dequantize_q, quantize_q, add_q, matmul_q) = q_nodes

        # Make sure the Q/DQ has the proper zero points and constant per-tensor scales
        if not FusionUtils.check_qdq_node_for_fusion(quantize_q, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(dequantize_q, self.model):
            return

        # Q MatMul weight
        dequantize_q_matmul_weight = self.model.match_parent_path(matmul_q, ["DequantizeLinear"], [1])

        if dequantize_q_matmul_weight is None:
            logger.debug("fuse_qordered_attention: failed to match q path")
            return

        dequantize_q_matmul_weight = dequantize_q_matmul_weight[0]

        if self.model.get_constant_value(dequantize_q_matmul_weight.input[0]) is None:
            return

        # Make sure the upstream DequantizeLinear-1 has the proper zero points and scales
        # Per-channel scales are supported for weights alone
        if not FusionUtils.check_qdq_node_for_fusion(dequantize_q_matmul_weight, self.model, False):
            return

        # K nodes
        k_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "DequantizeLinear", "QuantizeLinear", "Add", "MatMul"],
            [1, 0, 0, 0, 0, None],
        )

        if k_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match k path")
            return

        (_, _, dequantize_k, quantize_k, add_k, matmul_k) = k_nodes

        # Make sure the Q/DQ has the proper zero points and constant per-tensor scales
        if not FusionUtils.check_qdq_node_for_fusion(quantize_k, self.model):
            return

        if not FusionUtils.check_qdq_node_for_fusion(dequantize_k, self.model):
            return

        # K MatMul weight
        dequantize_k_matmul_weight = self.model.match_parent_path(matmul_k, ["DequantizeLinear"], [1])

        if dequantize_k_matmul_weight is None:
            logger.debug("fuse_qordered_attention: failed to match k path")
            return

        dequantize_k_matmul_weight = dequantize_k_matmul_weight[0]

        if self.model.get_constant_value(dequantize_k_matmul_weight.input[0]) is None:
            return

        # Make sure the upstream DequantizeLinear-1 has the proper zero points and scales
        # Per-channel scales are supported for weights alone
        if not FusionUtils.check_qdq_node_for_fusion(dequantize_k_matmul_weight, self.model, False):
            return

        # Mask nodes
        mask_nodes = self.model.match_parent_path(
            add_qk, ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0, 0]
        )

        if mask_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match mask_nodes path")
            return

        # Ascertain `qkv_hidden_sizes` attribute value
        q_weight = self.model.get_initializer(dequantize_q_matmul_weight.input[0])
        k_weight = self.model.get_initializer(dequantize_k_matmul_weight.input[0])
        v_weight = self.model.get_initializer(dequantize_v_matmul_weight.input[0])

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(kw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])

        # Form QOrderedAttention node
        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_k.input[0] == root_input:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

            # Ascertain `num_heads` and `hidden_size`
            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

            # Formulate the inputs
            # Actual quantized input
            attention_inputs = [dequantize_input.input[0]]
            attention_inputs.append(dequantize_input.input[1])

            attention_inputs.append(dequantize_q.input[1])
            attention_inputs.append(dequantize_k.input[1])
            attention_inputs.append(dequantize_v.input[1])

            attention_inputs.append(dequantize_q_matmul_weight.input[0])
            attention_inputs.append(dequantize_k_matmul_weight.input[0])
            attention_inputs.append(dequantize_v_matmul_weight.input[0])

            attention_inputs.append(dequantize_q_matmul_weight.input[1])
            attention_inputs.append(dequantize_k_matmul_weight.input[1])
            attention_inputs.append(dequantize_v_matmul_weight.input[1])

            if self.model.get_initializer(add_q.input[0]):
                attention_inputs.append(add_q.input[0])
            else:  # second input is the constant bias
                attention_inputs.append(add_q.input[1])

            if self.model.get_initializer(add_k.input[0]):
                attention_inputs.append(add_k.input[0])
            else:  # second input is the constant bias
                attention_inputs.append(add_k.input[1])

            if self.model.get_initializer(add_v.input[0]):
                attention_inputs.append(add_v.input[0])
            else:  # second input is the constant bias
                attention_inputs.append(add_v.input[1])

            attention_inputs.append(quantize_qk.input[1])
            attention_inputs.append(quantize_qk_softmax.input[1])
            attention_inputs.append(dequantize_qkv.input[1])

            # Mask input
            if mask_index is not None:
                attention_inputs.append(mask_index)
            else:
                attention_inputs.append("")

            # The MatMul weight 'B' and 'bias' need some post-processing
            # Transpose weight 'B' from order ROW to order COL
            # This offline transpose is needed only while using the CUDA EP
            # TODO: Make this fusion logic EP-agnostic ?
            q_weight_tensor = self.model.get_initializer(dequantize_q_matmul_weight.input[0])
            FusionUtils.transpose_2d_int8_tensor(q_weight_tensor)

            k_weight_tensor = self.model.get_initializer(dequantize_k_matmul_weight.input[0])
            FusionUtils.transpose_2d_int8_tensor(k_weight_tensor)

            v_weight_tensor = self.model.get_initializer(dequantize_v_matmul_weight.input[0])
            FusionUtils.transpose_2d_int8_tensor(v_weight_tensor)

            # Name and create Attention node
            attention_node_name = self.model.create_node_name("QOrderedAttention")

            attention_node = helper.make_node(
                "QOrderedAttention",
                inputs=attention_inputs,
                outputs=[reshape_qkv.output[0]],
                name=attention_node_name,
            )

            self.model.replace_node_input(dequantize_qkv, dequantize_qkv.input[0], attention_node.output[0])
            self.model.replace_node_input(projection_matmul, projection_matmul.input[0], dequantize_qkv.output[0])

            attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
            attention_node.attribute.extend([helper.make_attribute("order_input", 1)])
            attention_node.attribute.extend([helper.make_attribute("order_weight", 0)])
            attention_node.attribute.extend([helper.make_attribute("order_output", 1)])
            attention_node.attribute.extend(
                [helper.make_attribute("qkv_hidden_sizes", [qw_out_size, kw_out_size, vw_out_size])]
            )

            attention_node.domain = "com.microsoft"

            self.nodes_to_add.append(attention_node)
            self.node_name_to_graph_name[attention_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([reshape_qkv, transpose_qkv, quantize_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)
            self.nodes_to_remove.extend(
                [dequantize_q_matmul_weight, dequantize_k_matmul_weight, dequantize_v_matmul_weight]
            )

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            # self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\tz\_common.py ===
from six import PY2

from functools import wraps

from datetime import datetime, timedelta, tzinfo


ZERO = timedelta(0)

__all__ = ['tzname_in_python2', 'enfold']


def tzname_in_python2(namefunc):
    """Change unicode output into bytestrings in Python 2

    tzname() API changed in Python 3. It used to return bytes, but was changed
    to unicode strings
    """
    if PY2:
        @wraps(namefunc)
        def adjust_encoding(*args, **kwargs):
            name = namefunc(*args, **kwargs)
            if name is not None:
                name = name.encode()

            return name

        return adjust_encoding
    else:
        return namefunc


# The following is adapted from Alexander Belopolsky's tz library
# https://github.com/abalkin/tz
if hasattr(datetime, 'fold'):
    # This is the pre-python 3.6 fold situation
    def enfold(dt, fold=1):
        """
        Provides a unified interface for assigning the ``fold`` attribute to
        datetimes both before and after the implementation of PEP-495.

        :param fold:
            The value for the ``fold`` attribute in the returned datetime. This
            should be either 0 or 1.

        :return:
            Returns an object for which ``getattr(dt, 'fold', 0)`` returns
            ``fold`` for all versions of Python. In versions prior to
            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
            subclass of :py:class:`datetime.datetime` with the ``fold``
            attribute added, if ``fold`` is 1.

        .. versionadded:: 2.6.0
        """
        return dt.replace(fold=fold)

else:
    class _DatetimeWithFold(datetime):
        """
        This is a class designed to provide a PEP 495-compliant interface for
        Python versions before 3.6. It is used only for dates in a fold, so
        the ``fold`` attribute is fixed at ``1``.

        .. versionadded:: 2.6.0
        """
        __slots__ = ()

        def replace(self, *args, **kwargs):
            """
            Return a datetime with the same attributes, except for those
            attributes given new values by whichever keyword arguments are
            specified. Note that tzinfo=None can be specified to create a naive
            datetime from an aware datetime with no conversion of date and time
            data.

            This is reimplemented in ``_DatetimeWithFold`` because pypy3 will
            return a ``datetime.datetime`` even if ``fold`` is unchanged.
            """
            argnames = (
                'year', 'month', 'day', 'hour', 'minute', 'second',
                'microsecond', 'tzinfo'
            )

            for arg, argname in zip(args, argnames):
                if argname in kwargs:
                    raise TypeError('Duplicate argument: {}'.format(argname))

                kwargs[argname] = arg

            for argname in argnames:
                if argname not in kwargs:
                    kwargs[argname] = getattr(self, argname)

            dt_class = self.__class__ if kwargs.get('fold', 1) else datetime

            return dt_class(**kwargs)

        @property
        def fold(self):
            return 1

    def enfold(dt, fold=1):
        """
        Provides a unified interface for assigning the ``fold`` attribute to
        datetimes both before and after the implementation of PEP-495.

        :param fold:
            The value for the ``fold`` attribute in the returned datetime. This
            should be either 0 or 1.

        :return:
            Returns an object for which ``getattr(dt, 'fold', 0)`` returns
            ``fold`` for all versions of Python. In versions prior to
            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
            subclass of :py:class:`datetime.datetime` with the ``fold``
            attribute added, if ``fold`` is 1.

        .. versionadded:: 2.6.0
        """
        if getattr(dt, 'fold', 0) == fold:
            return dt

        args = dt.timetuple()[:6]
        args += (dt.microsecond, dt.tzinfo)

        if fold:
            return _DatetimeWithFold(*args)
        else:
            return datetime(*args)


def _validate_fromutc_inputs(f):
    """
    The CPython version of ``fromutc`` checks that the input is a ``datetime``
    object and that ``self`` is attached as its ``tzinfo``.
    """
    @wraps(f)
    def fromutc(self, dt):
        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        return f(self, dt)

    return fromutc


class _tzinfo(tzinfo):
    """
    Base class for all ``dateutil`` ``tzinfo`` objects.
    """

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """

        dt = dt.replace(tzinfo=self)

        wall_0 = enfold(dt, fold=0)
        wall_1 = enfold(dt, fold=1)

        same_offset = wall_0.utcoffset() == wall_1.utcoffset()
        same_dt = wall_0.replace(tzinfo=None) == wall_1.replace(tzinfo=None)

        return same_dt and not same_offset

    def _fold_status(self, dt_utc, dt_wall):
        """
        Determine the fold status of a "wall" datetime, given a representation
        of the same datetime as a (naive) UTC datetime. This is calculated based
        on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all
        datetimes, and that this offset is the actual number of hours separating
        ``dt_utc`` and ``dt_wall``.

        :param dt_utc:
            Representation of the datetime as UTC

        :param dt_wall:
            Representation of the datetime as "wall time". This parameter must
            either have a `fold` attribute or have a fold-naive
            :class:`datetime.tzinfo` attached, otherwise the calculation may
            fail.
        """
        if self.is_ambiguous(dt_wall):
            delta_wall = dt_wall - dt_utc
            _fold = int(delta_wall == (dt_utc.utcoffset() - dt_utc.dst()))
        else:
            _fold = 0

        return _fold

    def _fold(self, dt):
        return getattr(dt, 'fold', 0)

    def _fromutc(self, dt):
        """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """

        # Re-implement the algorithm from Python's datetime.py
        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() "
                             "result")

        # The original datetime.py code assumes that `dst()` defaults to
        # zero during ambiguous times. PEP 495 inverts this presumption, so
        # for pre-PEP 495 versions of python, we need to tweak the algorithm.
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta = dtoff - dtdst

        dt += delta
        # Set fold=1 so we can default to being in the fold for
        # ambiguous dates.
        dtdst = enfold(dt, fold=1).dst()
        if dtdst is None:
            raise ValueError("fromutc(): dt.dst gave inconsistent "
                             "results; cannot convert")
        return dt + dtdst

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """
        dt_wall = self._fromutc(dt)

        # Calculate the fold status given the two datetimes.
        _fold = self._fold_status(dt, dt_wall)

        # Set the default fold value for ambiguous dates
        return enfold(dt_wall, fold=_fold)


class tzrangebase(_tzinfo):
    """
    This is an abstract base class for time zones represented by an annual
    transition into and out of DST. Child classes should implement the following
    methods:

        * ``__init__(self, *args, **kwargs)``
        * ``transitions(self, year)`` - this is expected to return a tuple of
          datetimes representing the DST on and off transitions in standard
          time.

    A fully initialized ``tzrangebase`` subclass should also provide the
    following attributes:
        * ``hasdst``: Boolean whether or not the zone uses DST.
        * ``_dst_offset`` / ``_std_offset``: :class:`datetime.timedelta` objects
          representing the respective UTC offsets.
        * ``_dst_abbr`` / ``_std_abbr``: Strings representing the timezone short
          abbreviations in DST and STD, respectively.
        * ``_hasdst``: Whether or not the zone has DST.

    .. versionadded:: 2.6.0
    """
    def __init__(self):
        raise NotImplementedError('tzrangebase is an abstract base class')

    def utcoffset(self, dt):
        isdst = self._isdst(dt)

        if isdst is None:
            return None
        elif isdst:
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        isdst = self._isdst(dt)

        if isdst is None:
            return None
        elif isdst:
            return self._dst_base_offset
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        if self._isdst(dt):
            return self._dst_abbr
        else:
            return self._std_abbr

    def fromutc(self, dt):
        """ Given a datetime in UTC, return local time """
        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")

        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        # Get transitions - if there are none, fixed offset
        transitions = self.transitions(dt.year)
        if transitions is None:
            return dt + self.utcoffset(dt)

        # Get the transition times in UTC
        dston, dstoff = transitions

        dston -= self._std_offset
        dstoff -= self._std_offset

        utc_transitions = (dston, dstoff)
        dt_utc = dt.replace(tzinfo=None)

        isdst = self._naive_isdst(dt_utc, utc_transitions)

        if isdst:
            dt_wall = dt + self._dst_offset
        else:
            dt_wall = dt + self._std_offset

        _fold = int(not isdst and self.is_ambiguous(dt_wall))

        return enfold(dt_wall, fold=_fold)

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        if not self.hasdst:
            return False

        start, end = self.transitions(dt.year)

        dt = dt.replace(tzinfo=None)
        return (end <= dt < end + self._dst_base_offset)

    def _isdst(self, dt):
        if not self.hasdst:
            return False
        elif dt is None:
            return None

        transitions = self.transitions(dt.year)

        if transitions is None:
            return False

        dt = dt.replace(tzinfo=None)

        isdst = self._naive_isdst(dt, transitions)

        # Handle ambiguous dates
        if not isdst and self.is_ambiguous(dt):
            return not self._fold(dt)
        else:
            return isdst

    def _naive_isdst(self, dt, transitions):
        dston, dstoff = transitions

        dt = dt.replace(tzinfo=None)

        if dston < dstoff:
            isdst = dston <= dt < dstoff
        else:
            isdst = not dstoff <= dt < dston

        return isdst

    @property
    def _dst_base_offset(self):
        return self._dst_offset - self._std_offset

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s(...)" % self.__class__.__name__

    __reduce__ = object.__reduce__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\backends\base_backend.py ===
from sympy.plotting.series import BaseSeries, GenericDataSeries
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence


__doctest_requires__ = {
    ('Plot.append', 'Plot.extend'): ['matplotlib'],
}


# Global variable
# Set to False when running tests / doctests so that the plots don't show.
_show = True

def unset_show():
    """
    Disable show(). For use in the tests.
    """
    global _show
    _show = False


def _deprecation_msg_m_a_r_f(attr):
    sympy_deprecation_warning(
        f"The `{attr}` property is deprecated. The `{attr}` keyword "
        "argument should be passed to a plotting function, which generates "
        "the appropriate data series. If needed, index the plot object to "
        "retrieve a specific data series.",
        deprecated_since_version="1.13",
        active_deprecations_target="deprecated-markers-annotations-fill-rectangles",
        stacklevel=4)


def _create_generic_data_series(**kwargs):
    keywords = ["annotations", "markers", "fill", "rectangles"]
    series = []
    for kw in keywords:
        dictionaries = kwargs.pop(kw, [])
        if dictionaries is None:
            dictionaries = []
        if isinstance(dictionaries, dict):
            dictionaries = [dictionaries]
        for d in dictionaries:
            args = d.pop("args", [])
            series.append(GenericDataSeries(kw, *args, **d))
    return series


class Plot:
    """Base class for all backends. A backend represents the plotting library,
    which implements the necessary functionalities in order to use SymPy
    plotting functions.

    For interactive work the function :func:`plot` is better suited.

    This class permits the plotting of SymPy expressions using numerous
    backends (:external:mod:`matplotlib`, textplot, the old pyglet module for SymPy, Google
    charts api, etc).

    The figure can contain an arbitrary number of plots of SymPy expressions,
    lists of coordinates of points, etc. Plot has a private attribute _series that
    contains all data series to be plotted (expressions for lines or surfaces,
    lists of points, etc (all subclasses of BaseSeries)). Those data series are
    instances of classes not imported by ``from sympy import *``.

    The customization of the figure is on two levels. Global options that
    concern the figure as a whole (e.g. title, xlabel, scale, etc) and
    per-data series options (e.g. name) and aesthetics (e.g. color, point shape,
    line type, etc.).

    The difference between options and aesthetics is that an aesthetic can be
    a function of the coordinates (or parameters in a parametric plot). The
    supported values for an aesthetic are:

    - None (the backend uses default values)
    - a constant
    - a function of one variable (the first coordinate or parameter)
    - a function of two variables (the first and second coordinate or parameters)
    - a function of three variables (only in nonparametric 3D plots)

    Their implementation depends on the backend so they may not work in some
    backends.

    If the plot is parametric and the arity of the aesthetic function permits
    it the aesthetic is calculated over parameters and not over coordinates.
    If the arity does not permit calculation over parameters the calculation is
    done over coordinates.

    Only cartesian coordinates are supported for the moment, but you can use
    the parametric plots to plot in polar, spherical and cylindrical
    coordinates.

    The arguments for the constructor Plot must be subclasses of BaseSeries.

    Any global option can be specified as a keyword argument.

    The global options for a figure are:

    - title : str
    - xlabel : str or Symbol
    - ylabel : str or Symbol
    - zlabel : str or Symbol
    - legend : bool
    - xscale : {'linear', 'log'}
    - yscale : {'linear', 'log'}
    - axis : bool
    - axis_center : tuple of two floats or {'center', 'auto'}
    - xlim : tuple of two floats
    - ylim : tuple of two floats
    - aspect_ratio : tuple of two floats or {'auto'}
    - autoscale : bool
    - margin : float in [0, 1]
    - backend : {'default', 'matplotlib', 'text'} or a subclass of BaseBackend
    - size : optional tuple of two floats, (width, height); default: None

    The per data series options and aesthetics are:
    There are none in the base series. See below for options for subclasses.

    Some data series support additional aesthetics or options:

    :class:`~.LineOver1DRangeSeries`, :class:`~.Parametric2DLineSeries`, and
    :class:`~.Parametric3DLineSeries` support the following:

    Aesthetics:

    - line_color : string, or float, or function, optional
        Specifies the color for the plot, which depends on the backend being
        used.

        For example, if ``MatplotlibBackend`` is being used, then
        Matplotlib string colors are acceptable (``"red"``, ``"r"``,
        ``"cyan"``, ``"c"``, ...).
        Alternatively, we can use a float number, 0 < color < 1, wrapped in a
        string (for example, ``line_color="0.5"``) to specify grayscale colors.
        Alternatively, We can specify a function returning a single
        float value: this will be used to apply a color-loop (for example,
        ``line_color=lambda x: math.cos(x)``).

        Note that by setting line_color, it would be applied simultaneously
        to all the series.

    Options:

    - label : str
    - steps : bool
    - integers_only : bool

    :class:`~.SurfaceOver2DRangeSeries` and :class:`~.ParametricSurfaceSeries`
    support the following:

    Aesthetics:

    - surface_color : function which returns a float.

    Notes
    =====

    How the plotting module works:

    1. Whenever a plotting function is called, the provided expressions are
       processed and a list of instances of the
       :class:`~sympy.plotting.series.BaseSeries` class is created, containing
       the necessary information to plot the expressions
       (e.g. the expression, ranges, series name, ...). Eventually, these
       objects will generate the numerical data to be plotted.
    2. A subclass of :class:`~.Plot` class is instantiaed (referred to as
       backend, from now on), which stores the list of series and the main
       attributes of the plot (e.g. axis labels, title, ...).
       The backend implements the logic to generate the actual figure with
       some plotting library.
    3. When the ``show`` command is executed, series are processed one by one
       to generate numerical data and add it to the figure. The backend is also
       going to set the axis labels, title, ..., according to the values stored
       in the Plot instance.

    The backend should check if it supports the data series that it is given
    (e.g. :class:`TextBackend` supports only
    :class:`~sympy.plotting.series.LineOver1DRangeSeries`).

    It is the backend responsibility to know how to use the class of data series
    that it's given. Note that the current implementation of the ``*Series``
    classes is "matplotlib-centric": the numerical data returned by the
    ``get_points`` and ``get_meshes`` methods is meant to be used directly by
    Matplotlib. Therefore, the new backend will have to pre-process the
    numerical data to make it compatible with the chosen plotting library.
    Keep in mind that future SymPy versions may improve the ``*Series`` classes
    in order to return numerical data "non-matplotlib-centric", hence if you code
    a new backend you have the responsibility to check if its working on each
    SymPy release.

    Please explore the :class:`MatplotlibBackend` source code to understand
    how a backend should be coded.

    In order to be used by SymPy plotting functions, a backend must implement
    the following methods:

    * show(self): used to loop over the data series, generate the numerical
        data, plot it and set the axis labels, title, ...
    * save(self, path): used to save the current plot to the specified file
        path.
    * close(self): used to close the current plot backend (note: some plotting
        library does not support this functionality. In that case, just raise a
        warning).
    """

    def __init__(self, *args,
        title=None, xlabel=None, ylabel=None, zlabel=None, aspect_ratio='auto',
        xlim=None, ylim=None, axis_center='auto', axis=True,
        xscale='linear', yscale='linear', legend=False, autoscale=True,
        margin=0, annotations=None, markers=None, rectangles=None,
        fill=None, backend='default', size=None, **kwargs):

        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring of
        # Plot. They are based purely on convention, no checking is done.
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.aspect_ratio = aspect_ratio
        self.axis_center = axis_center
        self.axis = axis
        self.xscale = xscale
        self.yscale = yscale
        self.legend = legend
        self.autoscale = autoscale
        self.margin = margin
        self._annotations = annotations
        self._markers = markers
        self._rectangles = rectangles
        self._fill = fill

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)
        self._series.extend(_create_generic_data_series(
            annotations=annotations, markers=markers, rectangles=rectangles,
            fill=fill))

        is_real = \
            lambda lim: all(getattr(i, 'is_real', True) for i in lim)
        is_finite = \
            lambda lim: all(getattr(i, 'is_finite', True) for i in lim)

        # reduce code repetition
        def check_and_set(t_name, t):
            if t:
                if not is_real(t):
                    raise ValueError(
                    "All numbers from {}={} must be real".format(t_name, t))
                if not is_finite(t):
                    raise ValueError(
                    "All numbers from {}={} must be finite".format(t_name, t))
                setattr(self, t_name, (float(t[0]), float(t[1])))

        self.xlim = None
        check_and_set("xlim", xlim)
        self.ylim = None
        check_and_set("ylim", ylim)
        self.size = None
        check_and_set("size", size)

    @property
    def _backend(self):
        return self

    @property
    def backend(self):
        return type(self)

    def __str__(self):
        series_strs = [('[%d]: ' % i) + str(s)
                       for i, s in enumerate(self._series)]
        return 'Plot object containing:\n' + '\n'.join(series_strs)

    def __getitem__(self, index):
        return self._series[index]

    def __setitem__(self, index, *args):
        if len(args) == 1 and isinstance(args[0], BaseSeries):
            self._series[index] = args

    def __delitem__(self, index):
        del self._series[index]

    def append(self, arg):
        """Adds an element from a plot's series to an existing plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot's first series object to the first, use the
        ``append`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x*x, show=False)
           >>> p2 = plot(x, show=False)
           >>> p1.append(p2[0])
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           >>> p1.show()

        See Also
        ========

        extend

        """
        if isinstance(arg, BaseSeries):
            self._series.append(arg)
        else:
            raise TypeError('Must specify element of plot to append.')

    def extend(self, arg):
        """Adds all series from another plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot to the first, use the ``extend`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x**2, show=False)
           >>> p2 = plot(x, -x, show=False)
           >>> p1.extend(p2)
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           [2]: cartesian line: -x for x over (-10.0, 10.0)
           >>> p1.show()

        """
        if isinstance(arg, Plot):
            self._series.extend(arg._series)
        elif is_sequence(arg):
            self._series.extend(arg)
        else:
            raise TypeError('Expecting Plot or sequence of BaseSeries')

    def show(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    # deprecations

    @property
    def markers(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("markers")
        return self._markers

    @markers.setter
    def markers(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("markers")
        self._series.extend(_create_generic_data_series(markers=v))
        self._markers = v

    @property
    def annotations(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("annotations")
        return self._annotations

    @annotations.setter
    def annotations(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("annotations")
        self._series.extend(_create_generic_data_series(annotations=v))
        self._annotations = v

    @property
    def rectangles(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("rectangles")
        return self._rectangles

    @rectangles.setter
    def rectangles(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("rectangles")
        self._series.extend(_create_generic_data_series(rectangles=v))
        self._rectangles = v

    @property
    def fill(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("fill")
        return self._fill

    @fill.setter
    def fill(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("fill")
        self._series.extend(_create_generic_data_series(fill=v))
        self._fill = v

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_highlevel_open_tcp_listeners.py ===
from __future__ import annotations

import errno
import socket as stdlib_socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, cast, overload

import attrs
import pytest

import trio
from trio import (
    SocketListener,
    open_tcp_listeners,
    open_tcp_stream,
    serve_tcp,
)
from trio.abc import HostnameResolver, SendStream, SocketFactory
from trio.testing import open_stream_to_socket_listener

from .. import socket as tsocket
from .._core._tests.tutil import binds_ipv6, slow

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Buffer

    from trio._socket import AddressFormat


async def test_open_tcp_listeners_basic() -> None:
    listeners = await open_tcp_listeners(0)
    assert isinstance(listeners, list)
    for obj in listeners:
        assert isinstance(obj, SocketListener)
        # Binds to wildcard address by default
        assert obj.socket.family in [tsocket.AF_INET, tsocket.AF_INET6]
        assert obj.socket.getsockname()[0] in ["0.0.0.0", "::"]

    listener = listeners[0]
    # Make sure the backlog is at least 2
    c1 = await open_stream_to_socket_listener(listener)
    c2 = await open_stream_to_socket_listener(listener)

    s1 = await listener.accept()
    s2 = await listener.accept()

    # Note that we don't know which client stream is connected to which server
    # stream
    await s1.send_all(b"x")
    await s2.send_all(b"x")
    assert await c1.receive_some(1) == b"x"
    assert await c2.receive_some(1) == b"x"

    for resource in [c1, c2, s1, s2, *listeners]:
        await resource.aclose()


async def test_open_tcp_listeners_specific_port_specific_host() -> None:
    # Pick a port
    sock = tsocket.socket()
    await sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    (listener,) = await open_tcp_listeners(port, host=host)
    async with listener:
        assert listener.socket.getsockname() == (host, port)


@binds_ipv6
@slow
async def test_open_tcp_listeners_ipv6_v6only() -> None:
    # Check IPV6_V6ONLY is working properly
    (ipv6_listener,) = await open_tcp_listeners(0, host="::1")
    async with ipv6_listener:
        _, port, *_ = ipv6_listener.socket.getsockname()

        with pytest.raises(
            OSError,
            match=r"(Error|all attempts to) connect(ing)* to (\(')*127\.0\.0\.1(', |:)\d+(\): Connection refused| failed)$",
        ):
            # Windows retries failed connections so this takes seconds
            # (and that's why this is marked @slow)
            await open_tcp_stream("127.0.0.1", port)


async def test_open_tcp_listeners_rebind() -> None:
    (l1,) = await open_tcp_listeners(0, host="127.0.0.1")
    sockaddr1 = l1.socket.getsockname()

    # Plain old rebinding while it's still there should fail, even if we have
    # SO_REUSEADDR set
    with stdlib_socket.socket() as probe:
        probe.setsockopt(stdlib_socket.SOL_SOCKET, stdlib_socket.SO_REUSEADDR, 1)
        with pytest.raises(
            OSError,
            match=r"(Address (already )?in use|An attempt was made to access a socket in a way forbidden by its access permissions)$",
        ):
            probe.bind(sockaddr1)

    # Now use the first listener to set up some connections in various states,
    # and make sure that they don't create any obstacle to rebinding a second
    # listener after the first one is closed.
    c_established = await open_stream_to_socket_listener(l1)
    s_established = await l1.accept()

    c_time_wait = await open_stream_to_socket_listener(l1)
    s_time_wait = await l1.accept()
    # Server-initiated close leaves socket in TIME_WAIT
    await s_time_wait.aclose()

    await l1.aclose()
    (l2,) = await open_tcp_listeners(sockaddr1[1], host="127.0.0.1")
    sockaddr2 = l2.socket.getsockname()

    assert sockaddr1 == sockaddr2
    assert s_established.socket.getsockname() == sockaddr2
    assert c_time_wait.socket.getpeername() == sockaddr2

    for resource in [
        l1,
        l2,
        c_established,
        s_established,
        c_time_wait,
        s_time_wait,
    ]:
        await resource.aclose()


class FakeOSError(OSError):
    pass


@attrs.define(slots=False)
class FakeSocket(tsocket.SocketType):
    _family: AddressFamily = attrs.field(converter=AddressFamily)
    _type: SocketKind = attrs.field(converter=SocketKind)
    _proto: int

    closed: bool = False
    poison_listen: bool = False
    backlog: int | None = None

    @property
    def type(self) -> SocketKind:
        return self._type

    @property
    def family(self) -> AddressFamily:
        return self._family

    @property
    def proto(self) -> int:  # pragma: no cover
        return self._proto

    @overload
    def getsockopt(self, /, level: int, optname: int) -> int: ...

    @overload
    def getsockopt(self, /, level: int, optname: int, buflen: int) -> bytes: ...

    def getsockopt(
        self,
        /,
        level: int,
        optname: int,
        buflen: int | None = None,
    ) -> int | bytes:
        if (level, optname) == (tsocket.SOL_SOCKET, tsocket.SO_ACCEPTCONN):
            return True
        raise AssertionError()  # pragma: no cover

    @overload
    def setsockopt(self, /, level: int, optname: int, value: int | Buffer) -> None: ...

    @overload
    def setsockopt(
        self,
        /,
        level: int,
        optname: int,
        value: None,
        optlen: int,
    ) -> None: ...

    def setsockopt(
        self,
        /,
        level: int,
        optname: int,
        value: int | Buffer | None,
        optlen: int | None = None,
    ) -> None:
        pass

    async def bind(self, address: AddressFormat) -> None:
        pass

    def listen(self, /, backlog: int = min(stdlib_socket.SOMAXCONN, 128)) -> None:
        assert self.backlog is None
        assert backlog is not None
        self.backlog = backlog
        if self.poison_listen:
            raise FakeOSError("whoops")

    def close(self) -> None:
        self.closed = True


@attrs.define(slots=False)
class FakeSocketFactory(SocketFactory):
    poison_after: int
    sockets: list[tsocket.SocketType] = attrs.Factory(list)
    raise_on_family: dict[AddressFamily, int] = attrs.Factory(dict)  # family => errno

    def socket(
        self,
        family: AddressFamily | int | None = None,
        type_: SocketKind | int | None = None,
        proto: int = 0,
    ) -> tsocket.SocketType:
        assert family is not None
        assert type_ is not None
        if isinstance(family, int) and not isinstance(family, AddressFamily):
            family = AddressFamily(family)  # pragma: no cover
        if family in self.raise_on_family:
            raise OSError(self.raise_on_family[family], "nope")
        sock = FakeSocket(family, type_, proto)
        self.poison_after -= 1
        if self.poison_after == 0:
            sock.poison_listen = True
        self.sockets.append(sock)
        return sock


@attrs.define(slots=False)
class FakeHostnameResolver(HostnameResolver):
    family_addr_pairs: Sequence[tuple[AddressFamily, str]]

    async def getaddrinfo(
        self,
        host: bytes | None,
        port: bytes | str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[
        tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]
    ]:
        assert isinstance(port, int)
        return [
            (family, tsocket.SOCK_STREAM, 0, "", (addr, port))
            for family, addr in self.family_addr_pairs
        ]

    async def getnameinfo(
        self,
        sockaddr: tuple[str, int] | tuple[str, int, int, int],
        flags: int,
    ) -> tuple[str, str]:
        raise NotImplementedError()


async def test_open_tcp_listeners_multiple_host_cleanup_on_error() -> None:
    # If we were trying to bind to multiple hosts and one of them failed, they
    # call get cleaned up before returning
    fsf = FakeSocketFactory(3)
    tsocket.set_custom_socket_factory(fsf)
    tsocket.set_custom_hostname_resolver(
        FakeHostnameResolver(
            [
                (tsocket.AF_INET, "1.1.1.1"),
                (tsocket.AF_INET, "2.2.2.2"),
                (tsocket.AF_INET, "3.3.3.3"),
            ],
        ),
    )

    with pytest.raises(FakeOSError):
        await open_tcp_listeners(80, host="example.org")

    assert len(fsf.sockets) == 3
    for sock in fsf.sockets:
        # property only exists on FakeSocket
        assert sock.closed  # type: ignore[attr-defined]


async def test_open_tcp_listeners_port_checking() -> None:
    for host in ["127.0.0.1", None]:
        with pytest.raises(TypeError):
            await open_tcp_listeners(None, host=host)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await open_tcp_listeners(b"80", host=host)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await open_tcp_listeners("http", host=host)  # type: ignore[arg-type]


async def test_serve_tcp() -> None:
    async def handler(stream: SendStream) -> None:
        await stream.send_all(b"x")

    async with trio.open_nursery() as nursery:
        # nursery.start is incorrectly typed, awaiting #2773
        value = await nursery.start(serve_tcp, handler, 0)
        assert isinstance(value, list)
        listeners = cast("list[SocketListener]", value)
        stream = await open_stream_to_socket_listener(listeners[0])
        async with stream:
            assert await stream.receive_some(1) == b"x"
            nursery.cancel_scope.cancel()


@pytest.mark.parametrize(
    "try_families",
    [{tsocket.AF_INET}, {tsocket.AF_INET6}, {tsocket.AF_INET, tsocket.AF_INET6}],
)
@pytest.mark.parametrize(
    "fail_families",
    [{tsocket.AF_INET}, {tsocket.AF_INET6}, {tsocket.AF_INET, tsocket.AF_INET6}],
)
async def test_open_tcp_listeners_some_address_families_unavailable(
    try_families: set[AddressFamily],
    fail_families: set[AddressFamily],
) -> None:
    fsf = FakeSocketFactory(
        10,
        raise_on_family=dict.fromkeys(fail_families, errno.EAFNOSUPPORT),
    )
    tsocket.set_custom_socket_factory(fsf)
    tsocket.set_custom_hostname_resolver(
        FakeHostnameResolver([(family, "foo") for family in try_families]),
    )

    should_succeed = try_families - fail_families

    if not should_succeed:
        with pytest.raises(OSError, match="This system doesn't support") as exc_info:
            await open_tcp_listeners(80, host="example.org")

        # open_listeners always creates an exceptiongroup with the
        # unsupported address families, regardless of the value of
        # strict_exception_groups or number of unsupported families.
        assert isinstance(exc_info.value.__cause__, BaseExceptionGroup)
        for subexc in exc_info.value.__cause__.exceptions:
            assert "nope" in str(subexc)
    else:
        listeners = await open_tcp_listeners(80)
        for listener in listeners:
            should_succeed.remove(listener.socket.family)
        assert not should_succeed


async def test_open_tcp_listeners_socket_fails_not_afnosupport() -> None:
    fsf = FakeSocketFactory(
        10,
        raise_on_family={
            tsocket.AF_INET: errno.EAFNOSUPPORT,
            tsocket.AF_INET6: errno.EINVAL,
        },
    )
    tsocket.set_custom_socket_factory(fsf)
    tsocket.set_custom_hostname_resolver(
        FakeHostnameResolver([(tsocket.AF_INET, "foo"), (tsocket.AF_INET6, "bar")]),
    )

    with pytest.raises(OSError, match="nope") as exc_info:
        await open_tcp_listeners(80, host="example.org")
    assert exc_info.value.errno == errno.EINVAL
    assert exc_info.value.__cause__ is None
    assert "nope" in str(exc_info.value)


# We used to have an elaborate test that opened a real TCP listening socket
# and then tried to measure its backlog by making connections to it. And most
# of the time, it worked. But no matter what we tried, it was always fragile,
# because it had to do things like use timeouts to guess when the listening
# queue was full, sometimes the CI hosts go into SYN-cookie mode (where there
# effectively is no backlog), sometimes the host might not be enough resources
# to give us the full requested backlog... it was a mess. So now we just check
# that the backlog argument is passed through correctly.
async def test_open_tcp_listeners_backlog() -> None:
    fsf = FakeSocketFactory(99)
    tsocket.set_custom_socket_factory(fsf)
    for given, expected in [
        (None, 0xFFFF),
        (99999999, 0xFFFF),
        (10, 10),
        (1, 1),
    ]:
        listeners = await open_tcp_listeners(0, backlog=given)
        assert listeners
        for listener in listeners:
            # `backlog` only exists on FakeSocket
            assert listener.socket.backlog == expected  # type: ignore[attr-defined]


async def test_open_tcp_listeners_backlog_float_error() -> None:
    fsf = FakeSocketFactory(99)
    tsocket.set_custom_socket_factory(fsf)
    for should_fail in (0.0, 2.18, 3.15, 9.75):
        with pytest.raises(
            TypeError,
            match=f"backlog must be an int or None, not {should_fail!r}",
        ):
            await open_tcp_listeners(0, backlog=should_fail)  # type: ignore[arg-type]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\compat\numpy\function.py ===
"""
For compatibility with numpy libraries, pandas functions or methods have to
accept '*args' and '**kwargs' parameters to accommodate numpy arguments that
are not actually used or respected in the pandas implementation.

To ensure that users do not abuse these parameters, validation is performed in
'validators.py' to make sure that any extra parameters passed correspond ONLY
to those in the numpy signature. Part of that validation includes whether or
not the user attempted to pass in non-default values for these extraneous
parameters. As we want to discourage users from relying on these parameters
when calling the pandas implementation, we want them only to pass in the
default values for these parameters.

This module provides a set of commonly used default arguments for functions and
methods that are spread throughout the codebase. This module will make it
easier to adjust to future upstream changes in the analogous numpy signatures.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from numpy import ndarray

from pandas._libs.lib import (
    is_bool,
    is_integer,
)
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
    validate_args,
    validate_args_and_kwargs,
    validate_kwargs,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        AxisInt,
    )

    AxisNoneT = TypeVar("AxisNoneT", Axis, None)


class CompatValidator:
    def __init__(
        self,
        defaults,
        fname=None,
        method: str | None = None,
        max_fname_arg_count=None,
    ) -> None:
        self.fname = fname
        self.method = method
        self.defaults = defaults
        self.max_fname_arg_count = max_fname_arg_count

    def __call__(
        self,
        args,
        kwargs,
        fname=None,
        max_fname_arg_count=None,
        method: str | None = None,
    ) -> None:
        if not args and not kwargs:
            return None

        fname = self.fname if fname is None else fname
        max_fname_arg_count = (
            self.max_fname_arg_count
            if max_fname_arg_count is None
            else max_fname_arg_count
        )
        method = self.method if method is None else method

        if method == "args":
            validate_args(fname, args, max_fname_arg_count, self.defaults)
        elif method == "kwargs":
            validate_kwargs(fname, kwargs, self.defaults)
        elif method == "both":
            validate_args_and_kwargs(
                fname, args, kwargs, max_fname_arg_count, self.defaults
            )
        else:
            raise ValueError(f"invalid validation method '{method}'")


ARGMINMAX_DEFAULTS = {"out": None}
validate_argmin = CompatValidator(
    ARGMINMAX_DEFAULTS, fname="argmin", method="both", max_fname_arg_count=1
)
validate_argmax = CompatValidator(
    ARGMINMAX_DEFAULTS, fname="argmax", method="both", max_fname_arg_count=1
)


def process_skipna(skipna: bool | ndarray | None, args) -> tuple[bool, Any]:
    if isinstance(skipna, ndarray) or skipna is None:
        args = (skipna,) + args
        skipna = True

    return skipna, args


def validate_argmin_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmin' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmin(args, kwargs)
    return skipna


def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmax' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmax(args, kwargs)
    return skipna


ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None
ARGSORT_DEFAULTS["stable"] = None


validate_argsort = CompatValidator(
    ARGSORT_DEFAULTS, fname="argsort", max_fname_arg_count=0, method="both"
)

# two different signatures of argsort, this second validation for when the
# `kind` param is supported
ARGSORT_DEFAULTS_KIND: dict[str, int | None] = {}
ARGSORT_DEFAULTS_KIND["axis"] = -1
ARGSORT_DEFAULTS_KIND["order"] = None
ARGSORT_DEFAULTS_KIND["stable"] = None
validate_argsort_kind = CompatValidator(
    ARGSORT_DEFAULTS_KIND, fname="argsort", max_fname_arg_count=0, method="both"
)


def validate_argsort_with_ascending(ascending: bool | int | None, args, kwargs) -> bool:
    """
    If 'Categorical.argsort' is called via the 'numpy' library, the first
    parameter in its signature is 'axis', which takes either an integer or
    'None', so check if the 'ascending' parameter has either integer type or is
    None, since 'ascending' itself should be a boolean
    """
    if is_integer(ascending) or ascending is None:
        args = (ascending,) + args
        ascending = True

    validate_argsort_kind(args, kwargs, max_fname_arg_count=3)
    ascending = cast(bool, ascending)
    return ascending


CLIP_DEFAULTS: dict[str, Any] = {"out": None}
validate_clip = CompatValidator(
    CLIP_DEFAULTS, fname="clip", method="both", max_fname_arg_count=3
)


@overload
def validate_clip_with_axis(axis: ndarray, args, kwargs) -> None:
    ...


@overload
def validate_clip_with_axis(axis: AxisNoneT, args, kwargs) -> AxisNoneT:
    ...


def validate_clip_with_axis(
    axis: ndarray | AxisNoneT, args, kwargs
) -> AxisNoneT | None:
    """
    If 'NDFrame.clip' is called via the numpy library, the third parameter in
    its signature is 'out', which can takes an ndarray, so check if the 'axis'
    parameter is an instance of ndarray, since 'axis' itself should either be
    an integer or None
    """
    if isinstance(axis, ndarray):
        args = (axis,) + args
        # error: Incompatible types in assignment (expression has type "None",
        # variable has type "Union[ndarray[Any, Any], str, int]")
        axis = None  # type: ignore[assignment]

    validate_clip(args, kwargs)
    # error: Incompatible return value type (got "Union[ndarray[Any, Any],
    # str, int]", expected "Union[str, int, None]")
    return axis  # type: ignore[return-value]


CUM_FUNC_DEFAULTS: dict[str, Any] = {}
CUM_FUNC_DEFAULTS["dtype"] = None
CUM_FUNC_DEFAULTS["out"] = None
validate_cum_func = CompatValidator(
    CUM_FUNC_DEFAULTS, method="both", max_fname_arg_count=1
)
validate_cumsum = CompatValidator(
    CUM_FUNC_DEFAULTS, fname="cumsum", method="both", max_fname_arg_count=1
)


def validate_cum_func_with_skipna(skipna: bool, args, kwargs, name) -> bool:
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'dtype', which takes either a 'numpy' dtype or 'None', so
    check if the 'skipna' parameter is a boolean or not
    """
    if not is_bool(skipna):
        args = (skipna,) + args
        skipna = True
    elif isinstance(skipna, np.bool_):
        skipna = bool(skipna)

    validate_cum_func(args, kwargs, fname=name)
    return skipna


ALLANY_DEFAULTS: dict[str, bool | None] = {}
ALLANY_DEFAULTS["dtype"] = None
ALLANY_DEFAULTS["out"] = None
ALLANY_DEFAULTS["keepdims"] = False
ALLANY_DEFAULTS["axis"] = None
validate_all = CompatValidator(
    ALLANY_DEFAULTS, fname="all", method="both", max_fname_arg_count=1
)
validate_any = CompatValidator(
    ALLANY_DEFAULTS, fname="any", method="both", max_fname_arg_count=1
)

LOGICAL_FUNC_DEFAULTS = {"out": None, "keepdims": False}
validate_logical_func = CompatValidator(LOGICAL_FUNC_DEFAULTS, method="kwargs")

MINMAX_DEFAULTS = {"axis": None, "dtype": None, "out": None, "keepdims": False}
validate_min = CompatValidator(
    MINMAX_DEFAULTS, fname="min", method="both", max_fname_arg_count=1
)
validate_max = CompatValidator(
    MINMAX_DEFAULTS, fname="max", method="both", max_fname_arg_count=1
)

RESHAPE_DEFAULTS: dict[str, str] = {"order": "C"}
validate_reshape = CompatValidator(
    RESHAPE_DEFAULTS, fname="reshape", method="both", max_fname_arg_count=1
)

REPEAT_DEFAULTS: dict[str, Any] = {"axis": None}
validate_repeat = CompatValidator(
    REPEAT_DEFAULTS, fname="repeat", method="both", max_fname_arg_count=1
)

ROUND_DEFAULTS: dict[str, Any] = {"out": None}
validate_round = CompatValidator(
    ROUND_DEFAULTS, fname="round", method="both", max_fname_arg_count=1
)

SORT_DEFAULTS: dict[str, int | str | None] = {}
SORT_DEFAULTS["axis"] = -1
SORT_DEFAULTS["kind"] = "quicksort"
SORT_DEFAULTS["order"] = None
validate_sort = CompatValidator(SORT_DEFAULTS, fname="sort", method="kwargs")

STAT_FUNC_DEFAULTS: dict[str, Any | None] = {}
STAT_FUNC_DEFAULTS["dtype"] = None
STAT_FUNC_DEFAULTS["out"] = None

SUM_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
SUM_DEFAULTS["axis"] = None
SUM_DEFAULTS["keepdims"] = False
SUM_DEFAULTS["initial"] = None

PROD_DEFAULTS = SUM_DEFAULTS.copy()

MEAN_DEFAULTS = SUM_DEFAULTS.copy()

MEDIAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
MEDIAN_DEFAULTS["overwrite_input"] = False
MEDIAN_DEFAULTS["keepdims"] = False

STAT_FUNC_DEFAULTS["keepdims"] = False

validate_stat_func = CompatValidator(STAT_FUNC_DEFAULTS, method="kwargs")
validate_sum = CompatValidator(
    SUM_DEFAULTS, fname="sum", method="both", max_fname_arg_count=1
)
validate_prod = CompatValidator(
    PROD_DEFAULTS, fname="prod", method="both", max_fname_arg_count=1
)
validate_mean = CompatValidator(
    MEAN_DEFAULTS, fname="mean", method="both", max_fname_arg_count=1
)
validate_median = CompatValidator(
    MEDIAN_DEFAULTS, fname="median", method="both", max_fname_arg_count=1
)

STAT_DDOF_FUNC_DEFAULTS: dict[str, bool | None] = {}
STAT_DDOF_FUNC_DEFAULTS["dtype"] = None
STAT_DDOF_FUNC_DEFAULTS["out"] = None
STAT_DDOF_FUNC_DEFAULTS["keepdims"] = False
validate_stat_ddof_func = CompatValidator(STAT_DDOF_FUNC_DEFAULTS, method="kwargs")

TAKE_DEFAULTS: dict[str, str | None] = {}
TAKE_DEFAULTS["out"] = None
TAKE_DEFAULTS["mode"] = "raise"
validate_take = CompatValidator(TAKE_DEFAULTS, fname="take", method="kwargs")


def validate_take_with_convert(convert: ndarray | bool | None, args, kwargs) -> bool:
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'axis', which takes either an ndarray or 'None', so check
    if the 'convert' parameter is either an instance of ndarray or is None
    """
    if isinstance(convert, ndarray) or convert is None:
        args = (convert,) + args
        convert = True

    validate_take(args, kwargs, max_fname_arg_count=3, method="both")
    return convert


TRANSPOSE_DEFAULTS = {"axes": None}
validate_transpose = CompatValidator(
    TRANSPOSE_DEFAULTS, fname="transpose", method="both", max_fname_arg_count=0
)


def validate_groupby_func(name: str, args, kwargs, allowed=None) -> None:
    """
    'args' and 'kwargs' should be empty, except for allowed kwargs because all
    of their necessary parameters are explicitly listed in the function
    signature
    """
    if allowed is None:
        allowed = []

    kwargs = set(kwargs) - set(allowed)

    if len(args) + len(kwargs) > 0:
        raise UnsupportedFunctionCall(
            "numpy operations are not valid with groupby. "
            f"Use .groupby(...).{name}() instead"
        )


RESAMPLER_NUMPY_OPS = ("min", "max", "sum", "prod", "mean", "std", "var")


def validate_resampler_func(method: str, args, kwargs) -> None:
    """
    'args' and 'kwargs' should be empty because all of their necessary
    parameters are explicitly listed in the function signature
    """
    if len(args) + len(kwargs) > 0:
        if method in RESAMPLER_NUMPY_OPS:
            raise UnsupportedFunctionCall(
                "numpy operations are not valid with resample. "
                f"Use .resample(...).{method}() instead"
            )
        raise TypeError("too many arguments passed in")


def validate_minmax_axis(axis: AxisInt | None, ndim: int = 1) -> None:
    """
    Ensure that the axis argument passed to min, max, argmin, or argmax is zero
    or None, as otherwise it will be incorrectly ignored.

    Parameters
    ----------
    axis : int or None
    ndim : int, default 1

    Raises
    ------
    ValueError
    """
    if axis is None:
        return
    if axis >= ndim or (axis < 0 and ndim + axis < 0):
        raise ValueError(f"`axis` must be fewer than the number of dimensions ({ndim})")


_validation_funcs = {
    "median": validate_median,
    "mean": validate_mean,
    "min": validate_min,
    "max": validate_max,
    "sum": validate_sum,
    "prod": validate_prod,
}


def validate_func(fname, args, kwargs) -> None:
    if fname not in _validation_funcs:
        return validate_stat_func(args, kwargs, fname=fname)

    validation_func = _validation_funcs[fname]
    return validation_func(args, kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\profiler.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Profiler
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime


@dataclass
class ProfileNode:
    '''
    Profile node. Holds callsite information, execution statistics and child nodes.
    '''
    #: Unique id of the node.
    id_: int

    #: Function location.
    call_frame: runtime.CallFrame

    #: Number of samples where this node was on top of the call stack.
    hit_count: typing.Optional[int] = None

    #: Child node ids.
    children: typing.Optional[typing.List[int]] = None

    #: The reason of being not optimized. The function may be deoptimized or marked as don't
    #: optimize.
    deopt_reason: typing.Optional[str] = None

    #: An array of source position ticks.
    position_ticks: typing.Optional[typing.List[PositionTickInfo]] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['callFrame'] = self.call_frame.to_json()
        if self.hit_count is not None:
            json['hitCount'] = self.hit_count
        if self.children is not None:
            json['children'] = [i for i in self.children]
        if self.deopt_reason is not None:
            json['deoptReason'] = self.deopt_reason
        if self.position_ticks is not None:
            json['positionTicks'] = [i.to_json() for i in self.position_ticks]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=int(json['id']),
            call_frame=runtime.CallFrame.from_json(json['callFrame']),
            hit_count=int(json['hitCount']) if 'hitCount' in json else None,
            children=[int(i) for i in json['children']] if 'children' in json else None,
            deopt_reason=str(json['deoptReason']) if 'deoptReason' in json else None,
            position_ticks=[PositionTickInfo.from_json(i) for i in json['positionTicks']] if 'positionTicks' in json else None,
        )


@dataclass
class Profile:
    '''
    Profile.
    '''
    #: The list of profile nodes. First item is the root node.
    nodes: typing.List[ProfileNode]

    #: Profiling start timestamp in microseconds.
    start_time: float

    #: Profiling end timestamp in microseconds.
    end_time: float

    #: Ids of samples top nodes.
    samples: typing.Optional[typing.List[int]] = None

    #: Time intervals between adjacent samples in microseconds. The first delta is relative to the
    #: profile startTime.
    time_deltas: typing.Optional[typing.List[int]] = None

    def to_json(self):
        json = dict()
        json['nodes'] = [i.to_json() for i in self.nodes]
        json['startTime'] = self.start_time
        json['endTime'] = self.end_time
        if self.samples is not None:
            json['samples'] = [i for i in self.samples]
        if self.time_deltas is not None:
            json['timeDeltas'] = [i for i in self.time_deltas]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            nodes=[ProfileNode.from_json(i) for i in json['nodes']],
            start_time=float(json['startTime']),
            end_time=float(json['endTime']),
            samples=[int(i) for i in json['samples']] if 'samples' in json else None,
            time_deltas=[int(i) for i in json['timeDeltas']] if 'timeDeltas' in json else None,
        )


@dataclass
class PositionTickInfo:
    '''
    Specifies a number of samples attributed to a certain source position.
    '''
    #: Source line number (1-based).
    line: int

    #: Number of samples attributed to the source line.
    ticks: int

    def to_json(self):
        json = dict()
        json['line'] = self.line
        json['ticks'] = self.ticks
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line=int(json['line']),
            ticks=int(json['ticks']),
        )


@dataclass
class CoverageRange:
    '''
    Coverage data for a source range.
    '''
    #: JavaScript script source offset for the range start.
    start_offset: int

    #: JavaScript script source offset for the range end.
    end_offset: int

    #: Collected execution count of the source range.
    count: int

    def to_json(self):
        json = dict()
        json['startOffset'] = self.start_offset
        json['endOffset'] = self.end_offset
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            start_offset=int(json['startOffset']),
            end_offset=int(json['endOffset']),
            count=int(json['count']),
        )


@dataclass
class FunctionCoverage:
    '''
    Coverage data for a JavaScript function.
    '''
    #: JavaScript function name.
    function_name: str

    #: Source ranges inside the function with coverage data.
    ranges: typing.List[CoverageRange]

    #: Whether coverage data for this function has block granularity.
    is_block_coverage: bool

    def to_json(self):
        json = dict()
        json['functionName'] = self.function_name
        json['ranges'] = [i.to_json() for i in self.ranges]
        json['isBlockCoverage'] = self.is_block_coverage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            function_name=str(json['functionName']),
            ranges=[CoverageRange.from_json(i) for i in json['ranges']],
            is_block_coverage=bool(json['isBlockCoverage']),
        )


@dataclass
class ScriptCoverage:
    '''
    Coverage data for a JavaScript script.
    '''
    #: JavaScript script id.
    script_id: runtime.ScriptId

    #: JavaScript script name or url.
    url: str

    #: Functions contained in the script that has coverage data.
    functions: typing.List[FunctionCoverage]

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['functions'] = [i.to_json() for i in self.functions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            functions=[FunctionCoverage.from_json(i) for i in json['functions']],
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.enable',
    }
    json = yield cmd_dict


def get_best_effort_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[ScriptCoverage]]:
    '''
    Collect coverage data for the current isolate. The coverage data may be incomplete due to
    garbage collection.

    :returns: Coverage data for the current isolate.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.getBestEffortCoverage',
    }
    json = yield cmd_dict
    return [ScriptCoverage.from_json(i) for i in json['result']]


def set_sampling_interval(
        interval: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes CPU profiler sampling interval. Must be called before CPU profiles recording started.

    :param interval: New sampling interval in microseconds.
    '''
    params: T_JSON_DICT = dict()
    params['interval'] = interval
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.setSamplingInterval',
        'params': params,
    }
    json = yield cmd_dict


def start() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.start',
    }
    json = yield cmd_dict


def start_precise_coverage(
        call_count: typing.Optional[bool] = None,
        detailed: typing.Optional[bool] = None,
        allow_triggered_updates: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Enable precise code coverage. Coverage data for JavaScript executed before enabling precise code
    coverage may be incomplete. Enabling prevents running optimized code and resets execution
    counters.

    :param call_count: *(Optional)* Collect accurate call counts beyond simple 'covered' or 'not covered'.
    :param detailed: *(Optional)* Collect block-based coverage.
    :param allow_triggered_updates: *(Optional)* Allow the backend to send updates on its own initiative
    :returns: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    '''
    params: T_JSON_DICT = dict()
    if call_count is not None:
        params['callCount'] = call_count
    if detailed is not None:
        params['detailed'] = detailed
    if allow_triggered_updates is not None:
        params['allowTriggeredUpdates'] = allow_triggered_updates
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.startPreciseCoverage',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['timestamp'])


def stop() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Profile]:
    '''


    :returns: Recorded profile.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.stop',
    }
    json = yield cmd_dict
    return Profile.from_json(json['profile'])


def stop_precise_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable precise code coverage. Disabling releases unnecessary execution count records and allows
    executing optimized code.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.stopPreciseCoverage',
    }
    json = yield cmd_dict


def take_precise_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[ScriptCoverage], float]]:
    '''
    Collect coverage data for the current isolate, and resets execution counters. Precise code
    coverage needs to have started.

    :returns: A tuple with the following items:

        0. **result** - Coverage data for the current isolate.
        1. **timestamp** - Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.takePreciseCoverage',
    }
    json = yield cmd_dict
    return (
        [ScriptCoverage.from_json(i) for i in json['result']],
        float(json['timestamp'])
    )


@event_class('Profiler.consoleProfileFinished')
@dataclass
class ConsoleProfileFinished:
    id_: str
    #: Location of console.profileEnd().
    location: debugger.Location
    profile: Profile
    #: Profile title passed as an argument to console.profile().
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileFinished:
        return cls(
            id_=str(json['id']),
            location=debugger.Location.from_json(json['location']),
            profile=Profile.from_json(json['profile']),
            title=str(json['title']) if 'title' in json else None
        )


@event_class('Profiler.consoleProfileStarted')
@dataclass
class ConsoleProfileStarted:
    '''
    Sent when new profile recording is started using console.profile() call.
    '''
    id_: str
    #: Location of console.profile().
    location: debugger.Location
    #: Profile title passed as an argument to console.profile().
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileStarted:
        return cls(
            id_=str(json['id']),
            location=debugger.Location.from_json(json['location']),
            title=str(json['title']) if 'title' in json else None
        )


@event_class('Profiler.preciseCoverageDeltaUpdate')
@dataclass
class PreciseCoverageDeltaUpdate:
    '''
    **EXPERIMENTAL**

    Reports coverage delta since the last poll (either from an event like this, or from
    ``takePreciseCoverage`` for the current isolate. May only be sent if precise code
    coverage has been started. This event can be trigged by the embedder to, for example,
    trigger collection of coverage data immediately at a certain point in time.
    '''
    #: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    timestamp: float
    #: Identifier for distinguishing coverage events.
    occasion: str
    #: Coverage data for the current isolate.
    result: typing.List[ScriptCoverage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreciseCoverageDeltaUpdate:
        return cls(
            timestamp=float(json['timestamp']),
            occasion=str(json['occasion']),
            result=[ScriptCoverage.from_json(i) for i in json['result']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\profiler.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Profiler
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime


@dataclass
class ProfileNode:
    '''
    Profile node. Holds callsite information, execution statistics and child nodes.
    '''
    #: Unique id of the node.
    id_: int

    #: Function location.
    call_frame: runtime.CallFrame

    #: Number of samples where this node was on top of the call stack.
    hit_count: typing.Optional[int] = None

    #: Child node ids.
    children: typing.Optional[typing.List[int]] = None

    #: The reason of being not optimized. The function may be deoptimized or marked as don't
    #: optimize.
    deopt_reason: typing.Optional[str] = None

    #: An array of source position ticks.
    position_ticks: typing.Optional[typing.List[PositionTickInfo]] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['callFrame'] = self.call_frame.to_json()
        if self.hit_count is not None:
            json['hitCount'] = self.hit_count
        if self.children is not None:
            json['children'] = [i for i in self.children]
        if self.deopt_reason is not None:
            json['deoptReason'] = self.deopt_reason
        if self.position_ticks is not None:
            json['positionTicks'] = [i.to_json() for i in self.position_ticks]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=int(json['id']),
            call_frame=runtime.CallFrame.from_json(json['callFrame']),
            hit_count=int(json['hitCount']) if 'hitCount' in json else None,
            children=[int(i) for i in json['children']] if 'children' in json else None,
            deopt_reason=str(json['deoptReason']) if 'deoptReason' in json else None,
            position_ticks=[PositionTickInfo.from_json(i) for i in json['positionTicks']] if 'positionTicks' in json else None,
        )


@dataclass
class Profile:
    '''
    Profile.
    '''
    #: The list of profile nodes. First item is the root node.
    nodes: typing.List[ProfileNode]

    #: Profiling start timestamp in microseconds.
    start_time: float

    #: Profiling end timestamp in microseconds.
    end_time: float

    #: Ids of samples top nodes.
    samples: typing.Optional[typing.List[int]] = None

    #: Time intervals between adjacent samples in microseconds. The first delta is relative to the
    #: profile startTime.
    time_deltas: typing.Optional[typing.List[int]] = None

    def to_json(self):
        json = dict()
        json['nodes'] = [i.to_json() for i in self.nodes]
        json['startTime'] = self.start_time
        json['endTime'] = self.end_time
        if self.samples is not None:
            json['samples'] = [i for i in self.samples]
        if self.time_deltas is not None:
            json['timeDeltas'] = [i for i in self.time_deltas]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            nodes=[ProfileNode.from_json(i) for i in json['nodes']],
            start_time=float(json['startTime']),
            end_time=float(json['endTime']),
            samples=[int(i) for i in json['samples']] if 'samples' in json else None,
            time_deltas=[int(i) for i in json['timeDeltas']] if 'timeDeltas' in json else None,
        )


@dataclass
class PositionTickInfo:
    '''
    Specifies a number of samples attributed to a certain source position.
    '''
    #: Source line number (1-based).
    line: int

    #: Number of samples attributed to the source line.
    ticks: int

    def to_json(self):
        json = dict()
        json['line'] = self.line
        json['ticks'] = self.ticks
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line=int(json['line']),
            ticks=int(json['ticks']),
        )


@dataclass
class CoverageRange:
    '''
    Coverage data for a source range.
    '''
    #: JavaScript script source offset for the range start.
    start_offset: int

    #: JavaScript script source offset for the range end.
    end_offset: int

    #: Collected execution count of the source range.
    count: int

    def to_json(self):
        json = dict()
        json['startOffset'] = self.start_offset
        json['endOffset'] = self.end_offset
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            start_offset=int(json['startOffset']),
            end_offset=int(json['endOffset']),
            count=int(json['count']),
        )


@dataclass
class FunctionCoverage:
    '''
    Coverage data for a JavaScript function.
    '''
    #: JavaScript function name.
    function_name: str

    #: Source ranges inside the function with coverage data.
    ranges: typing.List[CoverageRange]

    #: Whether coverage data for this function has block granularity.
    is_block_coverage: bool

    def to_json(self):
        json = dict()
        json['functionName'] = self.function_name
        json['ranges'] = [i.to_json() for i in self.ranges]
        json['isBlockCoverage'] = self.is_block_coverage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            function_name=str(json['functionName']),
            ranges=[CoverageRange.from_json(i) for i in json['ranges']],
            is_block_coverage=bool(json['isBlockCoverage']),
        )


@dataclass
class ScriptCoverage:
    '''
    Coverage data for a JavaScript script.
    '''
    #: JavaScript script id.
    script_id: runtime.ScriptId

    #: JavaScript script name or url.
    url: str

    #: Functions contained in the script that has coverage data.
    functions: typing.List[FunctionCoverage]

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['functions'] = [i.to_json() for i in self.functions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            functions=[FunctionCoverage.from_json(i) for i in json['functions']],
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.enable',
    }
    json = yield cmd_dict


def get_best_effort_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[ScriptCoverage]]:
    '''
    Collect coverage data for the current isolate. The coverage data may be incomplete due to
    garbage collection.

    :returns: Coverage data for the current isolate.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.getBestEffortCoverage',
    }
    json = yield cmd_dict
    return [ScriptCoverage.from_json(i) for i in json['result']]


def set_sampling_interval(
        interval: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes CPU profiler sampling interval. Must be called before CPU profiles recording started.

    :param interval: New sampling interval in microseconds.
    '''
    params: T_JSON_DICT = dict()
    params['interval'] = interval
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.setSamplingInterval',
        'params': params,
    }
    json = yield cmd_dict


def start() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.start',
    }
    json = yield cmd_dict


def start_precise_coverage(
        call_count: typing.Optional[bool] = None,
        detailed: typing.Optional[bool] = None,
        allow_triggered_updates: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Enable precise code coverage. Coverage data for JavaScript executed before enabling precise code
    coverage may be incomplete. Enabling prevents running optimized code and resets execution
    counters.

    :param call_count: *(Optional)* Collect accurate call counts beyond simple 'covered' or 'not covered'.
    :param detailed: *(Optional)* Collect block-based coverage.
    :param allow_triggered_updates: *(Optional)* Allow the backend to send updates on its own initiative
    :returns: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    '''
    params: T_JSON_DICT = dict()
    if call_count is not None:
        params['callCount'] = call_count
    if detailed is not None:
        params['detailed'] = detailed
    if allow_triggered_updates is not None:
        params['allowTriggeredUpdates'] = allow_triggered_updates
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.startPreciseCoverage',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['timestamp'])


def stop() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Profile]:
    '''


    :returns: Recorded profile.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.stop',
    }
    json = yield cmd_dict
    return Profile.from_json(json['profile'])


def stop_precise_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable precise code coverage. Disabling releases unnecessary execution count records and allows
    executing optimized code.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.stopPreciseCoverage',
    }
    json = yield cmd_dict


def take_precise_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[ScriptCoverage], float]]:
    '''
    Collect coverage data for the current isolate, and resets execution counters. Precise code
    coverage needs to have started.

    :returns: A tuple with the following items:

        0. **result** - Coverage data for the current isolate.
        1. **timestamp** - Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.takePreciseCoverage',
    }
    json = yield cmd_dict
    return (
        [ScriptCoverage.from_json(i) for i in json['result']],
        float(json['timestamp'])
    )


@event_class('Profiler.consoleProfileFinished')
@dataclass
class ConsoleProfileFinished:
    id_: str
    #: Location of console.profileEnd().
    location: debugger.Location
    profile: Profile
    #: Profile title passed as an argument to console.profile().
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileFinished:
        return cls(
            id_=str(json['id']),
            location=debugger.Location.from_json(json['location']),
            profile=Profile.from_json(json['profile']),
            title=str(json['title']) if 'title' in json else None
        )


@event_class('Profiler.consoleProfileStarted')
@dataclass
class ConsoleProfileStarted:
    '''
    Sent when new profile recording is started using console.profile() call.
    '''
    id_: str
    #: Location of console.profile().
    location: debugger.Location
    #: Profile title passed as an argument to console.profile().
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileStarted:
        return cls(
            id_=str(json['id']),
            location=debugger.Location.from_json(json['location']),
            title=str(json['title']) if 'title' in json else None
        )


@event_class('Profiler.preciseCoverageDeltaUpdate')
@dataclass
class PreciseCoverageDeltaUpdate:
    '''
    **EXPERIMENTAL**

    Reports coverage delta since the last poll (either from an event like this, or from
    ``takePreciseCoverage`` for the current isolate. May only be sent if precise code
    coverage has been started. This event can be trigged by the embedder to, for example,
    trigger collection of coverage data immediately at a certain point in time.
    '''
    #: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    timestamp: float
    #: Identifier for distinguishing coverage events.
    occasion: str
    #: Coverage data for the current isolate.
    result: typing.List[ScriptCoverage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreciseCoverageDeltaUpdate:
        return cls(
            timestamp=float(json['timestamp']),
            occasion=str(json['occasion']),
            result=[ScriptCoverage.from_json(i) for i in json['result']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\profiler.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Profiler
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime


@dataclass
class ProfileNode:
    '''
    Profile node. Holds callsite information, execution statistics and child nodes.
    '''
    #: Unique id of the node.
    id_: int

    #: Function location.
    call_frame: runtime.CallFrame

    #: Number of samples where this node was on top of the call stack.
    hit_count: typing.Optional[int] = None

    #: Child node ids.
    children: typing.Optional[typing.List[int]] = None

    #: The reason of being not optimized. The function may be deoptimized or marked as don't
    #: optimize.
    deopt_reason: typing.Optional[str] = None

    #: An array of source position ticks.
    position_ticks: typing.Optional[typing.List[PositionTickInfo]] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['callFrame'] = self.call_frame.to_json()
        if self.hit_count is not None:
            json['hitCount'] = self.hit_count
        if self.children is not None:
            json['children'] = [i for i in self.children]
        if self.deopt_reason is not None:
            json['deoptReason'] = self.deopt_reason
        if self.position_ticks is not None:
            json['positionTicks'] = [i.to_json() for i in self.position_ticks]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=int(json['id']),
            call_frame=runtime.CallFrame.from_json(json['callFrame']),
            hit_count=int(json['hitCount']) if 'hitCount' in json else None,
            children=[int(i) for i in json['children']] if 'children' in json else None,
            deopt_reason=str(json['deoptReason']) if 'deoptReason' in json else None,
            position_ticks=[PositionTickInfo.from_json(i) for i in json['positionTicks']] if 'positionTicks' in json else None,
        )


@dataclass
class Profile:
    '''
    Profile.
    '''
    #: The list of profile nodes. First item is the root node.
    nodes: typing.List[ProfileNode]

    #: Profiling start timestamp in microseconds.
    start_time: float

    #: Profiling end timestamp in microseconds.
    end_time: float

    #: Ids of samples top nodes.
    samples: typing.Optional[typing.List[int]] = None

    #: Time intervals between adjacent samples in microseconds. The first delta is relative to the
    #: profile startTime.
    time_deltas: typing.Optional[typing.List[int]] = None

    def to_json(self):
        json = dict()
        json['nodes'] = [i.to_json() for i in self.nodes]
        json['startTime'] = self.start_time
        json['endTime'] = self.end_time
        if self.samples is not None:
            json['samples'] = [i for i in self.samples]
        if self.time_deltas is not None:
            json['timeDeltas'] = [i for i in self.time_deltas]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            nodes=[ProfileNode.from_json(i) for i in json['nodes']],
            start_time=float(json['startTime']),
            end_time=float(json['endTime']),
            samples=[int(i) for i in json['samples']] if 'samples' in json else None,
            time_deltas=[int(i) for i in json['timeDeltas']] if 'timeDeltas' in json else None,
        )


@dataclass
class PositionTickInfo:
    '''
    Specifies a number of samples attributed to a certain source position.
    '''
    #: Source line number (1-based).
    line: int

    #: Number of samples attributed to the source line.
    ticks: int

    def to_json(self):
        json = dict()
        json['line'] = self.line
        json['ticks'] = self.ticks
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            line=int(json['line']),
            ticks=int(json['ticks']),
        )


@dataclass
class CoverageRange:
    '''
    Coverage data for a source range.
    '''
    #: JavaScript script source offset for the range start.
    start_offset: int

    #: JavaScript script source offset for the range end.
    end_offset: int

    #: Collected execution count of the source range.
    count: int

    def to_json(self):
        json = dict()
        json['startOffset'] = self.start_offset
        json['endOffset'] = self.end_offset
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            start_offset=int(json['startOffset']),
            end_offset=int(json['endOffset']),
            count=int(json['count']),
        )


@dataclass
class FunctionCoverage:
    '''
    Coverage data for a JavaScript function.
    '''
    #: JavaScript function name.
    function_name: str

    #: Source ranges inside the function with coverage data.
    ranges: typing.List[CoverageRange]

    #: Whether coverage data for this function has block granularity.
    is_block_coverage: bool

    def to_json(self):
        json = dict()
        json['functionName'] = self.function_name
        json['ranges'] = [i.to_json() for i in self.ranges]
        json['isBlockCoverage'] = self.is_block_coverage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            function_name=str(json['functionName']),
            ranges=[CoverageRange.from_json(i) for i in json['ranges']],
            is_block_coverage=bool(json['isBlockCoverage']),
        )


@dataclass
class ScriptCoverage:
    '''
    Coverage data for a JavaScript script.
    '''
    #: JavaScript script id.
    script_id: runtime.ScriptId

    #: JavaScript script name or url.
    url: str

    #: Functions contained in the script that has coverage data.
    functions: typing.List[FunctionCoverage]

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['functions'] = [i.to_json() for i in self.functions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_id=runtime.ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            functions=[FunctionCoverage.from_json(i) for i in json['functions']],
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.enable',
    }
    json = yield cmd_dict


def get_best_effort_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[ScriptCoverage]]:
    '''
    Collect coverage data for the current isolate. The coverage data may be incomplete due to
    garbage collection.

    :returns: Coverage data for the current isolate.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.getBestEffortCoverage',
    }
    json = yield cmd_dict
    return [ScriptCoverage.from_json(i) for i in json['result']]


def set_sampling_interval(
        interval: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes CPU profiler sampling interval. Must be called before CPU profiles recording started.

    :param interval: New sampling interval in microseconds.
    '''
    params: T_JSON_DICT = dict()
    params['interval'] = interval
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.setSamplingInterval',
        'params': params,
    }
    json = yield cmd_dict


def start() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.start',
    }
    json = yield cmd_dict


def start_precise_coverage(
        call_count: typing.Optional[bool] = None,
        detailed: typing.Optional[bool] = None,
        allow_triggered_updates: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Enable precise code coverage. Coverage data for JavaScript executed before enabling precise code
    coverage may be incomplete. Enabling prevents running optimized code and resets execution
    counters.

    :param call_count: *(Optional)* Collect accurate call counts beyond simple 'covered' or 'not covered'.
    :param detailed: *(Optional)* Collect block-based coverage.
    :param allow_triggered_updates: *(Optional)* Allow the backend to send updates on its own initiative
    :returns: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    '''
    params: T_JSON_DICT = dict()
    if call_count is not None:
        params['callCount'] = call_count
    if detailed is not None:
        params['detailed'] = detailed
    if allow_triggered_updates is not None:
        params['allowTriggeredUpdates'] = allow_triggered_updates
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.startPreciseCoverage',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['timestamp'])


def stop() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Profile]:
    '''


    :returns: Recorded profile.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.stop',
    }
    json = yield cmd_dict
    return Profile.from_json(json['profile'])


def stop_precise_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disable precise code coverage. Disabling releases unnecessary execution count records and allows
    executing optimized code.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.stopPreciseCoverage',
    }
    json = yield cmd_dict


def take_precise_coverage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[ScriptCoverage], float]]:
    '''
    Collect coverage data for the current isolate, and resets execution counters. Precise code
    coverage needs to have started.

    :returns: A tuple with the following items:

        0. **result** - Coverage data for the current isolate.
        1. **timestamp** - Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Profiler.takePreciseCoverage',
    }
    json = yield cmd_dict
    return (
        [ScriptCoverage.from_json(i) for i in json['result']],
        float(json['timestamp'])
    )


@event_class('Profiler.consoleProfileFinished')
@dataclass
class ConsoleProfileFinished:
    id_: str
    #: Location of console.profileEnd().
    location: debugger.Location
    profile: Profile
    #: Profile title passed as an argument to console.profile().
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileFinished:
        return cls(
            id_=str(json['id']),
            location=debugger.Location.from_json(json['location']),
            profile=Profile.from_json(json['profile']),
            title=str(json['title']) if 'title' in json else None
        )


@event_class('Profiler.consoleProfileStarted')
@dataclass
class ConsoleProfileStarted:
    '''
    Sent when new profile recording is started using console.profile() call.
    '''
    id_: str
    #: Location of console.profile().
    location: debugger.Location
    #: Profile title passed as an argument to console.profile().
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileStarted:
        return cls(
            id_=str(json['id']),
            location=debugger.Location.from_json(json['location']),
            title=str(json['title']) if 'title' in json else None
        )


@event_class('Profiler.preciseCoverageDeltaUpdate')
@dataclass
class PreciseCoverageDeltaUpdate:
    '''
    **EXPERIMENTAL**

    Reports coverage delta since the last poll (either from an event like this, or from
    ``takePreciseCoverage`` for the current isolate. May only be sent if precise code
    coverage has been started. This event can be trigged by the embedder to, for example,
    trigger collection of coverage data immediately at a certain point in time.
    '''
    #: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    timestamp: float
    #: Identifier for distinguishing coverage events.
    occasion: str
    #: Coverage data for the current isolate.
    result: typing.List[ScriptCoverage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreciseCoverageDeltaUpdate:
        return cls(
            timestamp=float(json['timestamp']),
            occasion=str(json['occasion']),
            result=[ScriptCoverage.from_json(i) for i in json['result']]
        )