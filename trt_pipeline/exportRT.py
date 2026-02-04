import tensorrt as trt

ONNX_MODEL_PATH = "yolov7-tiny.onnx"
ENGINE_PATH =     "yolov7-tiny.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # FP16
    if builder.platform_has_fast_fp16:
        print("✔ FP16 enabled")
        config.set_flag(trt.BuilderFlag.FP16)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # Optimization profile (MANDATORY)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",              # YOLOv7 input name
        min=(1, 3, 640, 640),
        opt=(8, 3, 640, 640),
        max=(16, 3, 640, 640),
    )
    config.add_optimization_profile(profile)

    print("⏳ Building TensorRT engine...")
    engine = builder.build_engine_with_config(network, config)

    if engine is None:
        print("❌ Engine build failed")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"✅ Engine saved: {engine_path}")
    return engine


if __name__ == "__main__":
    build_engine(ONNX_MODEL_PATH, ENGINE_PATH)