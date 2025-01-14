from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipeline

pipeline = OVStableDiffusionPipeline.from_pretrained(
    "rupeshs/sdxs-512-0.9-openvino",
    ov_config={"CACHE_DIR": ""},
)
torch.save(pipeline, 'model.pth')
