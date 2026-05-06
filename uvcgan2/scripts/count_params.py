import torch
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import sys
import os

# Add uvcgan2 to path just in case
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    print("=" * 50)
    print("Model Parameter Counts")
    print("=" * 50)
    
    # DeepLabV3+
    deeplab = smp.DeepLabV3Plus(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    print(f"DeepLabV3+ (ResNet-34): {count_parameters(deeplab):,}")

    # LinkNet
    linknet = smp.Linknet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    print(f"LinkNet (ResNet-34):    {count_parameters(linknet):,}")

    # U-Net
    unet = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    print(f"U-Net (ResNet-34):      {count_parameters(unet):,}")

    # SegFormer
    import warnings
    warnings.filterwarnings("ignore")
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b0',
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    print(f"SegFormer (mit-b0):     {count_parameters(segformer):,}")
    
    # seGANmentation Generator
    try:
        from uvcgan2.presets import GEN_PRESETS
        # The project uses vit-modnet based on README and train translation default
        gen_args = GEN_PRESETS['vit-modnet'].copy()
        
        # We might need to import construct_generator or similar
        # If we can't easily instantiate it, we will skip it
        from uvcgan2.models.generators import construct_generator
        gen_model = construct_generator(**gen_args)
        print(f"seGANmentation (vit-modnet generator): {count_parameters(gen_model):,}")
    except Exception as e:
        print(f"seGANmentation: Could not instantiate generator directly ({e})")
        
if __name__ == '__main__':
    main()
