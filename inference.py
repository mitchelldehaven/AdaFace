from AdaFace import net
import torch
import os
from AdaFace.face_alignment import align
import numpy as np
from pathlib import Path


PRETRAINED_MODEL_DIR = Path(__file__).resolve().parent / "pretrained"
ADAFACE_MODELS = {
    "ir_18": PRETRAINED_MODEL_DIR / "adaface_ir18_webface4m.ckpt",
    "ir_50": PRETRAINED_MODEL_DIR / "adaface_ir50_webface4m.ckpt",
    "ir_101": PRETRAINED_MODEL_DIR / "adaface_ir101_webface12m.ckpt"
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in ADAFACE_MODELS.keys(), f"Expected model architecture to be in {ADAFACE_MODELS.keys()}"
    model = net.build_model(architecture)
    statedict = torch.load(ADAFACE_MODELS[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image, device="cpu", dtype=torch.float32):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(brg_img.transpose(2,0,1)).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_50')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = 'face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)
    

