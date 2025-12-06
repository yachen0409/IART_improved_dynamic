import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import sys
import torch
from torchvision import transforms
from basicsr.utils import tensor2img


project_root = "./"
archs_path   = os.path.join(project_root, "archs")

for p in [project_root, archs_path]:
    if p not in sys.path:
        sys.path.insert(0, p)




to_remove = [m for m in sys.modules.keys() if m.startswith("archs")]
for m in to_remove:
    del sys.modules[m]

# Also reset the arch registry inside basicsr
from basicsr.utils.registry import ARCH_REGISTRY
ARCH_REGISTRY._obj_map.clear()



from archs.iart_arch import IART



MODEL_PATH  = os.path.join(project_root, "net_g_latest.pth")
SPYNET_PATH = os.path.join(project_root, "flownet/spynet_sintel_final-3d2a1287.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()

@st.cache_resource
def load_iart_model():
    print("Loading IART model...")
    model = IART(
        mid_channels=64,
        embed_dim=120,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        window_size=[3, 8, 8],
        num_frames=3,
        cpu_cache_length=100,
        is_low_res_input=True,
        spynet_path=SPYNET_PATH
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    if "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=False)
    model.eval()

    return model

model = load_iart_model()


st.set_page_config(
    page_title="Elevating Legacy Footage Quality through Adaptive Dynamic Video Super-Resolution",
    layout="wide"
)


st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], 
[data-testid="stApp"], [data-testid="stDecoration"], 
section.main, section.main > div:first-child,
#root, .block-container, .main, .css-18ni7ap, .css-1outpf7  {
    margin-top: 0 ;
    padding-top: 0 ;
}

/* Remove browser default body padding */
body {
    margin: 0 ;
    padding: 0 ;
}

/* Remove Streamlit header + toolbar */
header[data-testid="stHeader"], [data-testid="stToolbar"] {
    display: none !important;
}

/* Force layout to start at exact top of viewport */
html, body, #root {
    height: 100%;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
        header[data-testid="stHeader"] { display: none !important; }
        div[data-testid="stToolbar"] { display: none !important; }

        .full-header {
            background-color:#0e8c63;
            padding: 16px 16px 8px 16px;

            width:100vw;
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);

            text-align:center;
        }
    </style>

    <div class="full-header">
        <h1 style="color:white; margin:0; font-weight:700;">
            Elevating Legacy Footage Quality through Adaptive Dynamic Video Super-Resolution
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


def enhance_frame_iart(frame_bgr):
    print("Starting inference")

    import time
    t_start = time.time()

    # BGR → RGB
    print(" - Converting BGR → RGB")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # To tensor [3, H, W]
    print(" - Converting image to tensor shape [1,3,3,H,W]")
    t = to_tensor(frame_rgb)
    t = t.unsqueeze(0).unsqueeze(1)  # [1,1,3,H,W]
    t = t.repeat(1, 3, 1, 1, 1)      # [1,3,3,H,W]

    with torch.no_grad():
        print(" - Moving tensor to device:", device)
        t = t.to(device)

        print(" - Running model.forward() ...")
        run_start = time.time()
        out_seq = model(t)
        run_time = time.time() - run_start
        print(f"   Model forward pass done ({run_time:.2f} sec)")

        print(" - Extracting center frame")
        out = out_seq[:, 1]  # [1,3,H_up,W_up]

    print(" - Converting tensor → image")
    sr_rgb = tensor2img(out, rgb2bgr=False, min_max=(0, 1))
    sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

    total = time.time() - t_start
    print(f"Inference complete ({total:.2f} seconds)")


    return sr_bgr



st.markdown("""
<style>
.upload-wrapper div[data-testid="stFileUploader"],
.upload-wrapper div[data-testid="stMarkdownContainer"],
.upload-wrapper div[data-testid="stColumn"] {
    padding-left: 40px !important;
    padding-right: 40px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-wrapper">', unsafe_allow_html=True)

st.subheader("Upload Image or Video")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"]
)





st.markdown('</div>', unsafe_allow_html=True)



if uploaded_file:

    file_type = uploaded_file.type



    if "image" in file_type:

        # Read file bytes ONCE
        bytes_data = uploaded_file.getvalue()


        st.image(bytes_data, caption="Uploaded Image")

        # Decode for OpenCV
        img_array = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        st.markdown("---")
        st.subheader("Enhance Image")

        if st.button("Enhance Image"):
            with st.spinner("Enhancing image… please wait"):
                enhanced = enhance_frame_iart(img)


            enhanced = enhance_frame_iart(img)

            st.success("Image enhancement complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(img, use_container_width=True)
            with col2:
                st.markdown("**Enhanced Image**")
                st.image(enhanced, use_container_width=True)

            # Download button
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            cv2.imwrite(out_path, enhanced)
            with open(out_path, "rb") as f:
                st.download_button("Download Enhanced Image", f, "enhanced.png")

