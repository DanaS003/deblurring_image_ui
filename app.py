import torch
import os
import gradio as gr
import tempfile
from PIL import Image
from torchvision import transforms

# Import models from their respective files
from models.MIMOUNetPlus import MIMOUNetPlus
# Corrected: Import NADeblurPlus from models.NADeblurPlus
from models.NADeblurPlus import NADeblurPlus 

# Import utility functions from utils.py
# Corrected: Removed the circular import from utils.py itself
from utils import inference_transform, denormalize_img, calculate_metrics 

# --- Konfigurasi Model ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {DEVICE}")

# Path for MIMOUNetPlus model weights
MIMOUNET_MODEL_PATH = 'noaugmented_mimoplus.pth' # Ensure this file exists
MIMOUNET_NUM_RES = 20

# Path for NADeblurPlus model weights
NADEBLUR_MODEL_PATH = 'deblurdinat.pth' # Ensure this file exists. It was 'NADeblurPlus.pth' in one of your provided app.py but 'model_phase_2.pth' in the notebook. Using 'model_phase_2.pth' as it's consistent with notebook output.
NADEBLUR_DIM = 32 # From the notebook
NADEBLUR_NUM_BLOCKS = [4, 8, 12] # From the notebook
NADEBLUR_NUM_HEADS = [2, 4, 8] # From the notebook
NADEBLUR_KERNEL = 3 # From the notebook
NADEBLUR_FFN_EXPANSION_FACTOR = 1.5 # From the notebook
NADEBLUR_BIAS = False # From the notebook

# Load MIMOUNetPlus Model
try:
    mimounet_model = MIMOUNetPlus(num_res=MIMOUNET_NUM_RES).to(DEVICE)
    mimounet_model.load_state_dict(torch.load(MIMOUNET_MODEL_PATH, map_location=DEVICE))
    mimounet_model.eval()
    print(f"Model MIMOUNetPlus berhasil dimuat dari {MIMOUNET_MODEL_PATH}")
except Exception as e:
    print(f"Error saat memuat model MIMOUNetPlus: {e}")
    mimounet_model = None # Set to None if loading fails

# Load NADeblurPlus Model
try:
    nadeblur_model = NADeblurPlus(
        dim=NADEBLUR_DIM,
        num_blocks=NADEBLUR_NUM_BLOCKS,
        num_heads=NADEBLUR_NUM_HEADS,
        kernel=NADEBLUR_KERNEL,
        ffn_expansion_factor=NADEBLUR_FFN_EXPANSION_FACTOR,
        bias=NADEBLUR_BIAS
    ).to(DEVICE)
    # NADeblurPlus saves 'model_state_dict' inside the checkpoint
    # *** CHANGE THIS LINE ***
    nadeblur_checkpoint = torch.load(NADEBLUR_MODEL_PATH, map_location=DEVICE, weights_only=False) # Add weights_only=False
    nadeblur_model.load_state_dict(nadeblur_checkpoint['model_state_dict'])
    nadeblur_model.eval()
    print(f"Model NADeblurPlus berhasil dimuat dari {NADEBLUR_MODEL_PATH}")
except Exception as e:
    print(f"Error saat memuat model NADeblurPlus: {e}")
    nadeblur_model = None
# --- Fungsi Utama untuk Inferensi dan Pemilihan Model ---
def deblur_image_ui(input_blur_img: Image.Image, ground_truth_img: Image.Image = None, model_selector: float = 0):
    output_pil = None
    metrics_info = ""
    temp_filepath = None
    model_used = ""

    input_tensor = inference_transform(input_blur_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_selector == 0 and mimounet_model is not None:
            output = mimounet_model(input_tensor)[-1] # MIMOUNetPlus returns a list, take the last one
            model_used = "MIMO-UNetPlus"
        elif model_selector == 1 and nadeblur_model is not None:
            output = nadeblur_model(input_tensor) # NADeblurPlus returns a single tensor
            model_used = "NA-DeblurPlus"
        else:
            if model_selector == 0 and mimounet_model is None:
                metrics_info = "Error: MIMO-UNetPlus model tidak dapat dimuat."
            elif model_selector == 1 and nadeblur_model is None:
                metrics_info = "Error: NA-DeblurPlus model tidak dapat dimuat."
            return None, metrics_info, None # Return None for image and path if model fails

    output_tensor = denormalize_img(output.squeeze(0))
    output_pil = transforms.ToPILImage()(output_tensor.cpu())

    # --- Hitung metrik jika Ground Truth diberikan ---
    if ground_truth_img is not None:
        gt_tensor = inference_transform(ground_truth_img).to(DEVICE)
        gt_tensor = denormalize_img(gt_tensor)
        psnr, ssim = calculate_metrics(output_tensor, gt_tensor)
        metrics_info = f"Model Digunakan: **{model_used}** | **PSNR:** {psnr:.2f} dB | **SSIM:** {ssim:.4f}"
    else:
        metrics_info = f"Model Digunakan: **{model_used}** | Ground Truth tidak disediakan. PSNR dan SSIM tidak dihitung."

    # Save the PIL image to a temporary file for download
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        output_pil.save(tmp_file.name)
        temp_filepath = tmp_file.name

    return output_pil, metrics_info, temp_filepath


# --- Antarmuka Gradio ---
title = "✨ Aplikasi Deblur Gambar dengan MIMO-UNetPlus & NA-DeblurPlus ✨"
description = """
Selamat datang di Aplikasi Deblur Gambar! 
Unggah gambar buram Anda untuk mendapatkan versi yang lebih tajam menggunakan dua model berbeda. 
Untuk evaluasi kualitas (PSNR dan SSIM), Anda juga dapat menyediakan gambar tajam (Ground Truth) yang sesuai.

**Petunjuk:**
1.  **Unggah Gambar Buram:** Klik di area 'Gambar Buram (Input)' untuk mengunggah gambar yang ingin Anda deblur.
2.  **(Opsional) Unggah Ground Truth:** Jika Anda memiliki versi tajam dari gambar yang sama, unggah di 'Ground Truth (Opsional)' untuk menghitung metrik PSNR dan SSIM.
3.  **Pilih Model:** Gunakan slider 'Pilih Model Deblur' untuk beralih antara output dari MIMO-UNetPlus dan NA-DeblurPlus.
    * `0`: MIMO-UNetPlus
    * `1`: NA-DeblurPlus
4.  **Klik 'Deblur Gambar':** Tunggu beberapa saat hingga proses selesai.
5.  **Lihat Hasil:** Gambar yang sudah di-deblur akan ditampilkan di area 'Gambar Hasil Deblurring'.
6.  **Informasi Metrik:** PSNR dan SSIM akan muncul di bawah hasil deblurring jika Ground Truth disediakan.
7.  **Unduh Hasil:** Gunakan tombol unduh di bawah gambar hasil untuk menyimpannya.
"""

# Example images for Gradio
if not os.path.exists("example_images"):
    os.makedirs("example_images")
    # Create dummy blur_example.png
    Image.new('RGB', (256, 256), color = 'red').save("example_images/blur_example.png")
    # Create dummy gt_example.png
    Image.new('RGB', (256, 256), color = 'blue').save("example_images/gt_example.png")

examples = [
    ["example_images/blur_example.png", "example_images/gt_example.png", 0], # MIMO-UNetPlus example
    ["example_images/blur_example.png", "example_images/gt_example.png", 1]  # NA-DeblurPlus example
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)

    with gr.Tab("Deblurring Ajaib"):
        with gr.Row():
            with gr.Column():
                input_blur_img = gr.Image(type="pil", label="Gambar Buram (Input)", show_label=True)
                ground_truth_img = gr.Image(type="pil", label="Ground Truth (Opsional)", show_label=True)
                
                # Slider for model selection
                model_selector_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0, # Default to MIMO-UNetPlus
                    label="Pilih Model Deblur (0: MIMO-UNetPlus, 1: NA-DeblurPlus)",
                    interactive=True
                )
                
                with gr.Row():
                    deblur_btn = gr.Button("✨ Deblur Gambar ✨", variant="primary")
                    clear_btn = gr.ClearButton()

            with gr.Column():
                output_image = gr.Image(type="pil", label="Gambar Hasil Deblurring", show_label=True)
                download_button = gr.DownloadButton(value="Unduh Hasil", label="Unduh Gambar Deblur", visible=False)
                metrics_output = gr.Textbox(label="Informasi Metrik", interactive=False)
        
        deblur_btn.click(
            fn=deblur_image_ui,
            inputs=[input_blur_img, ground_truth_img, model_selector_slider],
            outputs=[output_image, metrics_output, download_button]
        )
        
        clear_btn.add([input_blur_img, ground_truth_img, output_image, metrics_output, download_button, model_selector_slider])
        clear_btn.click(lambda: gr.update(visible=False), inputs=None, outputs=[download_button])

    with gr.Tab("Tentang Aplikasi"):
        gr.Markdown("""
        ### Tentang Model
        Aplikasi ini menawarkan dua model canggih untuk deblurring gambar:

        * **MIMO-UNetPlus:** Arsitektur U-Net yang kuat, dirancang untuk merekonstruksi detail halus dan mengurangi kekaburan.
        * **NA-DeblurPlus:** Model yang lebih baru dengan arsitektur transformer dan Neighborhood Attention (NA) untuk penanganan spasial yang efisien, menawarkan kinerja deblurring yang kompetitif.

        ### Metrik Evaluasi
        * **PSNR (Peak Signal-to-Noise Ratio):** Mengukur rasio antara daya maksimum sinyal dan daya noise yang mengganggu fidelitasnya. Nilai PSNR yang lebih tinggi menunjukkan kualitas rekonstruksi yang lebih baik.
        * **SSIM (Structural Similarity Index Measure):** Mengevaluasi kesamaan antara dua gambar berdasarkan struktur, kontras, dan kecerahan. Nilai SSIM mendekati 1 menunjukkan kesamaan yang tinggi (kualitas lebih baik).

        ### Pengembang
        Dibuat oleh Kelompok 6 Mata Kuliah Deep Learning ITS menggunakan PyTorch dan Gradio.
        """)

# --- Jalankan Gradio ---
if __name__ == "__main__":
    print("Memulai aplikasi Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7861)