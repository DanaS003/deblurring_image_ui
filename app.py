import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
import os
import gradio as gr

# --- Definisi Model MIMOUNetPlus ---
# (Import kelas model sesuai kebutuhan, ini dipersingkat)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

# --- Konfigurasi Model ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {DEVICE}")

MODEL_PATH = 'noaugmented_mimoplus.pth' # Make sure this file is in the same directory or provide full path
MODEL_NUM_RES = 20

try:
    model = MIMOUNetPlus(num_res=MODEL_NUM_RES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model MIMOUNetPlus berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    # It's better to raise an error and stop if the model cannot be loaded
    # or provide a graceful fallback. For a Gradio app, exiting is fine if it's critical.
    raise RuntimeError(f"Failed to load model: {e}")

# --- Transformasi dan Denormalisasi ---
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def denormalize_img(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)

# --- PSNR dan SSIM ---
def calculate_metrics(deblurred_tensor, ground_truth_tensor):
    # Ensure sizes are compatible
    # Ground truth might be different size if user uploads arbitrary images
    if deblurred_tensor.shape[1:] != ground_truth_tensor.shape[1:]: # Check H, W
        # Resize ground truth to match deblurred output for metric calculation
        ground_truth_tensor = F.interpolate(
            ground_truth_tensor.unsqueeze(0),
            size=deblurred_tensor.shape[1:], # Target H, W
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    # Convert to numpy [H, W, C]
    deblur_np = deblurred_tensor.permute(1, 2, 0).cpu().numpy()
    gt_np = ground_truth_tensor.permute(1, 2, 0).cpu().numpy()

    # Convert to 0-255 range if they are floats 0-1, for metrics if necessary,
    # but skimage metrics can handle 0-1 if data_range is set.
    # We already clamped to 0-1, so data_range=1 is correct.
    psnr = psnr_metric(gt_np, deblur_np, data_range=1)
    ssim = ssim_metric(gt_np, deblur_np, data_range=1, channel_axis=-1)
    return psnr, ssim

# --- Fungsi Utama ---
def deblur_image_ui(input_blur_img: Image.Image, ground_truth_img: Image.Image = None):
    # Apply transformation to input blurred image
    input_tensor = inference_transform(input_blur_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)[-1] # Take the last output as the final deblurred image

    output_tensor = denormalize_img(output.squeeze(0))
    output_pil = transforms.ToPILImage()(output_tensor.cpu())

    metrics_info = ""
    # --- Hitung metrik jika Ground Truth diberikan ---
    if ground_truth_img is not None:
        gt_tensor = inference_transform(ground_truth_img).to(DEVICE)
        gt_tensor = denormalize_img(gt_tensor) # Denormalize GT as well for consistent range

        psnr, ssim = calculate_metrics(output_tensor, gt_tensor)
        metrics_info = f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"
    else:
        metrics_info = "Ground Truth tidak disediakan. PSNR dan SSIM tidak dihitung."

    return output_pil, metrics_info

# --- Antarmuka Gradio ---
title = "✨ Aplikasi Deblur Gambar dengan MIMO-UNetPlus ✨"
description = """
Selamat datang di Aplikasi Deblur Gambar! 
Unggah gambar buram Anda untuk mendapatkan versi yang lebih tajam. 
Untuk evaluasi kualitas (PSNR dan SSIM), Anda juga dapat menyediakan gambar tajam (Ground Truth) yang sesuai.

**Petunjuk:**
1.  **Unggah Gambar Buram:** Klik di area 'Gambar Buram (Input)' untuk mengunggah gambar yang ingin Anda deblur.
2.  **(Opsional) Unggah Ground Truth:** Jika Anda memiliki versi tajam dari gambar yang sama, unggah di 'Ground Truth (Opsional)' untuk menghitung metrik PSNR dan SSIM.
3.  **Klik 'Deblur Gambar':** Tunggu beberapa saat hingga proses selesai.
4.  **Lihat Hasil:** Gambar yang sudah di-deblur akan ditampilkan di area 'Gambar Hasil Deblurring'.
5.  **Informasi Metrik:** PSNR dan SSIM akan muncul di bawah hasil deblurring jika Ground Truth disediakan.
"""

# Example images for Gradio
# Ensure 'example_images' directory exists and contains 'blur_example.png' and 'gt_example.png'
# Create dummy files if they don't exist for demonstration purposes
if not os.path.exists("example_images"):
    os.makedirs("example_images")
    # Create dummy blur_example.png
    Image.new('RGB', (256, 256), color = 'red').save("example_images/blur_example.png")
    # Create dummy gt_example.png
    Image.new('RGB', (256, 256), color = 'blue').save("example_images/gt_example.png")


examples = [
    ["example_images/blur_example.png", "example_images/gt_example.png"]
]

with gr.Blocks(theme=gr.themes.Soft()) as demo: # Using gr.Blocks for more control over layout and gr.themes.Soft()
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)

    with gr.Tab("Deblurring"):
        with gr.Row():
            with gr.Column():
                input_blur_img = gr.Image(type="pil", label="Gambar Buram (Input)", show_label=True)
                ground_truth_img = gr.Image(type="pil", label="Ground Truth (Opsional)", show_label=True)
                # You can add more controls here if your model allowed tunable parameters
                # deblur_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Kekuatan Deblur", interactive=True)
                
                with gr.Row():
                    deblur_btn = gr.Button("✨ Deblur Gambar ✨", variant="primary")
                    clear_btn = gr.ClearButton()

            with gr.Column():
                output_image = gr.Image(type="pil", label="Gambar Hasil Deblurring", show_label=True)
                download_button = gr.DownloadButton(value="Unduh Hasil", label="Unduh Gambar Deblur", visible=False)
                metrics_output = gr.Textbox(label="Informasi Metrik", interactive=False)
        
        # Link button to function and update outputs
        deblur_btn.click(
            fn=deblur_image_ui,
            inputs=[input_blur_img, ground_truth_img], # Add other inputs if you create them
            outputs=[output_image, metrics_output]
        ).then( # Chain another function to enable download button after deblurring
            fn=lambda x: gr.update(value=x, visible=True),
            inputs=[output_image],
            outputs=[download_button]
        )

        clear_btn.add([input_blur_img, ground_truth_img, output_image, metrics_output, download_button])
        clear_btn.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=[download_button])

    with gr.Tab("Tentang Aplikasi"):
        gr.Markdown("""
        ### Tentang Model
        Aplikasi ini menggunakan model `MIMO-UNetPlus`, arsitektur canggih berbasis U-Net 
        yang dirancang khusus untuk tugas deblurring gambar. Model ini efektif dalam 
        merekonstruksi detail halus dan mengurangi kekaburan dari berbagai jenis gambar.

        ### Metrik Evaluasi
        * **PSNR (Peak Signal-to-Noise Ratio):** Mengukur rasio antara daya maksimum sinyal dan daya noise yang mengganggu fidelitasnya. Nilai PSNR yang lebih tinggi menunjukkan kualitas rekonstruksi yang lebih baik.
        * **SSIM (Structural Similarity Index Measure):** Mengevaluasi kesamaan antara dua gambar berdasarkan struktur, kontras, dan kecerahan. Nilai SSIM mendekati 1 menunjukkan kesamaan yang tinggi (kualitas lebih baik).

        ### Pengembang
        Dibuat oleh Kelompok 6 Mata Kuliah Deep Learning ITS 2024/2025 menggunakan PyTorch dan Gradio.
        """)

# --- Jalankan Gradio ---
if __name__ == "__main__":
    print("Memulai aplikasi Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7861) # Ganti port jika perlu