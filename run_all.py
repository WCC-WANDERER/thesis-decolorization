import os

from methods.utils import load_image, save_gray
from methods.average import average_gray
from methods.luminance import luminance_gray
from methods.lab_l import lab_l_gray
from methods.grundland import grundland_decolor
from methods.color2gray_spcr import color2gray_spcr
from methods.corrc2g import corrc2g

from metrics.timing import timed
from metrics.rms import rms
from metrics.nrms import nrms
from metrics.c2g_ssim import c2g_ssim
from metrics.fsim import fsim
from metrics.grr import grr
from metrics.escore import escore
from metrics.ccpr_ccfr import ccpr, ccfr

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

METHODS = {
    "average": average_gray,
    "luminance": luminance_gray,
    "cielab": lab_l_gray,
    "decolorize": grundland_decolor,
    "corrc2g": corrc2g,
    "color2gray": color2gray_spcr
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    ensure_dir(OUTPUT_DIR)

    results = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"[WARN] Skipping unsupported file format: {filename}")
            continue

        img_path = os.path.join(INPUT_DIR, filename)
        img = load_image(img_path)

        for name, func in METHODS.items():
            # output paths
            out_dir = os.path.join(OUTPUT_DIR, name)
            ensure_dir(out_dir)

            # execute + time
            gray, t = timed(func, img)

            # save output
            out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".png")
            save_gray(gray, out_path)

            # metrics (use luminance as baseline for RMS)
            # lum_gray = luminance_gray(img)
            m_rms = rms(gray)
            # m_rms = rms(lum_gray, gray)
            # m_nrms = nrms(lum_gray, gray)
            m_nrms = nrms(img, gray)   # NRMS with original color image

            # Convert images to float 0..1 for FSIM and C2G-SSIM
            img_float = img.astype(float) / 255.0
            gray_float = gray.astype(float) / 255.0
            # img_float = (img / 255.0).astype(float)
            # gray_float = (gray / 255.0).astype(float)
            # gray_float = gray.astype(float) / 255.0

            #m_fsim = fsim(img_float.mean(axis=2), gray_float)   # FSIM uses grayscale
            m_grr = grr(img_float, gray_float)
            m_c2g = c2g_ssim(img_float, gray_float)

            m_ccpr = ccpr(img_float, gray_float, tau_gray=0.03, tau_lab=7.0)
            m_ccfr = ccfr(img_float, gray_float, tau_gray=0.03, tau_lab=7.0)
            #m_ccpr = ccpr(img_float, gray_float, tau=tau, color_tau=color_tau)
            #m_ccfr = ccfr(img_float, gray_float, tau=tau, color_tau=color_tau)
            m_escore = escore(m_ccpr, m_ccfr)
            # m_ccpr = ccpr(img_float, gray_float)
            # m_ccfr = ccfr(img_float, gray_float)
            # m_escore = escore(img_float, gray_float)

            results.append((filename, name, t, m_rms, m_nrms, m_grr, m_c2g, m_ccpr, m_ccfr, m_escore))

            print(f"{filename} - {name}: time={t:.4f}s  RMS={m_rms:.4f}  NRMS={m_nrms:.4f}  GRR={m_grr:.4f}  C2G-SSIM={m_c2g:.4f}  E-Score={m_escore:.4f}")

    # summary file
    with open("metrics_summary.csv", "w") as f:
        f.write("Image,Method,Time(sec),RMS,NRMS,GRR,C2G-SSIM,CCPR,CCFR,E-Score\n")
        for r in results:
            formatted_metrics = ["{:.4f}".format(float(x)) for x in r[2:]]
            f.write(",".join([r[0], r[1]] + formatted_metrics) + "\n")

if __name__ == "__main__":
    main()
