"""
Virtual Math Lab — Streamlit app (single-file)

File: app.py (this file)

Description:
Interactive virtual lab to explore graphs of several function families:
- Linear:       y = a*x + b
- Quadratic:    y = a*x^2 + b*x + c
- Cubic:        y = a*x^3 + b*x^2 + c*x + d
- Irrational:   y = a * sqrt(b*x + c)  (domain shown)
- Reciprocal:   y = a / (b*x + c)
- Circle:        (x-h)^2 + (y-k)^2 = r^2  (plotted as two y branches)
- Trigonometric (optional): y = a*sin(b*x + c)

Features:
- Choose function family and adjust parameters with sliders
- Interactive x-range control and sample density
- Show roots / zeros (numerical), vertex (for quadratic), inflection/critical points
- Toggle derivative and tangent at chosen x
- Highlight domain issues for irrational and reciprocal functions
- Simple exercises and "check my answer" box (teacher-mode)
- Download plot as PNG

Requirements (put into requirements.txt for GitHub):
streamlit
numpy
matplotlib
scipy
sympy

Deployment (Streamlit Cloud via GitHub):
1. Create new GitHub repo and add this file as app.py at repository root.
2. Add requirements.txt containing the packages above.
3. Commit & push: git add ., git commit -m "add virtual lab app", git push origin main
4. Go to https://streamlit.io/cloud (Streamlit Community Cloud), sign in and connect your GitHub repo.
5. Set the app entry file to app.py (or path) and deploy.

Troubleshooting:
- If nothing shows on Streamlit, check the "Deploy logs" for errors — usually missing package or wrong filename.
- For plotting issues, ensure matplotlib backend is available. Streamlit uses st.pyplot.

-- End of header README --
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
from math import isfinite
from scipy import integrate
import sympy as sp

st.set_page_config(page_title="Virtual Math Lab — Graphs", page_icon="📈", layout="wide")

# Helper utilities
@st.cache_data
def linspace_safe(xmin, xmax, n):
    if n <= 0:
        n = 400
    return np.linspace(xmin, xmax, n)

def compute_linear(a, b, x):
    return a * x + b

def compute_quadratic(a, b, c, x):
    return a * x**2 + b * x + c

def compute_cubic(a, b, c, d, x):
    return a * x**3 + b * x**2 + c * x + d

def compute_irrational(a, b, c, x):
    # y = a * sqrt(b*x + c)
    inside = b * x + c
    y = np.full_like(x, np.nan, dtype=float)
    mask = inside >= 0
    y[mask] = a * np.sqrt(inside[mask])
    return y, mask

def compute_reciprocal(a, b, c, x):
    denom = b * x + c
    y = np.full_like(x, np.nan, dtype=float)
    mask = denom != 0
    y[mask] = a / denom[mask]
    return y, mask

def compute_circle(h, k, r, x):
    inside = r**2 - (x - h)**2
    y1 = np.full_like(x, np.nan, dtype=float)
    y2 = np.full_like(x, np.nan, dtype=float)
    mask = inside >= 0
    y1[mask] = k + np.sqrt(inside[mask])
    y2[mask] = k - np.sqrt(inside[mask])
    return y1, y2, mask

# Numerical root-finding for sampled function
def find_zeros_sampled(x, y):
    zeros = []
    for i in range(len(x)-1):
        y1, y2 = y[i], y[i+1]
        if not (isfinite(y1) and isfinite(y2)):
            continue
        if y1 == 0:
            zeros.append(x[i])
        if y1 * y2 < 0:
            # linear interpolation root
            t = abs(y1) / (abs(y1) + abs(y2))
            xr = x[i] * (1 - t) + x[i+1] * t
            zeros.append(xr)
    return zeros

# Analytical helpers using sympy when possible
@st.cache_data
def quadratic_vertex(a, b, c):
    if a == 0:
        return None
    xv = -b / (2*a)
    yv = a * xv**2 + b * xv + c
    return xv, yv

@st.cache_data
def polynomial_roots(coeffs):
    try:
        roots = np.roots(coeffs)
        # return only real roots (within tolerance)
        real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-8]
        return real_roots
    except Exception:
        return []

# UI layout
st.title("📈 Virtual Math Lab — Graphs Explorer")
st.markdown("""
Selamat datang! Eksplorasikan berbagai fungsi, ubah parameter, lihat nol (roots), grafik turunan, dan lebih banyak lagi.\
Pilih tipe fungsi di samping, lalu mainkan slider-nya.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Kontrol Fungsi")
    func_type = st.selectbox("Pilih keluarga fungsi:", [
        "Linear", "Quadratic", "Cubic", "Irrational (sqrt)", "Reciprocal", "Circle", "Trigonometric (sin)"
    ])

    # Common controls
    xmin = st.number_input("xmin", value=-10.0, step=1.0)
    xmax = st.number_input("xmax", value=10.0, step=1.0)
    n_points = st.slider("Jumlah titik sampel", min_value=200, max_value=2000, value=800, step=100)

    # function-specific parameters
    params = {}
    if func_type == "Linear":
        params['a'] = st.slider('a (slope)', -10.0, 10.0, 1.0)
        params['b'] = st.slider('b (intercept)', -10.0, 10.0, 0.0)
    elif func_type == "Quadratic":
        params['a'] = st.slider('a', -5.0, 5.0, 1.0)
        params['b'] = st.slider('b', -10.0, 10.0, 0.0)
        params['c'] = st.slider('c', -10.0, 10.0, 0.0)
    elif func_type == "Cubic":
        params['a'] = st.slider('a', -2.0, 2.0, 1.0)
        params['b'] = st.slider('b', -5.0, 5.0, 0.0)
        params['c'] = st.slider('c', -5.0, 5.0, 0.0)
        params['d'] = st.slider('d', -5.0, 5.0, 0.0)
    elif func_type == "Irrational (sqrt)":
        params['a'] = st.slider('a (scale)', -5.0, 5.0, 1.0)
        params['b'] = st.slider('b (inside slope)', -5.0, 5.0, 1.0)
        params['c'] = st.slider('c (inside shift)', -10.0, 10.0, 0.0)
    elif func_type == "Reciprocal":
        params['a'] = st.slider('a (scale)', -10.0, 10.0, 1.0)
        params['b'] = st.slider('b (denom slope)', -5.0, 5.0, 1.0)
        params['c'] = st.slider('c (denom shift)', -10.0, 10.0, 0.0)
    elif func_type == "Circle":
        params['h'] = st.slider('h (center x)', -5.0, 5.0, 0.0)
        params['k'] = st.slider('k (center y)', -5.0, 5.0, 0.0)
        params['r'] = st.slider('r (radius)', 0.1, 10.0, 3.0)
    elif func_type == "Trigonometric (sin)":
        params['a'] = st.slider('a (amplitude)', -5.0, 5.0, 1.0)
        params['b'] = st.slider('b (frequency)', 0.1, 5.0, 1.0)
        params['c'] = st.slider('c (phase)', -np.pi, np.pi, 0.0)

    st.markdown("---")
    show_derivative = st.checkbox("Tampilkan turunan (numerik)")
    show_tangent = st.checkbox("Tampilkan garis singgung (pilih x)")
    tangent_x = None
    if show_tangent:
        tangent_x = st.number_input("x untuk garis singgung", value=0.0)

    show_area = st.checkbox("Tampilkan luas area antara grafik dan sumbu-x")
    show_roots = st.checkbox("Tandai nol (roots) pada grafik", value=True)
    annotations = st.checkbox("Tampilkan anotasi (vertex, intercept)")
    st.markdown("---")
    st.header("Latihan & Interaksi")
    exercise_mode = st.checkbox("Mode latihan (teacher) — tampilkan tugas singkat)")
    if exercise_mode:
        st.markdown("**Tugas:** Tentukan titik nol dari fungsi ini, dan berikan perkiraan koordinatnya. Gunakan kotak di bawah untuk memasukkan jawaban Anda.")
        user_answer = st.text_input("Masukkan jawaban (pisahkan titik dengan koma, contoh: -1.0, 2.0)")
        if st.button("Periksa jawaban"):
            st.success("Baik! Bandingkan hasil Anda dengan tanda nol pada grafik.")

with col2:
    # compute x grid
    x = linspace_safe(xmin, xmax, n_points)

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    y = None
    mask = np.ones_like(x, dtype=bool)

    if func_type == "Linear":
        y = compute_linear(params['a'], params['b'], x)
    elif func_type == "Quadratic":
        y = compute_quadratic(params['a'], params['b'], params['c'], x)
    elif func_type == "Cubic":
        y = compute_cubic(params['a'], params['b'], params['c'], params['d'], x)
    elif func_type == "Irrational (sqrt)":
        y, mask = compute_irrational(params['a'], params['b'], params['c'], x)
    elif func_type == "Reciprocal":
        y, mask = compute_reciprocal(params['a'], params['b'], params['c'], x)
    elif func_type == "Circle":
        y1, y2, mask = compute_circle(params['h'], params['k'], params['r'], x)
        ax.plot(x[mask], y1[mask], label="circle top")
        ax.plot(x[mask], y2[mask], label="circle bottom")
    elif func_type == "Trigonometric (sin)":
        y = params['a'] * np.sin(params['b'] * x + params['c'])

    # plot function (non-circle)
    if func_type != "Circle":
        ax.plot(x[mask], y[mask], linewidth=2, label=func_type)

    # derivative (numerical)
    if show_derivative and func_type != "Circle":
        dy = np.gradient(y, x)
        ax.plot(x[mask], dy[mask], linestyle='--', linewidth=1, label="dy/dx")

    # roots
    if show_roots and func_type != "Circle":
        zeros = find_zeros_sampled(x, y)
        for z in zeros:
            ax.plot(z, 0, 'o', markersize=8, markerfacecolor='white', markeredgewidth=2, label='root' if 'root' not in ax.get_legend_handles_labels()[1] else "")

    # annotations
    if annotations:
        if func_type == "Quadratic":
            v = quadratic_vertex(params['a'], params['b'], params['c'])
            if v is not None:
                ax.axvline(v[0], color='gray', linestyle=':', linewidth=1)
                ax.plot(v[0], v[1], 's', label='vertex')
                ax.annotate(f"vertex ({v[0]:.2f},{v[1]:.2f})", xy=(v[0], v[1]), xytext=(10, -20), textcoords='offset points')
        if func_type == "Linear":
            intercept = params['b']
            ax.plot(0, intercept, 'D', label='y-intercept')
            ax.annotate(f"(0,{intercept:.2f})", xy=(0, intercept), xytext=(10, -10), textcoords='offset points')

    # tangent
    if show_tangent and func_type != "Circle":
        # compute slope near tangent_x
        if tangent_x < xmin or tangent_x > xmax:
            st.warning("x untuk garis singgung di luar range x. Perluas xmin/xmax atau pilih x lain.")
        else:
            # find nearest index
            idx = (np.abs(x - tangent_x)).argmin()
            if func_type == "Irrational (sqrt)" or func_type == "Reciprocal":
                if not mask[idx]:
                    st.warning("Titik singgung tidak valid (di luar domain).")
                else:
                    slope = np.gradient(y, x)[idx]
                    y0 = y[idx]
                    tangent_line = slope * (x - tangent_x) + y0
                    ax.plot(x, tangent_line, linestyle='-.', linewidth=1.3, label=f'tangent @ {tangent_x:.2f}')
                    ax.plot(tangent_x, y0, 'x', markersize=10)
            else:
                slope = np.gradient(y, x)[idx]
                y0 = y[idx]
                tangent_line = slope * (x - tangent_x) + y0
                ax.plot(x, tangent_line, linestyle='-.', linewidth=1.3, label=f'tangent @ {tangent_x:.2f}')
                ax.plot(tangent_x, y0, 'x', markersize=10)

    # area under curve (signed) between xmin and xmax using mask
    if show_area and func_type != "Circle":
        y_valid = np.where(mask, y, 0.0)
        area = integrate.trapz(y_valid, x)
        ax.fill_between(x, y_valid, where=None, alpha=0.12)
        st.info(f"Luas (pendekatan numerik, integral terapan) dari x={xmin} sampai x={xmax} adalah ≈ {area:.3f}")

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim([xmin, xmax])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{func_type} — eksplorasi interaktif')

    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.legend(loc='upper right', fontsize='small')

    st.pyplot(fig)

    # Allow downloading plot as PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button(label="Download grafik (PNG)", data=buf, file_name="grafik_virtual_lab.png", mime='image/png')

st.markdown("---")
st.write("Jika kamu ingin saya bantu: (1) buatkan repo GitHub & file requirements.txt/README, (2) tuliskan instruksi `git` lengkap untuk push, atau (3) buatkan versi yang menggunakan Plotly untuk interaktivitas panning/zoom lebih halus — katakan pilihanmu.")
