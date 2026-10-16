import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

st.set_page_config(page_title="Numerical Integration", layout="centered")

st.title("Project3: Comparison of Numerical Integration Approximations")

# ---------------------------------------------------------------
# เลือกฟังก์ชัน (แก้ตรงนี้ได้ถ้าโจทย์ของคุณเป็นอีกแบบ)
# ---------------------------------------------------------------
FUNCTIONS = {
    "√(1 + eˣ)": lambda x: np.sqrt(1 + np.exp(x)),
    "1 + eˣ": lambda x: 1 + np.exp(x),
}
fname = st.selectbox("เลือกฟังก์ชัน f(x)", list(FUNCTIONS.keys()))
f = FUNCTIONS[fname]

st.latex(r"A = \int_a^b f(x)\,dx,\quad f(x) = " + fname.replace("√", r"\sqrt").replace("(", "{(").replace(")", ")}").replace("eˣ", "e^x"))


# ---------------------------------------------------------------
# วิธีเชิงตัวเลข
# ---------------------------------------------------------------
def trapezoidal(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (y.sum() - 0.5 * (y[0] + y[-1]))


def simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson's rule ต้องใช้ n เป็นเลขคู่")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def exact_value(f, a, b):
    val, _ = quad(f, a, b, epsabs=1e-13, epsrel=1e-13)
    return val


# ---------------------------------------------------------------
# 1) รับค่า a, b, n
# ---------------------------------------------------------------
st.header("1. กำหนดค่า a, b และ n")
c1, c2, c3 = st.columns(3)
a = c1.number_input("a (ขอบล่าง)", value=0.0, format="%.4f")
b = c2.number_input("b (ขอบบน)", value=2.0, format="%.4f")
n = c3.number_input("n (จำนวนช่วงย่อย)", min_value=2, value=8, step=2)
n = int(n)

if b <= a:
    st.error("ต้องกำหนดให้ b > a")
    st.stop()

A_true = exact_value(f, a, b)
A_trap = trapezoidal(f, a, b, n)
A_simp = simpson(f, a, b, n)

st.write(f"ค่าจริง A (คำนวณด้วย scipy.integrate.quad) = **{A_true:.10f}**")

# ---------------------------------------------------------------
# 2) แสดงพื้นที่ใต้กราฟ
# ---------------------------------------------------------------
st.header("2. พื้นที่ใต้กราฟ")
pad = (b - a) * 0.3
xs = np.linspace(a - pad, b + pad, 500)
xi = np.linspace(a, b, 300)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, f(xs), color="red", lw=1.2, label="f(x)")
ax.fill_between(xi, f(xi), color="dodgerblue", alpha=0.8, label=f"พื้นที่ [{a}, {b}]")
# วาดช่วงย่อยของ Trapezoidal
xn = np.linspace(a, b, n + 1)
ax.plot(xn, f(xn), "k.--", lw=0.6, ms=3, label=f"จุดแบ่ง n = {n}")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------------------------------------------------------------
# 3) ตาราง Error และกราฟ n กับ error
# ---------------------------------------------------------------
st.header("3. ตาราง Error ของแต่ละวิธี")
ns = [8, 16, 32, 64]
rows = []
for k in ns:
    rows.append(
        {
            "n": k,
            "Trapezoidal Error": abs(A_true - trapezoidal(f, a, b, k)),
            "Simpson Error": abs(A_true - simpson(f, a, b, k)),
        }
    )
df_err = pd.DataFrame(rows)
st.dataframe(
    df_err.style.format({"Trapezoidal Error": "{:.5E}", "Simpson Error": "{:.5E}"}),
    hide_index=True,
    use_container_width=True,
)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(df_err["n"], df_err["Trapezoidal Error"], "o-", label="Trapezoidal")
ax2.plot(df_err["n"], df_err["Simpson Error"], "s-", label="Simpson's")
ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.set_xticks(ns)
ax2.set_xticklabels(ns)
ax2.set_xlabel("n")
ax2.set_ylabel("Error = |A - Ã|")
ax2.set_title("n vs Error")
ax2.legend()
ax2.grid(alpha=0.3, which="both")
st.pyplot(fig2)

# ---------------------------------------------------------------
# 4) ตารางเปรียบเทียบเมื่อ n = 64
# ---------------------------------------------------------------
st.header("4. ตารางเปรียบเทียบ Error เมื่อ n = 64")
t64 = trapezoidal(f, a, b, 64)
s64 = simpson(f, a, b, 64)
df_cmp = pd.DataFrame(
    {
        "Method": ["Trapezoidal", "Simpson"],
        "A": [A_true, A_true],
        "Ã": [t64, s64],
        "Error = |A - Ã|": [abs(A_true - t64), abs(A_true - s64)],
    }
)
st.dataframe(
    df_cmp.style.format(
        {"A": "{:.6f}", "Ã": "{:.6f}", "Error = |A - Ã|": "{:.5E}"}
    ),
    hide_index=True,
    use_container_width=True,
)

# ---------------------------------------------------------------
# Output format ตามที่โจทย์กำหนด
# ---------------------------------------------------------------
st.subheader("Output (Format ตามโจทย์) ที่ n ที่เลือก")
lines = [
    f"{'Method':<14}{'A~1':>12}{'Error A1':>16}",
    "-" * 42,
    f"{'Trapezoidal':<14}{A_trap:>12.6f}{abs(A_true - A_trap):>16.5E}",
    f"{'Simpson':<14}{A_simp:>12.6f}{abs(A_true - A_simp):>16.5E}",
]
st.code("\n".join(lines), language="text")
