import numpy as np
from scipy.integrate import solve_ivp
from collections import deque
import sympy as sm
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation

# ===== SYMPY DERIVATION =====
print("Deriving equations of motion...")
t_sym = sm.symbols('t')
m_1, m_2, g_sym, L_1, L_2 = sm.symbols('m_1 m_2 g L_1 L_2', positive=True)
the1_fn, the2_fn = sm.symbols(r'\theta_1 \theta_2', cls=sm.Function)
the1_fn = the1_fn(t_sym);
the2_fn = the2_fn(t_sym)

x1 = L_1 * sm.sin(the1_fn);
y1 = -L_1 * sm.cos(the1_fn)
x2 = x1 + L_2 * sm.sin(the2_fn);
y2 = y1 - L_2 * sm.cos(the2_fn)

the1_d = sm.diff(the1_fn, t_sym);
the1_dd = sm.diff(the1_d, t_sym)
the2_d = sm.diff(the2_fn, t_sym);
the2_dd = sm.diff(the2_d, t_sym)

x1_d = sm.diff(x1, t_sym);
y1_d = sm.diff(y1, t_sym)
x2_d = sm.diff(x2, t_sym);
y2_d = sm.diff(y2, t_sym)

T_1 = sm.Rational(1, 2) * m_1 * (x1_d ** 2 + y1_d ** 2)
T_2 = sm.Rational(1, 2) * m_2 * (x2_d ** 2 + y2_d ** 2)
Lag = T_1 + T_2 - m_1 * g_sym * y1 - m_2 * g_sym * y2

LE1 = (sm.diff(sm.diff(Lag, the1_d), t_sym) - sm.diff(Lag, the1_fn)).simplify()
LE2 = (sm.diff(sm.diff(Lag, the2_d), t_sym) - sm.diff(Lag, the2_fn)).simplify()

sols = sm.solve([LE1, LE2], the1_dd, the2_dd)
LEF1 = sm.lambdify((the1_fn, the2_fn, the1_d, the2_d, t_sym, m_1, m_2, g_sym, L_1, L_2), sols[the1_dd])
LEF2 = sm.lambdify((the1_fn, the2_fn, the1_d, the2_d, t_sym, m_1, m_2, g_sym, L_1, L_2), sols[the2_dd])
print("Ready.")


# ===== ODE =====
def system_of_odes(t, y, m1, m2, g, l1, l2, b):
    a1, a1d, a2, a2d = y
    a1dd = LEF1(a1, a2, a1d, a2d, t, m1, m2, g, l1, l2) - b * a1d
    a2dd = LEF2(a1, a2, a1d, a2d, t, m1, m2, g, l1, l2) - b * a2d
    return [a1d, a1dd, a2d, a2dd]


# ===== CONSTANTS =====
COLORS = ['#3a7bd5', '#e05a5a', '#50c878', '#f5a623', '#b44fff', '#00e5cc', '#ff69b4', '#ffd700']
SLIDER_DEFAULTS = {'m1': 2.0, 'm2': 2.0, 'L1': 1.0, 'L2': 1.0, 'g': 9.81, 'b': 0.0}
INTERVAL = 16
MAX_N = 8


# ===================== SETUP SCREEN =====================
def run_setup():
    result = {}
    fig = plt.figure(figsize=(8, 9.5), facecolor='#0d0d0d')
    fig.suptitle('Double Pendulum — Setup', color='white', fontsize=14, y=0.97)

    ax_n = fig.add_axes([0.20, 0.91, 0.60, 0.028])
    sl_n = widgets.Slider(ax_n, 'Number of Pendulums', 1, MAX_N, valinit=2, valstep=1,
                          color='#3a7bd5', track_color='#2a2a2a')
    sl_n.label.set_color('#aaa');
    sl_n.valtext.set_color('#aaa')

    fig.text(0.08, 0.86, 'Duration (seconds or "inf"):', color='#aaa', fontsize=9)
    ax_dur = fig.add_axes([0.58, 0.845, 0.28, 0.03])
    tb_dur = widgets.TextBox(ax_dur, '', initial='60')
    tb_dur.text_disp.set_color('white')

    fig.text(0.30, 0.80, 'θ₁ initial (rad)', color='#666', fontsize=9, ha='center')
    fig.text(0.70, 0.80, 'θ₂ initial (rad)', color='#666', fontsize=9, ha='center')

    tb_th1_list, tb_th2_list = [], []
    for i in range(MAX_N):
        y_pos = 0.74 - i * 0.078
        c = COLORS[i]
        fig.text(0.05, y_pos + 0.008, f'P{i + 1}', color=c, fontsize=11, fontweight='bold')

        ax1 = fig.add_axes([0.18, y_pos, 0.32, 0.032])
        ax2 = fig.add_axes([0.58, y_pos, 0.32, 0.032])

        tb1 = widgets.TextBox(ax1, '', initial=str(round(2.5 + i * 0.07, 3)))
        tb2 = widgets.TextBox(ax2, '', initial=str(round(2.5 - i * 0.07, 3)))

        for tb in (tb1, tb2):
            tb.text_disp.set_color(c)
            tb.text_disp.set_fontsize(10)

        tb_th1_list.append(tb1)
        tb_th2_list.append(tb2)

    ax_start = fig.add_axes([0.30, 0.03, 0.40, 0.06])
    btn_start = widgets.Button(ax_start, '▶ Start Simulation', color='#1e3a5f', hovercolor='#2a5298')
    btn_start.label.set_color('white');
    btn_start.label.set_fontsize(11)

    def on_start(_):
        n = int(sl_n.val)
        ic_list = []
        for i in range(n):
            try:
                th1 = float(tb_th1_list[i].text)
            except:
                th1 = 2.5
            try:
                th2 = float(tb_th2_list[i].text)
            except:
                th2 = 2.5
            ic_list.append([th1, 0.0, th2, 0.0])

        try:
            raw = tb_dur.text.strip().lower()
            dur = np.inf if raw == 'inf' else float(raw)
        except:
            dur = 60.0

        result['n'] = n
        result['ic_list'] = ic_list
        result['dur'] = dur
        plt.close(fig)

    btn_start.on_clicked(on_start)
    plt.show(block=True)
    return result


# ===================== SIMULATION SCREEN =====================
def run_simulation(n_pends, ic_list, duration, slider_vals=None):
    if slider_vals is None:
        slider_vals = dict(SLIDER_DEFAULTS)

    action_result = {'action': None}
    paused = [False]

    fig = plt.figure(figsize=(16, 8.8), facecolor='#0d0d0d')
    ax_anim = fig.add_axes([0.04, 0.18, 0.58, 0.77])  # Bigger animation area
    ax_phase = fig.add_axes([0.65, 0.38, 0.32, 0.57])

    for ax in (ax_anim, ax_phase):
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

    ax_anim.set_xlim(-5, 5)
    ax_anim.set_ylim(-5, 5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(color='#1e1e1e', lw=1.0)
    ax_anim.set_title('Double Pendulum Simulator', color='#ccc', fontsize=13, pad=10)

    ax_phase.set_xlim(-np.pi, np.pi)
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_xlabel('θ₁ (rad)', color='#888', fontsize=9)
    ax_phase.set_ylabel('θ₂ (rad)', color='#888', fontsize=9)
    ax_phase.set_title('Phase Portrait θ₁ vs θ₂', color='#ccc', fontsize=10)

    timer_txt = ax_anim.text(0.02, 0.97, 'T = 0.00 s', transform=ax_anim.transAxes,
                             color='#aaa', fontsize=11, va='top', weight='bold')

    # ==================== SLIDERS + TEXTBOXES (more compact) ====================
    slider_defs = [
        ('m1', 'Mass 1 (kg)', 0.03, 0.13, 0.1, 10.0),
        ('m2', 'Mass 2 (kg)', 0.03, 0.09, 0.1, 10.0),
        ('L1', 'Rod 1 (m)', 0.20, 0.13, 0.1, 3.0),
        ('L2', 'Rod 2 (m)', 0.20, 0.09, 0.1, 3.0),
        ('g', 'Gravity (m/s²)', 0.37, 0.13, 0.0, 30.0),
        ('b', 'Damping', 0.37, 0.09, 0.0, 5.0),
    ]

    sliders = {}
    textboxes = {}

    for key, label, left, bottom, vmin, vmax in slider_defs:
        ax_s = fig.add_axes([left, bottom, 0.13, 0.023])
        s = widgets.Slider(ax_s, label, vmin, vmax, valinit=slider_vals.get(key, SLIDER_DEFAULTS[key]),
                           color='#3a7bd5', track_color='#2a2a2a')
        s.label.set_color('#bbb');
        s.label.set_fontsize(9)
        s.valtext.set_color('#bbb')
        sliders[key] = s

        ax_t = fig.add_axes([left + 0.135, bottom, 0.042, 0.023])
        tb = widgets.TextBox(ax_t, '', initial=f"{s.val:.3f}")
        tb.text_disp.set_color('#ddd')
        textboxes[key] = tb

        def make_slider_updater(k):
            def updater(text):
                try:
                    sliders[k].set_val(float(text))
                except:
                    pass

            return updater

        def make_text_updater(k):
            def updater(val):
                textboxes[k].set_val(f"{float(val):.3f}")

            return updater

        tb.on_submit(make_slider_updater(key))
        s.on_changed(make_text_updater(key))

    # ==================== BUTTONS (clean layout) ====================
    btn_y = 0.06
    btn_h = 0.055

    ax_pause = fig.add_axes([0.66, btn_y, 0.09, btn_h])
    ax_restart = fig.add_axes([0.76, btn_y, 0.09, btn_h])
    ax_clear = fig.add_axes([0.86, btn_y, 0.09, btn_h])
    ax_new = fig.add_axes([0.96, btn_y, 0.08, btn_h])

    btn_pause = widgets.Button(ax_pause, 'Pause', color='#1e5f3a', hovercolor='#2a8b5a')
    btn_restart = widgets.Button(ax_restart, 'Restart', color='#1e3a5f', hovercolor='#2a5298')
    btn_clear = widgets.Button(ax_clear, 'Clear Trails', color='#3a3a1e', hovercolor='#6b6b2a')
    btn_new = widgets.Button(ax_new, 'New Session', color='#3a1e1e', hovercolor='#8b2020')

    for btn in (btn_pause, btn_restart, btn_clear, btn_new):
        btn.label.set_color('white')
        btn.label.set_fontsize(9.5)

    def toggle_pause(_):
        paused[0] = not paused[0]
        btn_pause.label.set_text('Resume' if paused[0] else 'Pause')
        fig.canvas.draw_idle()

    btn_pause.on_clicked(toggle_pause)
    btn_restart.on_clicked(lambda _: restart_sim())
    btn_clear.on_clicked(lambda _: clear_trails())
    btn_new.on_clicked(lambda _: new_session())

    # ==================== ARTISTS & STATE ====================
    pivot_dot, = ax_anim.plot([0], [0], 'o', color='white', markersize=6.5, zorder=10)

    pend_artists = []
    pend_states = []

    for i in range(n_pends):
        c = COLORS[i]
        pend_artists.append({
            'r1': ax_anim.plot([], [], '-', color=c, lw=2.3, zorder=2)[0],
            'r2': ax_anim.plot([], [], '-', color=c, lw=1.7, zorder=2, alpha=0.78)[0],
            'm1': ax_anim.plot([], [], 'o', color=c, markersize=11, zorder=4)[0],
            'm2': ax_anim.plot([], [], 'o', color='white', markersize=7.5,
                               markeredgecolor=c, markeredgewidth=1.8, zorder=4)[0],
            'tr1': ax_anim.plot([], [], '-', color=c, alpha=0.45, lw=1.1, zorder=0)[0],
            'tr2': ax_anim.plot([], [], '-', color=c, alpha=0.32, lw=0.85, zorder=0)[0],
            'ph': ax_phase.plot([], [], '-', color=c, alpha=0.65, lw=0.95)[0],
            'pd': ax_phase.plot([], [], 'o', color=c, markersize=4.5)[0],
        })

        pend_states.append({
            'state': list(ic_list[i]),
            'tx1': [], 'ty1': [],
            'tx2': [], 'ty2': [],
            'ph1': [], 'ph2': [],
        })

    sim_time = [0.0]

    def clear_trails(_=None):
        for ps in pend_states:
            ps['tx1'].clear();
            ps['ty1'].clear()
            ps['tx2'].clear();
            ps['ty2'].clear()
            ps['ph1'].clear();
            ps['ph2'].clear()

    def restart_sim(_=None):
        clear_trails()
        for i, ic in enumerate(ic_list):
            pend_states[i]['state'] = list(ic)
        sim_time[0] = 0.0

    def new_session():
        action_result['action'] = 'new_session'
        action_result['slider_vals'] = {k: s.val for k, s in sliders.items()}
        plt.close(fig)

    all_artists = [pivot_dot, timer_txt]
    for arts in pend_artists:
        all_artists.extend(arts.values())

    def update(_frame):
        if paused[0] or (duration != np.inf and sim_time[0] >= duration):
            return all_artists

        p = {k: s.val for k, s in sliders.items()}
        dt = INTERVAL / 1000.0

        for i in range(n_pends):
            ps = pend_states[i]
            t0 = sim_time[0]

            res = solve_ivp(system_of_odes, [t0, t0 + dt], ps['state'],
                            method='RK45', args=(p['m1'], p['m2'], p['g'], p['L1'], p['L2'], p['b']),
                            rtol=1e-8, atol=1e-8)

            ps['state'] = res.y[:, -1].tolist()
            a1, _, a2, _ = ps['state']

            px1 = p['L1'] * np.sin(a1)
            py1 = -p['L1'] * np.cos(a1)
            px2 = px1 + p['L2'] * np.sin(a2)
            py2 = py1 - p['L2'] * np.cos(a2)

            ps['tx1'].append(px1);
            ps['ty1'].append(py1)
            ps['tx2'].append(px2);
            ps['ty2'].append(py2)
            ps['ph1'].append((a1 + np.pi) % (2 * np.pi) - np.pi)
            ps['ph2'].append((a2 + np.pi) % (2 * np.pi) - np.pi)

            arts = pend_artists[i]
            arts['r1'].set_data([0, px1], [0, py1])
            arts['r2'].set_data([px1, px2], [py1, py2])
            arts['m1'].set_data([px1], [py1])
            arts['m2'].set_data([px2], [py2])
            arts['m1'].set_markersize(3.5 * p['m1'] + 4)
            arts['m2'].set_markersize(3.5 * p['m2'] + 4)

            arts['tr1'].set_data(ps['tx1'], ps['ty1'])
            arts['tr2'].set_data(ps['tx2'], ps['ty2'])
            arts['ph'].set_data(ps['ph1'], ps['ph2'])
            if ps['ph1']:
                arts['pd'].set_data([ps['ph1'][-1]], [ps['ph2'][-1]])

        sim_time[0] += dt

        if duration == np.inf:
            timer_txt.set_text(f'T = {sim_time[0]:.2f} s')
        else:
            rem = max(0.0, duration - sim_time[0])
            timer_txt.set_text(f'T = {sim_time[0]:.2f} / {duration:.1f} s')

        return all_artists

    anim = FuncAnimation(fig, update, frames=None, interval=INTERVAL, blit=True, cache_frame_data=False)
    plt.show(block=True)
    return action_result


# ===================== MAIN LOOP =====================
slider_vals = None
while True:
    setup = run_setup()
    if not setup:
        break
    while True:
        res = run_simulation(setup['n'], setup['ic_list'], setup['dur'], slider_vals)
        if res.get('action') == 'new_session':
            slider_vals = res.get('slider_vals')
            break
        else:
            break
    if res.get('action') != 'new_session':
        break