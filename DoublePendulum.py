import numpy as np
from collections import deque
import sympy as sm
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation

print("Deriving equations of motion...")
t_sym = sm.symbols('t')
m_1, m_2, g_sym, L_1, L_2 = sm.symbols('m_1 m_2 g L_1 L_2', positive=True)
the1_fn, the2_fn = sm.symbols(r'\theta_1 \theta_2', cls=sm.Function)
the1_fn = the1_fn(t_sym)
the2_fn = the2_fn(t_sym)

x1 = L_1 * sm.sin(the1_fn)
y1 = -L_1 * sm.cos(the1_fn)
x2 = x1 + L_2 * sm.sin(the2_fn)
y2 = y1 - L_2 * sm.cos(the2_fn)

the1_d = sm.diff(the1_fn, t_sym)
the1_dd = sm.diff(the1_d, t_sym)
the2_d = sm.diff(the2_fn, t_sym)
the2_dd = sm.diff(the2_d, t_sym)

x1_d = sm.diff(x1, t_sym)
y1_d = sm.diff(y1, t_sym)
x2_d = sm.diff(x2, t_sym)
y2_d = sm.diff(y2, t_sym)

T_1 = sm.Rational(1, 2) * m_1 * (x1_d ** 2 + y1_d ** 2)
T_2 = sm.Rational(1, 2) * m_2 * (x2_d ** 2 + y2_d ** 2)
Lag = T_1 + T_2 - m_1 * g_sym * y1 - m_2 * g_sym * y2

LE1 = (sm.diff(sm.diff(Lag, the1_d), t_sym) - sm.diff(Lag, the1_fn)).simplify()
LE2 = (sm.diff(sm.diff(Lag, the2_d), t_sym) - sm.diff(Lag, the2_fn)).simplify()

sols = sm.solve([LE1, LE2], the1_dd, the2_dd)
args = (the1_fn, the2_fn, the1_d, the2_d, m_1, m_2, g_sym, L_1, L_2)
LEF1 = sm.lambdify(args, sols[the1_dd], modules='numpy', cse=True)
LEF2 = sm.lambdify(args, sols[the2_dd], modules='numpy', cse=True)
print("Ready.")

def derivs(a1, a1d, a2, a2d, m1, m2, g, l1, l2, b):
    a1dd = LEF1(a1, a2, a1d, a2d, m1, m2, g, l1, l2) - b * a1d
    a2dd = LEF2(a1, a2, a1d, a2d, m1, m2, g, l1, l2) - b * a2d
    return a1d, a1dd, a2d, a2dd

def rk4_step(state, dt, m1, m2, g, l1, l2, b):
    a1, a1d, a2, a2d = state
    k1a, k1b, k1c, k1d = derivs(a1, a1d, a2, a2d, m1, m2, g, l1, l2, b)
    k2a, k2b, k2c, k2d = derivs(a1 + 0.5 * dt * k1a, a1d + 0.5 * dt * k1b,
                                a2 + 0.5 * dt * k1c, a2d + 0.5 * dt * k1d,
                                m1, m2, g, l1, l2, b)
    k3a, k3b, k3c, k3d = derivs(a1 + 0.5 * dt * k2a, a1d + 0.5 * dt * k2b,
                                a2 + 0.5 * dt * k2c, a2d + 0.5 * dt * k2d,
                                m1, m2, g, l1, l2, b)
    k4a, k4b, k4c, k4d = derivs(a1 + dt * k3a, a1d + dt * k3b,
                                a2 + dt * k3c, a2d + dt * k3d,
                                m1, m2, g, l1, l2, b)
    a1n = a1 + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
    a1dn = a1d + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
    a2n = a2 + (dt / 6.0) * (k1c + 2 * k2c + 2 * k3c + k4c)
    a2dn = a2d + (dt / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
    return np.array([a1n, a1dn, a2n, a2dn])

COLORS = ['#3a7bd5', '#e05a5a', '#50c878', '#f5a623', '#b44fff', '#00e5cc', '#ff69b4', '#ffd700']
SLIDER_DEFAULTS = {'m1': 2.0, 'm2': 2.0, 'L1': 1.0, 'L2': 1.0, 'g': 9.81, 'b': 0.0}
INTERVAL = 16
SUB_STEPS = 4
TRAIL_LEN = 1500
MAX_N = 8

def run_setup():
    result = {}
    fig = plt.figure(figsize=(8, 9.5), facecolor='#0d0d0d')
    fig.canvas.manager.set_window_title('Double Pendulum — Setup')
    fig.suptitle('Double Pendulum — Setup', color='white', fontsize=14, y=0.97)

    ax_n = fig.add_axes([0.20, 0.91, 0.60, 0.028])
    sl_n = widgets.Slider(ax_n, 'Number of Pendulums', 1, MAX_N, valinit=2, valstep=1,
                          color='#3a7bd5', track_color='#2a2a2a')
    sl_n.label.set_color('#aaa')
    sl_n.valtext.set_color('#aaa')

    fig.text(0.08, 0.86, 'Duration (seconds or "inf"):', color='#aaa', fontsize=9)
    ax_dur = fig.add_axes([0.58, 0.845, 0.28, 0.03])
    tb_dur = widgets.TextBox(ax_dur, '', initial='60')
    tb_dur.text_disp.set_color('white')

    fig.text(0.30, 0.80, 'θ₁ initial (rad)', color='#888', fontsize=9, ha='center')
    fig.text(0.70, 0.80, 'θ₂ initial (rad)', color='#888', fontsize=9, ha='center')

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
    btn_start.label.set_color('white')
    btn_start.label.set_fontsize(11)

    def on_start(_):
        n = int(sl_n.val)
        ic_list = []
        for i in range(n):
            try:
                th1 = float(tb_th1_list[i].text)
            except Exception:
                th1 = 2.5
            try:
                th2 = float(tb_th2_list[i].text)
            except Exception:
                th2 = 2.5
            ic_list.append([th1, 0.0, th2, 0.0])

        try:
            raw = tb_dur.text.strip().lower()
            dur = np.inf if raw == 'inf' else float(raw)
        except Exception:
            dur = 60.0

        result['n'] = n
        result['ic_list'] = ic_list
        result['dur'] = dur
        plt.close(fig)

    btn_start.on_clicked(on_start)
    plt.show(block=True)
    return result

def run_simulation(n_pends, ic_list, duration, slider_vals=None):
    if slider_vals is None:
        slider_vals = dict(SLIDER_DEFAULTS)

    action_result = {'action': None}
    paused = [False]

    fig = plt.figure(figsize=(16, 8.8), facecolor='#0d0d0d')
    fig.canvas.manager.set_window_title('Double Pendulum')
    ax_anim = fig.add_axes([0.04, 0.20, 0.56, 0.74])
    ax_phase = fig.add_axes([0.64, 0.40, 0.32, 0.54])

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
    fps_txt = ax_anim.text(0.98, 0.97, '', transform=ax_anim.transAxes,
                           color='#666', fontsize=9, va='top', ha='right')

    slider_defs = [
        ('m1', 'Mass 1 (kg)', 0.05, 0.13, 0.1, 10.0),
        ('m2', 'Mass 2 (kg)', 0.05, 0.09, 0.1, 10.0),
        ('L1', 'Rod 1 (m)', 0.24, 0.13, 0.1, 3.0),
        ('L2', 'Rod 2 (m)', 0.24, 0.09, 0.1, 3.0),
        ('g', 'Gravity (m/s²)', 0.43, 0.13, 0.0, 30.0),
        ('b', 'Damping', 0.43, 0.09, 0.0, 5.0),
    ]

    sliders = {}
    textboxes = {}
    _suppress = {'flag': False}

    for key, label, left, bottom, vmin, vmax in slider_defs:
        ax_s = fig.add_axes([left, bottom, 0.11, 0.023])
        s = widgets.Slider(ax_s, label, vmin, vmax, valinit=slider_vals.get(key, SLIDER_DEFAULTS[key]),
                           color='#3a7bd5', track_color='#2a2a2a')
        s.label.set_color('#bbb')
        s.label.set_fontsize(9)
        s.valtext.set_visible(False)
        sliders[key] = s

        ax_t = fig.add_axes([left + 0.115, bottom, 0.045, 0.023])
        tb = widgets.TextBox(ax_t, '', initial=f"{s.val:.3f}", color='#1a1a1a', hovercolor='#222')
        tb.text_disp.set_color('#ddd')
        tb.text_disp.set_fontsize(9)
        textboxes[key] = tb

        def make_slider_updater(k):
            def updater(text):
                if _suppress['flag']:
                    return
                try:
                    val = float(text)
                except Exception:
                    return
                _suppress['flag'] = True
                try:
                    sliders[k].set_val(val)
                finally:
                    _suppress['flag'] = False
            return updater

        def make_text_updater(k):
            def updater(val):
                if _suppress['flag']:
                    return
                _suppress['flag'] = True
                try:
                    textboxes[k].set_val(f"{float(val):.3f}")
                finally:
                    _suppress['flag'] = False
            return updater

        tb.on_submit(make_slider_updater(key))
        s.on_changed(make_text_updater(key))

    btn_y = 0.06
    btn_h = 0.055
    btn_w = 0.085
    btn_gap = 0.012
    btn_x0 = 0.64

    ax_pause = fig.add_axes([btn_x0 + 0 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_restart = fig.add_axes([btn_x0 + 1 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_clear = fig.add_axes([btn_x0 + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_new = fig.add_axes([btn_x0 + 3 * (btn_w + btn_gap), btn_y, btn_w, btn_h])

    btn_pause = widgets.Button(ax_pause, 'Pause', color='#1e5f3a', hovercolor='#2a8b5a')
    btn_restart = widgets.Button(ax_restart, 'Restart', color='#1e3a5f', hovercolor='#2a5298')
    btn_clear = widgets.Button(ax_clear, 'Clear Trails', color='#3a3a1e', hovercolor='#6b6b2a')
    btn_new = widgets.Button(ax_new, 'New Session', color='#3a1e1e', hovercolor='#8b2020')

    for btn in (btn_pause, btn_restart, btn_clear, btn_new):
        btn.label.set_color('white')
        btn.label.set_fontsize(9.5)

    pivot_dot, = ax_anim.plot([0], [0], 'o', color='white', markersize=6.5, zorder=10)

    pend_artists = []

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

    state = np.array(ic_list, dtype=float).T.copy()

    trails = [{
        'tx1': deque(maxlen=TRAIL_LEN), 'ty1': deque(maxlen=TRAIL_LEN),
        'tx2': deque(maxlen=TRAIL_LEN), 'ty2': deque(maxlen=TRAIL_LEN),
        'ph1': deque(maxlen=TRAIL_LEN), 'ph2': deque(maxlen=TRAIL_LEN),
        'prev_ph1': None, 'prev_ph2': None,
    } for _ in range(n_pends)]

    sim_time = [0.0]
    last_marker = {'m1': None, 'm2': None}
    fps_state = {'last_t': None, 'frames': 0, 'avg': 0.0}

    def clear_trails(_=None):
        for tr in trails:
            tr['tx1'].clear(); tr['ty1'].clear()
            tr['tx2'].clear(); tr['ty2'].clear()
            tr['ph1'].clear(); tr['ph2'].clear()
            tr['prev_ph1'] = None
            tr['prev_ph2'] = None

    def restart_sim(_=None):
        nonlocal state
        clear_trails()
        state = np.array(ic_list, dtype=float).T.copy()
        sim_time[0] = 0.0

    def new_session(_=None):
        action_result['action'] = 'new_session'
        action_result['slider_vals'] = {k: s.val for k, s in sliders.items()}
        plt.close(fig)

    def toggle_pause(_):
        paused[0] = not paused[0]
        btn_pause.label.set_text('Resume' if paused[0] else 'Pause')
        fig.canvas.draw_idle()

    btn_pause.on_clicked(toggle_pause)
    btn_restart.on_clicked(restart_sim)
    btn_clear.on_clicked(clear_trails)
    btn_new.on_clicked(new_session)

    all_artists = [pivot_dot, timer_txt, fps_txt]
    for arts in pend_artists:
        all_artists.extend(arts.values())

    import time as _time

    def update(_frame):
        nonlocal state
        now = _time.perf_counter()
        if fps_state['last_t'] is not None:
            dt_real = now - fps_state['last_t']
            if dt_real > 0:
                inst = 1.0 / dt_real
                fps_state['avg'] = 0.9 * fps_state['avg'] + 0.1 * inst if fps_state['avg'] else inst
        fps_state['last_t'] = now
        fps_state['frames'] += 1
        if fps_state['frames'] % 15 == 0:
            fps_txt.set_text(f'{fps_state["avg"]:.0f} fps')

        if paused[0] or (duration != np.inf and sim_time[0] >= duration):
            return all_artists

        m1v = sliders['m1'].val
        m2v = sliders['m2'].val
        gv = sliders['g'].val
        l1v = sliders['L1'].val
        l2v = sliders['L2'].val
        bv = sliders['b'].val

        dt_frame = INTERVAL / 1000.0
        h = dt_frame / SUB_STEPS
        for _ in range(SUB_STEPS):
            state = rk4_step(state, h, m1v, m2v, gv, l1v, l2v, bv)
        sim_time[0] += dt_frame

        a1_arr = state[0]
        a2_arr = state[2]
        px1 = l1v * np.sin(a1_arr)
        py1 = -l1v * np.cos(a1_arr)
        px2 = px1 + l2v * np.sin(a2_arr)
        py2 = py1 - l2v * np.cos(a2_arr)

        ms1 = 3.5 * m1v + 4
        ms2 = 3.5 * m2v + 4
        if last_marker['m1'] != ms1:
            last_marker['m1'] = ms1
            for arts in pend_artists:
                arts['m1'].set_markersize(ms1)
        if last_marker['m2'] != ms2:
            last_marker['m2'] = ms2
            for arts in pend_artists:
                arts['m2'].set_markersize(ms2)

        for i in range(n_pends):
            tr = trails[i]
            arts = pend_artists[i]

            tr['tx1'].append(px1[i]); tr['ty1'].append(py1[i])
            tr['tx2'].append(px2[i]); tr['ty2'].append(py2[i])

            wp1 = ((a1_arr[i] + np.pi) % (2 * np.pi)) - np.pi
            wp2 = ((a2_arr[i] + np.pi) % (2 * np.pi)) - np.pi

            if tr['prev_ph1'] is not None and (
                abs(wp1 - tr['prev_ph1']) > np.pi or abs(wp2 - tr['prev_ph2']) > np.pi
            ):
                tr['ph1'].append(np.nan)
                tr['ph2'].append(np.nan)
            tr['ph1'].append(wp1)
            tr['ph2'].append(wp2)
            tr['prev_ph1'] = wp1
            tr['prev_ph2'] = wp2

            arts['r1'].set_data([0, px1[i]], [0, py1[i]])
            arts['r2'].set_data([px1[i], px2[i]], [py1[i], py2[i]])
            arts['m1'].set_data([px1[i]], [py1[i]])
            arts['m2'].set_data([px2[i]], [py2[i]])

            arts['tr1'].set_data(tr['tx1'], tr['ty1'])
            arts['tr2'].set_data(tr['tx2'], tr['ty2'])
            arts['ph'].set_data(tr['ph1'], tr['ph2'])
            arts['pd'].set_data([wp1], [wp2])

        if duration == np.inf:
            timer_txt.set_text(f'T = {sim_time[0]:.2f} s')
        else:
            timer_txt.set_text(f'T = {sim_time[0]:.2f} / {duration:.1f} s')

        return all_artists

    anim = FuncAnimation(fig, update, frames=None, interval=INTERVAL,
                         blit=True, cache_frame_data=False)
    plt.show(block=True)
    return action_result

def main():
    slider_vals = None
    while True:
        setup = run_setup()
        if not setup:
            break
        res = run_simulation(setup['n'], setup['ic_list'], setup['dur'], slider_vals)
        if res.get('action') == 'new_session':
            slider_vals = res.get('slider_vals')
            continue
        break

if __name__ == '__main__':
    main()
