#!/usr/bin/env python3
"""
Arterial Tree - Paper Figure 3 Style
Structure (top to bottom):
  1. HEAD: brain fan symmetric L/R, CoW ring, carotids up into it
  2. TORSO: heart red dot top, aorta straight down, organ stubs L/R,
            arms curve down sides
  3. LEGS: iliac split, each leg straight down, femoral/tibial at knee
Node style: white open circle = junction, gray filled = terminal, red = heart
No background boxes. Vessels follow body contour.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PaperStyleTree:
    def __init__(self, model_name, results_base_path):
        self.model_name = model_name
        self.results_path = Path(results_base_path) / model_name / "arterial"
        self.M3S_TO_MLMIN = 60.0 * 1e6

    def load_mean_flow(self, vessel_id):
        file_path = self.results_path / f"{vessel_id}.txt"
        if not file_path.exists():
            return 0.0
        try:
            data = np.loadtxt(file_path, delimiter=',')
            if data.ndim == 1 or data.shape[1] < 6:
                return 0.0
            return np.mean(data[:, 5]) * self.M3S_TO_MLMIN
        except:
            return 0.0

    def vessel_color(self, flow):
        """Blue if reversed, red otherwise. Gray if minimal."""
        if flow < -0.01:
            return 'blue', 2.2
        elif abs(flow) <= 0.01:
            return '#bbbbbb', 0.9
        else:
            return 'red', 1.6

    def line(self, ax, pts, vid):
        """Draw vessel as polyline, color by flow"""
        flow = self.load_mean_flow(vid)
        color, lw = self.vessel_color(flow)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle='round',
                zorder=2, alpha=0.85)
        return flow

    def junc(self, ax, x, y):
        """White open circle junction node (paper style)"""
        ax.plot(x, y, 'o', color='white', markersize=6,
                markeredgecolor='black', markeredgewidth=0.9, zorder=4)

    def term(self, ax, x, y):
        """Gray filled terminal node"""
        ax.plot(x, y, 'o', color='#777777', markersize=4.5, zorder=4)

    def create(self, save_path=None):
        fig, ax = plt.subplots(figsize=(9, 17))
        # Total height: head 0-12, torso 13-26, legs 27-38
        # We invert so head is at TOP visually: use y directly, higher = top
        ax.set_xlim(-9, 9)
        ax.set_ylim(-1, 39)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.suptitle(f'Arterial Network: {self.model_name}',
                     fontsize=14, fontweight='bold', y=0.99)

        # ==============================================================
        # SECTION 1: HEAD (y = 26 .. 38)   -- TOP of figure
        # ==============================================================
        # Layout: carotids come up from torso at x=-1.2 and x=+1.2
        # They enter CoW ring, brain vessels fan outward symmetrically

        # Carotid entry points (bottom of head section)
        r_cca_top = (1.2, 27.0)   # R common carotid top
        l_cca_top = (-1.2, 27.0)  # L common carotid top

        # ICA entry into CoW
        r_ica = (1.0, 28.5)
        l_ica = (-1.0, 28.5)

        # External carotid branches (go outward)
        r_eca = (2.2, 27.5)
        l_eca = (-2.2, 27.5)

        # CoW ring center ~ y=29.5
        # PCoA junctions (where ICA meets posterior)
        r_pcoa_jxn = (0.8, 29.5)   # n45
        l_pcoa_jxn = (-0.8, 29.5)  # n38

        # M1 segment (ICA distal, before MCA/ACA split)
        r_m1 = (0.7, 30.5)   # n43
        l_m1 = (-0.7, 30.5)  # n40

        # Basilar top (PCA bifurcation)
        ba_bif = (0, 29.0)    # n49
        ba_mid = (0, 28.2)    # n50
        vert_conf = (0, 27.2) # n31

        # PCA P1 endpoints
        r_p1 = (0.5, 29.8)   # n47
        l_p1 = (-0.5, 29.8)  # n48

        # Brain terminal fans
        # R side: MCA out to right, ACA up-right, PCA right
        r_mca_end = (4.0, 31.5)
        r_mca_m2a = (4.5, 32.5)
        r_mca_m2b = (3.5, 33.0)
        r_aca_a1  = (1.5, 31.5)  # n42
        r_aca_a2  = (2.0, 33.0)
        r_pca_p2  = (2.5, 30.5)

        # L side: mirror
        l_mca_end = (-4.0, 31.5)
        l_mca_m2a = (-4.5, 32.5)
        l_mca_m2b = (-3.5, 33.0)
        l_aca_a1  = (-1.5, 31.5)  # n41
        l_aca_a2  = (-2.0, 33.0)
        l_pca_p2  = (-2.5, 30.5)

        # ACoA connects the two ACA A1 nodes
        acoa_mid = (0, 32.0)

        # --- Draw head vessels ---

        # R Common carotid top segment + external carotid branch
        self.line(ax, [r_cca_top, r_ica], 'A12')          # R ICA
        self.line(ax, [r_cca_top, r_eca], 'A13')          # R ext carotid 1
        self.junc(ax, *r_cca_top)
        self.junc(ax, *r_ica)

        # L Common carotid
        self.line(ax, [l_cca_top, l_ica], 'A16')
        self.line(ax, [l_cca_top, l_eca], 'A17')
        self.junc(ax, *l_cca_top)
        self.junc(ax, *l_ica)

        # External carotid sub-branches (stubs fanning out)
        r_eca2 = (3.0, 27.0)
        r_eca3 = (3.5, 28.0)
        r_eca4 = (4.0, 27.5)
        self.line(ax, [r_eca, r_eca2], 'A83')
        self.junc(ax, *r_eca)
        self.junc(ax, *r_eca2)
        self.line(ax, [r_eca2, r_eca3], 'A87')
        self.junc(ax, *r_eca3)
        self.line(ax, [r_eca3, r_eca4], 'A91')
        self.term(ax, *r_eca4)
        self.line(ax, [r_eca3, (4.5, 27.2)], 'A92')
        self.term(ax, 4.5, 27.2)
        self.line(ax, [r_eca2, (3.8, 26.2)], 'A88')
        self.term(ax, 3.8, 26.2)
        self.line(ax, [r_eca, (2.8, 26.5)], 'A84')
        self.term(ax, 2.8, 26.5)

        # L external carotid (mirror)
        l_eca2 = (-3.0, 27.0)
        l_eca3 = (-3.5, 28.0)
        l_eca4 = (-4.0, 27.5)
        self.line(ax, [l_eca, l_eca2], 'A85')
        self.junc(ax, *l_eca)
        self.junc(ax, *l_eca2)
        self.line(ax, [l_eca2, l_eca3], 'A89')
        self.junc(ax, *l_eca3)
        self.line(ax, [l_eca3, l_eca4], 'A93')
        self.term(ax, *l_eca4)
        self.line(ax, [l_eca3, (-4.5, 27.2)], 'A94')
        self.term(ax, -4.5, 27.2)
        self.line(ax, [l_eca2, (-3.8, 26.2)], 'A90')
        self.term(ax, -3.8, 26.2)
        self.line(ax, [l_eca, (-2.8, 26.5)], 'A86')
        self.term(ax, -2.8, 26.5)

        # ICA sinus segment -> PCoA junction -> distal -> M1
        self.line(ax, [r_ica, r_pcoa_jxn], 'A79')
        self.junc(ax, *r_pcoa_jxn)
        self.line(ax, [r_pcoa_jxn, (0.75, 30.0)], 'A66')
        self.junc(ax, 0.75, 30.0)
        self.line(ax, [(0.75, 30.0), r_m1], 'A101')
        self.junc(ax, *r_m1)

        self.line(ax, [l_ica, l_pcoa_jxn], 'A81')
        self.junc(ax, *l_pcoa_jxn)
        self.line(ax, [l_pcoa_jxn, (-0.75, 30.0)], 'A67')
        self.junc(ax, -0.75, 30.0)
        self.line(ax, [(-0.75, 30.0), l_m1], 'A103')
        self.junc(ax, *l_m1)

        # Ophthalmic (short stubs from ICA)
        self.line(ax, [r_ica, (1.7, 28.8)], 'A80')
        self.term(ax, 1.7, 28.8)
        self.line(ax, [l_ica, (-1.7, 28.8)], 'A82')
        self.term(ax, -1.7, 28.8)

        # Vertebrals converge to confluence
        # (drawn as coming up from torso, meeting at vert_conf)
        self.junc(ax, *vert_conf)

        # Basilar
        self.line(ax, [vert_conf, ba_mid], 'A56')
        self.junc(ax, *ba_mid)
        # Superior cerebellar stubs
        self.line(ax, [ba_mid, (0.6, 28.0)], 'A57')
        self.term(ax, 0.6, 28.0)
        self.line(ax, [ba_mid, (-0.6, 28.0)], 'A58')
        self.term(ax, -0.6, 28.0)

        self.line(ax, [ba_mid, ba_bif], 'A59')
        self.junc(ax, *ba_bif)

        # PCA P1 --- KEY: R-PCA may be REVERSED (blue)
        r_pca_flow = self.line(ax, [ba_bif, r_p1], 'A60')
        self.junc(ax, *r_p1)
        self.line(ax, [ba_bif, l_p1], 'A61')
        self.junc(ax, *l_p1)

        # PCoA connections (P1 end -> ICA sinus junction)
        self.line(ax, [r_p1, r_pcoa_jxn], 'A62')
        self.line(ax, [l_p1, l_pcoa_jxn], 'A63')

        # PCA P2 (terminals)
        self.line(ax, [r_p1, r_pca_p2], 'A64')
        self.term(ax, *r_pca_p2)
        self.line(ax, [l_p1, l_pca_p2], 'A65')
        self.term(ax, *l_pca_p2)

        # MCA from M1
        self.line(ax, [r_m1, r_mca_end], 'A70')
        self.junc(ax, *r_mca_end)
        self.line(ax, [r_mca_end, r_mca_m2a], 'A71')
        self.term(ax, *r_mca_m2a)
        self.line(ax, [r_mca_end, r_mca_m2b], 'A72')
        self.term(ax, *r_mca_m2b)

        self.line(ax, [l_m1, l_mca_end], 'A73')
        self.junc(ax, *l_mca_end)
        self.line(ax, [l_mca_end, l_mca_m2a], 'A74')
        self.term(ax, *l_mca_m2a)
        self.line(ax, [l_mca_end, l_mca_m2b], 'A75')
        self.term(ax, *l_mca_m2b)

        # ACA from M1
        self.line(ax, [r_m1, r_aca_a1], 'A68')
        self.junc(ax, *r_aca_a1)
        self.line(ax, [r_aca_a1, r_aca_a2], 'A76')
        self.term(ax, *r_aca_a2)

        self.line(ax, [l_m1, l_aca_a1], 'A69')
        self.junc(ax, *l_aca_a1)
        self.line(ax, [l_aca_a1, l_aca_a2], 'A78')
        self.term(ax, *l_aca_a2)

        # ACoA connects L-ACA-A1 to R-ACA-A1
        self.line(ax, [l_aca_a1, r_aca_a1], 'A77')

        # Anterior choroidal (stubs)
        self.line(ax, [(0.75, 30.0), (1.3, 30.8)], 'A100')
        self.term(ax, 1.3, 30.8)
        self.line(ax, [(-0.75, 30.0), (-1.3, 30.8)], 'A102')
        self.term(ax, -1.3, 30.8)

        # --- Flow reversal annotation ---
        if r_pca_flow < 0:
            ax.annotate('FLOW REVERSAL\n(Fetal variant)',
                        xy=(0.5, 29.8), xytext=(3.0, 29.0),
                        fontsize=7, fontweight='bold', color='blue',
                        bbox=dict(boxstyle='round', fc='cyan', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                        zorder=6)

        # ==============================================================
        # SECTION 2: TORSO  (y = 13 .. 26)
        # ==============================================================
        # Heart at top-center, aorta descends, arms curve down sides

        heart = (0, 24.5)
        ax.plot(*heart, 'o', color='red', markersize=9, zorder=5)

        # Ascending aorta: heart up to arch
        n1 = (0, 25.2)
        n2 = (0, 25.8)  # aortic arch
        self.line(ax, [heart, n1], 'A1')
        self.junc(ax, *n1)
        # Coronaries (short stubs)
        self.line(ax, [n1, (-0.6, 24.8)], 'A96')
        self.term(ax, -0.6, 24.8)
        self.line(ax, [n1, (-0.8, 25.0)], 'A97')
        self.junc(ax, -0.8, 25.0)
        self.line(ax, [(-0.8, 25.0), (-1.2, 24.6)], 'A98')
        self.term(ax, -1.2, 24.6)
        self.line(ax, [(-0.8, 25.0), (-1.0, 25.2)], 'A99')
        self.term(ax, -1.0, 25.2)

        self.line(ax, [n1, n2], 'A95')
        self.junc(ax, *n2)

        # Arch branches
        # Brachiocephalic -> R subclavian + R common carotid
        n6 = (1.5, 25.5)   # brachiocephalic end
        self.line(ax, [n2, n6], 'A3')
        self.junc(ax, *n6)

        # Arch A -> L side
        n3 = (-1.0, 25.8)
        self.line(ax, [n2, n3], 'A2')
        self.junc(ax, *n3)

        # Arch B
        n4 = (-2.0, 25.5)
        self.line(ax, [n3, n4], 'A14')
        self.junc(ax, *n4)

        # R Common carotid goes UP into head
        r_cca_bot = (1.2, 26.2)
        self.line(ax, [n6, r_cca_bot], 'A5')
        self.junc(ax, *r_cca_bot)
        # connect to head section
        ax.plot([r_cca_bot[0], r_cca_top[0]], [r_cca_bot[1], r_cca_top[1]],
                color='red', linewidth=1.6, alpha=0.85, zorder=2)

        # L Common carotid goes UP into head
        l_cca_bot = (-1.2, 26.2)
        self.line(ax, [n3, l_cca_bot], 'A15')
        self.junc(ax, *l_cca_bot)
        ax.plot([l_cca_bot[0], l_cca_top[0]], [l_cca_bot[1], l_cca_top[1]],
                color='red', linewidth=1.6, alpha=0.85, zorder=2)

        # Vertebrals go UP into head (from subclavian ends)
        # R vertebral: n7 -> vert_conf
        n7 = (2.5, 24.8)   # R subclavian end
        self.line(ax, [n6, n7], 'A4')
        self.junc(ax, *n7)
        # vertebral curves up to confluence
        self.line(ax, [n7, (1.5, 26.0), vert_conf], 'A6')

        # L vertebral: n10 -> vert_conf
        n10 = (-2.8, 24.8)  # L subclavian end
        self.line(ax, [n4, n10], 'A19')
        self.junc(ax, *n10)
        self.line(ax, [n10, (-1.5, 26.0), vert_conf], 'A20')

        # R ARM: curves down right side of torso
        # n7 -> n8 (brachial) -> n9 (ulnar split)
        n8 = (4.0, 23.0)
        self.line(ax, [n7, (3.2, 24.2), n8], 'A7')
        self.junc(ax, *n8)

        n9 = (4.5, 21.5)
        self.line(ax, [n8, n9], 'A9')
        self.junc(ax, *n9)

        # Radial (terminal)
        self.line(ax, [n8, (4.8, 22.0)], 'A8')
        self.term(ax, 4.8, 22.0)
        # Interosseous + Ulnar B (terminals from n9)
        self.line(ax, [n9, (5.0, 20.5)], 'A10')
        self.term(ax, 5.0, 20.5)
        self.line(ax, [n9, (4.2, 20.2)], 'A11')
        self.term(ax, 4.2, 20.2)

        # L ARM: mirror on left side
        n11 = (-4.0, 23.0)
        self.line(ax, [n10, (-3.2, 24.2), n11], 'A21')
        self.junc(ax, *n11)

        n12 = (-4.5, 21.5)
        self.line(ax, [n11, n12], 'A23')
        self.junc(ax, *n12)

        self.line(ax, [n11, (-4.8, 22.0)], 'A22')
        self.term(ax, -4.8, 22.0)
        self.line(ax, [n12, (-5.0, 20.5)], 'A24')
        self.term(ax, -5.0, 20.5)
        self.line(ax, [n12, (-4.2, 20.2)], 'A25')
        self.term(ax, -4.2, 20.2)

        # DESCENDING AORTA: straight down center from n4
        # Thoracic aorta A
        n51 = (0, 23.5)
        self.line(ax, [n4, (-1.5, 24.8), (-0.5, 24.0), n51], 'A18')
        self.junc(ax, *n51)
        # Intercostals stub
        self.line(ax, [n51, (0.8, 23.3)], 'A26')
        self.term(ax, 0.8, 23.3)

        # Thoracic aorta B
        n52 = (0, 22.0)
        self.line(ax, [n51, n52], 'A27')
        self.junc(ax, *n52)

        # Celiac
        self.line(ax, [n52, (-0.9, 21.8)], 'A29')
        self.junc(ax, -0.9, 21.8)
        self.line(ax, [(-0.9, 21.8), (-1.5, 21.5)], 'A30')
        self.junc(ax, -1.5, 21.5)
        self.line(ax, [(-1.5, 21.5), (-2.0, 21.2)], 'A31')
        self.term(ax, -2.0, 21.2)
        self.line(ax, [(-1.5, 21.5), (-1.8, 20.8)], 'A33')
        self.term(ax, -1.8, 20.8)
        self.line(ax, [(-0.9, 21.8), (-1.2, 21.0)], 'A32')
        self.term(ax, -1.2, 21.0)

        # Abdominal aorta A
        n22 = (0, 20.5)
        self.line(ax, [n52, n22], 'A28')
        self.junc(ax, *n22)
        # Superior mesenteric
        self.line(ax, [n22, (-0.9, 20.3)], 'A34')
        self.term(ax, -0.9, 20.3)

        # Abdominal aorta B
        n23 = (0, 19.0)
        self.line(ax, [n22, n23], 'A35')
        self.junc(ax, *n23)
        # Renal R
        self.line(ax, [n23, (0.9, 18.8)], 'A36')
        self.term(ax, 0.9, 18.8)
        # Renal L
        self.line(ax, [n23, (-0.9, 18.8)], 'A38')
        self.term(ax, -0.9, 18.8)

        # Abdominal aorta C
        n24 = (0, 17.5)
        self.line(ax, [n23, n24], 'A37')
        self.junc(ax, *n24)

        # Abdominal aorta D
        n25 = (0, 16.0)
        self.line(ax, [n24, n25], 'A39')
        self.junc(ax, *n25)
        # Inferior mesenteric
        self.line(ax, [n25, (-0.9, 15.8)], 'A40')
        self.term(ax, -0.9, 15.8)

        # Abdominal aorta E -> iliac bifurcation
        n13 = (0, 14.5)
        self.line(ax, [n25, n13], 'A41')
        self.junc(ax, *n13)

        # ==============================================================
        # SECTION 3: LEGS  (y = 0 .. 14)
        # ==============================================================
        # R leg (positive x), L leg (negative x)

        # R iliac
        n17 = (1.5, 13.0)
        self.line(ax, [n13, n17], 'A42')
        self.junc(ax, *n17)
        # R inner iliac stub
        self.line(ax, [n17, (0.8, 12.5)], 'A45')
        self.term(ax, 0.8, 12.5)

        # R external iliac
        n18 = (2.0, 11.5)
        self.line(ax, [n17, n18], 'A44')
        self.junc(ax, *n18)
        # R deep femoral stub
        self.line(ax, [n18, (2.8, 11.2)], 'A47')
        self.term(ax, 2.8, 11.2)

        # R femoral
        n19 = (2.2, 9.5)
        self.line(ax, [n18, n19], 'A46')
        self.junc(ax, *n19)

        # R posterior tibial
        self.line(ax, [n19, (2.1, 6.0)], 'A48')
        self.term(ax, 2.1, 6.0)
        # R anterior tibial
        self.line(ax, [n19, (2.8, 6.0)], 'A49')
        self.term(ax, 2.8, 6.0)

        # L iliac (mirror)
        n14 = (-1.5, 13.0)
        self.line(ax, [n13, n14], 'A43')
        self.junc(ax, *n14)
        self.line(ax, [n14, (-0.8, 12.5)], 'A51')
        self.term(ax, -0.8, 12.5)

        n15 = (-2.0, 11.5)
        self.line(ax, [n14, n15], 'A50')
        self.junc(ax, *n15)
        self.line(ax, [n15, (-2.8, 11.2)], 'A53')
        self.term(ax, -2.8, 11.2)

        n16 = (-2.2, 9.5)
        self.line(ax, [n15, n16], 'A52')
        self.junc(ax, *n16)

        self.line(ax, [n16, (-2.1, 6.0)], 'A54')
        self.term(ax, -2.1, 6.0)
        self.line(ax, [n16, (-2.8, 6.0)], 'A55')
        self.term(ax, -2.8, 6.0)

        # ==============================================================
        # LEGEND
        # ==============================================================
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Forward flow'),
            Line2D([0], [0], color='blue', linewidth=2, label='Reversed flow'),
            Line2D([0], [0], color='#bbbbbb', linewidth=1, label='Minimal flow'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=7, linewidth=0, label='Junction'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#777777',
                   markersize=5, linewidth=0, label='Terminal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, linewidth=0, label='Heart'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
                  framealpha=0.95)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"[SUCCESS] Body tree saved: {save_path}")
        return fig


def main():
    results_base = Path.home() / "first_blood/projects/simple_run/results"
    output_dir = Path.home() / "first_blood/analysis_V3/visualizations"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING ARTERIAL TREE - PAPER STYLE")
    print("  Head fan at top | Torso center | Legs bottom")
    print("  White open junctions | Gray terminals | No boxes")
    print("=" * 70 + "\n")

    viz = PaperStyleTree('patient025_CoW_v2', results_base)
    path = output_dir / "arterial_tree_paper_style.png"
    viz.create(save_path=path)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()