from pprint import pprint

import numpy as np
import tomopy
from bluesky import RunEngine
from bluesky.callbacks.broker import LiveImage
from matplotlib.colors import SymLogNorm
from ophyd import sim
from scipy.ndimage import rotate
from ophyd.sim import hw
from tqdm import tqdm
from xpdan.vend.callbacks.best_effort import BestEffortCallback
from xpdan.vend.callbacks.mpl_plotting import LiveGrid
from xpdtools.tools import load_geo
from pyFAI.geometry import Geometry
import matplotlib.pyplot as plt
from xpdan.startup.analysis_server import create_analysis_pipeline, order
from xpdan.startup.viz_server import create_rr
from xpdan.startup.portable_db_server import create_rr as create_db_rr


chif = "/media/christopher/DATA/Research/Columbia/abe/nsls-ii/Ni_calib/iq_q/" \
       "Ni_calib_20180316-183045_030b1a_001_Q.chi.chi"

q, iq = np.loadtxt(chif, skiprows=8).T
q[0] = 0
iq[0] = 0
from scipy.interpolate import interp1d

f = interp1d(q, iq)

hw = hw()

cal_params = {
    "detector": "Dexela 2923",
    "pixel1": 7.5e-05,
    "pixel2": 7.5e-05,
    "max_shape": [3888, 3072],
    "dist": 0.7621271401380089,
    "poni1": 0.10015331959623631,
    "poni2": 0.10956963478404293,
    "rot1": -0.03187787222553664,
    "rot2": -0.058161739315479535,
    "rot3": 4.012025389093514e-11,
    "wavelength": 1.899e-11,
    "pixelX": 75.0,
    "pixelY": 75.0,
    "splineFile": None,
    "directDist": 763.8060733224734,
    "centerX": 1784.971456489912,
    "centerY": 743.3876261493672,
    "tilt": 3.799639270890394,
    "tiltPlanRotation": -61.30462061017937,
    "poni_file_name": "/tmp/tmpq9idb0rk/from_calib_func.poni",
    "time": "20190512-161858",
    "dSpacing": [
        2.03458234862,
        1.761935,
        1.24592214845,
        1.06252597829,
        1.01729117431,
        0.881,
        0.80846104616,
        0.787990355271,
        0.719333487797,
        0.678194116208,
        0.622961074225,
        0.595664718733,
        0.587333333333,
        0.557193323722,
        0.537404961852,
        0.531262989146,
        0.508645587156,
        0.493458701611,
        0.488690872874,
        0.47091430825,
        0.458785722296,
        0.4405,
        0.430525121912,
        0.427347771314,
    ],
    "calibrant_name": "undefined",
}
calibration = load_geo(
    cal_params
)


class take_data:
    def __init__(
            self,
            translation_motor,
            rotation_motor,
            phases,
            raster_pixel_size,
            poni: Geometry,
            order=3,
    ):
        """Pencil beam x-ray diffraction tomography simulation

        Parameters
        ----------
        translation_motor : ophyd.Device
            The translation motor
        rotation_motor : ophyd.Device
            The rotation motor
        phases : list
            A dictionary of phase arrays and functions
        raster_pixel_size : float
            The size of the pixels in meters
        poni : pyfai.geometry.Geometry instance
            The calibration instance
        """
        self.order = order
        self.pixel_size = raster_pixel_size
        # {func: f(geo)->2D scattering pattern,
        # phase: 2D array of phase locations at phi=0}
        self.phases = phases
        self.phase0 = phases[0]["map"]
        self.poni = poni
        self.dist = poni.dist
        self.rotation_motor = rotation_motor
        self.translation_motor = translation_motor
        self.iqs = {}

    def __call__(self, *args, **kwargs):
        rotation = self.rotation_motor.get().readback

        rotated_phases = rotate(self.phase0, rotation, order=self.order)

        # calculate distance to detector
        grid_x, _ = np.mgrid[
                    0: rotated_phases.shape[0], 0: rotated_phases.shape[1]
                    ]
        grid_x -= rotated_phases.shape[0] // 2

        # distance to center is invariant
        translation = rotated_phases.shape[1] // 2 + int(
            self.translation_motor.get().readback / self.pixel_size
        )

        # calculate all the sample to detector distances
        distances = (grid_x * self.pixel_size) + self.dist

        img = np.zeros(self.poni.detector.max_shape)

        # if the translation is bigger than our simulated volume return empty
        # scattering
        if translation >= rotated_phases.shape[1] or translation < 0:
            # return 0.
            return img

        # TODO: parallelize me?
        for idx in range(rotated_phases.shape[0]):
            phases = [
                np.round(
                    rotate(phase["map"], rotation, order=self.order)[
                        idx, translation
                    ],
                    4,
                )
                for phase in self.phases
            ]
            if all(p == 0.0 for p in phases):
                continue
            funcs = [phase["func"] for phase in self.phases]

            d = distances[idx, translation]

            # TODO: accelerate this part! almost 40% own time 45% of total
            self.poni.set_dist(d)
            q = self.poni.qArray() / 10  # convert to Angstroms

            # TODO: parallelize me?
            for i, (f, p) in enumerate(zip(funcs, phases)):
                # phases less than zero make no sense, and we don't need to
                # compute zero phases
                if p <= 0.0:
                    continue
                iq = f(q)
                # contribution is equal to raw scattering (at distance) times
                # the phase fraction
                # print(p)
                img += iq * p
        # return float(np.sum(img))
        return img


t = hw.motor1
t.name = 'x'
t.precision = 5
r = hw.motor2
r.name = 'theta'

from skimage import draw

# arr = np.zeros((40, 40))
# stroke = 2
# # Create an outer and inner circle. Then subtract the inner from the outer.
# radius = 15
# inner_radius = radius - (stroke // 2) + (stroke % 2) - 1
# outer_radius = radius + ((stroke + 1) // 2)
# ri, ci = draw.circle(20, 20, radius=inner_radius, shape=arr.shape)
# ro, co = draw.circle(20, 20, radius=outer_radius, shape=arr.shape)
# arr[ro, co] = 1
# arr[ri, ci] = 0
import bluesky.plans as bp

RE = RunEngine()
viz_rr = create_rr()
db_rr = create_db_rr(
    '/media/christopher/DATA/Research/Columbia/data/tomo_sim_db')

ns = create_analysis_pipeline(order=order, image_names=['dexela'],
                              publisher=db_rr,
                              mask_setting={'setting': 'first'},
                              stage_blacklist=['fq', 'sq', 'pdf', 'mask', 'mask_overlay','calib',
                                               'dark_sub',
                                               'bg_sub'])

RE.subscribe(lambda *x: ns['raw_source'].emit(x))
RE.md = dict(calibration_md=cal_params,
             composition_string='Ni',
             bt_wavelength=.1899,
             analysis_stage='raw')

# important config things!
# TODO: run as pchain
pixel_size = .0002
holder_size = 11e-3  # m
array_size = int(holder_size/pixel_size) + 1
arr = np.zeros((array_size,)*2)

# 1 mm capilary inside {holder_size} mm holder offset near the front
arr[1:7, array_size // 2 - 3:array_size // 2 + 3] = 1

phases = [{"func": f, "map": arr}]
td = take_data(t, r, phases, pixel_size, calibration)

det = sim.SynSignal(name="dexela", func=td)
det.kind = "hinted"

phase_dets = []
for i, phase in enumerate(phases):
    phase_det = sim.SynSignal(
        name=f"phase_{i}",
        func=lambda: rotate(phase["map"], r.get().readback, order=td.order),
    )
    phase_det.kind = "hinted"
    phase_dets.append(phase_det)

# Take a reference shot
# 5mm procession
RE(
    bp.grid_scan(
        [det] + phase_dets,
        r,
        0,
        180,
        19,
        t,
        -holder_size/2.,
        holder_size/2.,
        array_size,
        True,
        md=dict(sample_name='Ni_11mm',
                tomo={'rotation': 'theta',
                      'translation': 'x',
                      'center': array_size/2.,
                      'type': 'pencil'})
    )
)
# 10mm procession
# solid body on axis

"""
Notes:
------

The slow step is the calculation of the q array in pyFAI cython code, it may
be possible to speed this up on GPUs, although we might not care enough to
worry about it.

We could also speed up the calculations over the phases by parallel execution
"""
