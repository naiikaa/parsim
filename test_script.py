import numpy as np
import deeptrack as dt
import tqdm
import cv2
import multiprocessing as mp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--width', type=int, default=1920, help='Width of the image')
parser.add_argument('--height', type=int, default=1080, help='Height of the image')
parser.add_argument('--num_particles_min', type=int, default=3, help='Minimum number of particles')
parser.add_argument('--num_particles_max', type=int, default=4, help='Maximum number of particles')
parser.add_argument('--wavelength_start', type=float, default=300e-9, help='Start wavelength in meters')
parser.add_argument('--wavelength_end', type=float, default=700e-9, help='End wavelength in meters')
parser.add_argument('--num_wavelengths', type=int, default=10, help='Number of wavelengths in the spectrum')
parser.add_argument('--particle_radius', type=float, default=1.4e-6, help='Average particle radius in meters')
parser.add_argument('--particle_radius_variation', type=float, default=0.1, help='Variation in particle radius in percent')
parser.add_argument('--z_min', type=float, default=0e-6, help='Minimum z position of particles')
parser.add_argument('--z_max', type=float, default=1e-6, help='Maximum z position of particles')
parser.add_argument('--intensity', type=float, default=10, help='Intensity of the particles')
parser.add_argument('--na', type=float, default=0.7, help='Numerical aperture of the microscope')
parser.add_argument('--magnification', type=int, default=50, help='Magnification of the microscope')
parser.add_argument('--magnification_multiplier', type=int, default=1, help='Magnification multiplier for fine-tuning')
parser.add_argument('--refractive_index', type=float, default=1.44, help='Refractive index of the particles')
parser.add_argument('--refractive_index_medium', type=float, default=1.333, help='Refractive index of the medium')
parser.add_argument('--noise_sigma', type=float, default=0.1, help='Sigma of the Gaussian noise')
parser.add_argument('--sensor_resolution', type=float, default=14e-6, help='Sensor resolution in meters per pixel')
args = parser.parse_args()

particles = dt.MieSphere(
        refractive_index = args.refractive_index, 
        position = lambda: (
            np.random.random()*args.height,
            np.random.random()*args.width,
        ),
        radius = args.particle_radius ,
        position_unit="meter",
        z = lambda: args.z_min + np.random.random() * (args.z_max - args.z_min),
        L="auto",
        intensity=args.intensity,
    )

spectrum = np.linspace(args.wavelength_start, args.wavelength_end, args.num_wavelengths)


microscopes = [dt.Brightfield(NA=args.na,
                                wavelength=wl,
                                magnification=args.magnification*args.magnification_multiplier,
                                output_region=(0, 0, args.height, args.width),
                                refractive_index_medium=args.refractive_index_medium,
                                resolution = args.sensor_resolution,
                                )
                for wl in spectrum]

noise = dt.Gaussian(sigma=args.noise_sigma)

particle_number = lambda: np.random.randint(args.num_particles_min, args.num_particles_max)

sample = (particles^particle_number)
image = sum([microscope(sample) for microscope in tqdm.tqdm(microscopes)])
augmented_image = (image >> noise >> dt.NormalizeMinMax())

res = None
for _ in tqdm.tqdm(range(5)):
    intermediate_image = augmented_image()
    if res is None:
        res = intermediate_image
    else:
        res += intermediate_image
    augmented_image.update()

#save image with cv2
cv2.imwrite("simulated_image.png", (res*255).astype(np.uint8))
