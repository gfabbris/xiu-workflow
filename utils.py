import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from spec2nexus import spec
import os
from matplotlib import cm, gridspec
import xrayutilities as xu
from PIL import Image
import lmfit
from skimage.draw import line_nd
from rsMap3D.datasource.InstForXrayutilitiesReader import (
    InstForXrayutilitiesReader,
)
from rsMap3D.datasource.DetectorGeometryForXrayutilitiesReader import (
    DetectorGeometryForXrayutilitiesReader,
)

# PLOTTING FUNCTIONS ----------------------------------------------------------


def plot_1d_data(spec_data_file, scans, x, y) -> None:
    # Plots 1D spec column data from a provided list of scans

    # Defines layout for plot
    cols = 2
    rows = len(y) // cols
    if len(y) % cols != 0:
        rows += 1
    position = range(1, len(y) + 1)

    fig = plt.figure(layout="compressed")

    colors = cm.jet(np.linspace(0, 1, len(scans)))

    for idx, y_ in enumerate(y):
        ax = fig.add_subplot(rows, cols, position[idx])
        ax.set_xlabel(x)
        ax.set_ylabel(y_)

        for j, s_n in enumerate(scans):
            scan = spec_data_file.getScan(s_n)
            ax.plot(
                scan.data[x],
                scan.data[y_],
                label=str(s_n),
                color=colors[j],
                alpha=0.5,
            )

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.show()


def plot_3d_isosurface(
    array: np.ndarray, contours=10, alpha=0.2, cmap="jet"
) -> None:
    def get_isosurface(array):
        obj = mlab.contour3d(
            array,
            opacity=alpha,
            contours=contours,
            transparent=True,
            colormap=cmap,
        )

        return obj

    get_isosurface(array)


def plot_3d_volume(array: np.ndarray) -> None:
    def get_volume(array):
        obj = mlab.pipeline.volume(mlab.pipeline.scalar_field(array))
        return obj

    get_volume(array)


# SPECIFIC DATA PLOTTING ------------------------------------------------------

def plot_spec_scan(scan_number, spec_file, x=None, y=None, y0=None, ax=None):
    scan = spec_file.getScan(scan_number)
    valid_keys = list(scan.data.keys())

    if x not in valid_keys:
        if x is not None:
            print(f"The {x} column was not found, using the default.")
        x = valid_keys[0]

    if y not in valid_keys:
        if y is not None:
            print(f"The {y} column was not found, using the default.")
        y = valid_keys[-1]

    if y0 not in valid_keys:
        if y0 is not None:
            print(f"The {y0} column was not found, using the default.")
        y0 = None

    xi = np.array(scan.data[x])
    monitor = 1.0 if y0 == None else np.array(scan.data[y0])
    yi = np.array(scan.data[y])/monitor

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.plot(xi, yi)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()

    if fig is not None:
        return fig, ax
    else:
        return None, None


def plot_fwhm_cent_auc(spec_data_file, scans, x, y, z) -> None:
    fig = plt.figure(layout="compressed", figsize=(10, 6))

    gs = gridspec.GridSpec(nrows=6, ncols=2)
    ax_raw = fig.add_subplot(gs[0:3, 0])
    ax_fit = fig.add_subplot(gs[3:, 0], sharex=ax_raw)
    ax_fwhm = fig.add_subplot(gs[0:2, 1])
    ax_cent = fig.add_subplot(gs[2:4, 1])
    ax_auc = fig.add_subplot(gs[4:, 1])

    colors = cm.jet(np.linspace(0, 1, len(scans)))

    model = lmfit.models.GaussianModel()
    fwhm = []
    centroid = []
    area_under_curve = []
    z_axis_values = []

    for idx, s_n in enumerate(scans):
        scan = spec_data_file.getScan(s_n)

        params = model.guess(np.array(scan.data[y]), x=np.array(scan.data[x]))

        out = model.fit(
            np.array(scan.data[y]), params, x=np.array(scan.data[x])
        )

        fwhm.append(out.params["fwhm"])
        centroid.append(out.params["center"])
        area_under_curve.append(out.params["amplitude"])

        avg_z = scan.data[z][0]
        z_axis_values.append(avg_z)

        ax_raw.plot(
            scan.data[x],
            scan.data[y],
            label=str(round(avg_z, 6)),
            color=colors[idx],
            alpha=0.5,
        )
        ax_fit.plot(
            scan.data[x],
            out.best_fit,
            label=str(round(avg_z, 6)),
            color=colors[idx],
            alpha=0.5,
        )

        ax_raw.set_xlabel(x)
        ax_raw.set_ylabel(y)

        ax_fit.set_xlabel(x)
        ax_fit.set_ylabel(y)

    ax_fwhm.plot(z_axis_values, fwhm)
    ax_fwhm.set_title("FWHM")
    ax_fwhm.set_xlabel(z)
    ax_cent.plot(z_axis_values, centroid)
    ax_cent.set_title("Centroid")
    ax_cent.set_xlabel(z)
    ax_auc.plot(z_axis_values, area_under_curve)
    ax_auc.set_title("Amplitude")
    ax_auc.set_xlabel(z)

    handles, labels = ax_raw.get_legend_handles_labels()
    fig.legend(handles, labels, ncols=1, title=z)

    fig.tight_layout()

    fig.show()


def plot_strain_series(spec_data_file, scans, wavelength, offset, x_val):
    fig = plt.figure(layout="compressed", figsize=(10, 6))

    gs = gridspec.GridSpec(nrows=6, ncols=2)
    ax_raw = fig.add_subplot(gs[0:3, 0])
    ax_fit = fig.add_subplot(gs[3:, 0], sharex=ax_raw)
    ax_d = fig.add_subplot(gs[0:3, 1])
    ax_strain = fig.add_subplot(gs[3:, 1])

    colors = cm.jet(np.linspace(0, 1, len(scans)))

    model = lmfit.models.GaussianModel()

    strain = []
    centers = []
    for idx, s_n in enumerate(scans):
        scan = spec_data_file.getScan(s_n)

        x = np.array(scan.data["Delta"])
        y = np.array(scan.data["imroi1"])
        norm = np.array(scan.data["Ion_Ch_3"])

        y_norm = y / norm

        params = model.guess(y_norm, x=x)

        out = model.fit(y_norm, params, x=x)

        centers.append(out.params["center"])

        if x_val == "force":
            strain.append(
                cap_to_force(scan.data["cap"][0], scan.data["Temp_sam"][0])
            )
        elif x_val == "rbv":
            strain.append(int(scan.data["Rbv2"][0] - scan.data["Rbv1"][0]))
        elif x_val == "cap":
            strain.append(scan.data["cap"][0])

        ax_raw.plot(x, (y_norm + offset * idx), color=colors[idx])
        ax_fit.plot(x, (out.best_fit + offset * idx), color=colors[idx])

    d = wavelength / 2 / np.sin(np.deg2rad(np.array(centers) / 2))

    strain = np.array(strain)

    ax_d.plot(strain, d)
    ax_strain.plot(strain, (d / d[np.argmin(strain)] - 1) * 100)

    if x_val == "force":
        ax_d.set_xlabel("Force (N)")
        ax_strain.set_xlabel("Force (N)")
    elif x_val == "rbv":
        ax_d.set_xlabel("Razorbill Voltage (V)")
        ax_strain.set_xlabel("Razorbill Voltage (V)")
    elif x_val == "cap":
        ax_d.set_xlabel("Capacitance (pF)")
        ax_strain.set_xlabel("Capacitance (pF)")

    ax_raw.set_xlabel("Delta")
    ax_fit.set_xlabel("Delta")

    fig.tight_layout()

    fig.show()


def plot_2d_orthogonal_slice(
    data,
    coords=None,
    x=None,
    y=None,
    z=None,
    scale="linear",
    contour=False,
    cmap="jet",
    axes=["x", "y", "z"],
):
    if coords is None:
        coords = np.array([np.linspace(0, data.shape[i]) for i in range(0, 3)])

    if type(x) == int or type(x) == float:
        idx = np.searchsorted(coords[0], x)
        y_range = [
            np.searchsorted(coords[1], y[0]),
            np.searchsorted(coords[1], y[1]),
        ]
        z_range = [
            np.searchsorted(coords[2], z[0]),
            np.searchsorted(coords[2], z[1]),
        ]

        orth_slice = data[
            idx, y_range[0]: y_range[1], z_range[0]: z_range[1]
        ].T
        extent = (y[0], y[1], z[0], z[1])

        x_axis_lbl = axes[1]
        y_axis_lbl = axes[2]

    elif type(y) == int or type(y) == float:
        idx = np.searchsorted(coords[1], y)
        x_range = [
            np.searchsorted(coords[0], x[0]),
            np.searchsorted(coords[0], x[1]),
        ]
        z_range = [
            np.searchsorted(coords[2], z[0]),
            np.searchsorted(coords[2], z[1]),
        ]

        orth_slice = data[
            x_range[0]: x_range[1], idx, z_range[0]: z_range[1]
        ].T
        extent = (x[0], x[1], z[0], z[1])

        x_axis_lbl = axes[0]
        y_axis_lbl = axes[2]

    elif type(z) == int or type(z) == float:
        idx = np.searchsorted(coords[2], z)
        x_range = [
            np.searchsorted(coords[0], x[0]),
            np.searchsorted(coords[0], x[1]),
        ]
        y_range = [
            np.searchsorted(coords[1], y[0]),
            np.searchsorted(coords[1], y[1]),
        ]

        orth_slice = data[
            x_range[0]: x_range[1], y_range[0]: y_range[1], idx
        ].T
        extent = (x[0], x[1], y[0], y[1])

        x_axis_lbl = axes[0]
        y_axis_lbl = axes[1]

    if contour:
        levels = np.linspace(orth_slice.min(), orth_slice.max(), 20)

        plt.contourf(
            orth_slice,
            norm=scale,
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect="auto",
            levels=levels,
        )
        plt.xlabel(x_axis_lbl)
        plt.ylabel(y_axis_lbl)
        plt.show()
    else:
        plt.imshow(
            orth_slice,
            norm=scale,
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect="auto",
            interpolation="none",
        )
        plt.xlabel(x_axis_lbl)
        plt.ylabel(y_axis_lbl)
        plt.show()


def plot_1d_line_cuts(
    spec_data_file,
    scans,
    point_1,
    point_2,
    z_val,
    instr_config_path,
    det_config_path,
    radius=5,
    log_scale=False,
    plot_3d=True,
):
    line_cuts = []
    z_values = []

    colors = cm.turbo(np.linspace(0, 1, len(scans)))

    fig = plt.figure()
    if plot_3d:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.add_subplot()

    set_bounds = None
    for idx, s_n in enumerate(scans):
        scan_image_directory = (
            "/home/beams/USER6IDB/Data/run23_2_lab2/staff/EuAl4/images/" +
            "EuAl4_001_1/S" + str(s_n)
        )
        scan = spec_data_file.getScan(s_n)
        raw_image_data = get_raw_image_data(
            spec_scan=scan,
            instr_config_path=instr_config_path,
            image_dir=scan_image_directory,
        )
        rsm = create_rsm(
            spec_scan=scan,
            instr_config_path=instr_config_path,
            det_config_path=det_config_path,
        )
        rsm_data_shape = (200, 200, 200)

        h_min, h_max = np.amin(rsm[:, :, :, 0]), np.amax(rsm[:, :, :, 0])
        k_min, k_max = np.amin(rsm[:, :, :, 1]), np.amax(rsm[:, :, :, 1])
        l_min, l_max = np.amin(rsm[:, :, :, 2]), np.amax(rsm[:, :, :, 2])

        rsm_data_bounds = [(h_min, h_max), (k_min, k_max), (l_min, l_max)]

        rsm_data, rsm_data_coords = grid_data(
            raw_image_data=raw_image_data,
            rsm=rsm,
            shape=rsm_data_shape,
            bounds=rsm_data_bounds,
        )

        line_cut_data, lcc = get_1d_line_cut(
            rsm_data, rsm_data_coords, point_1, point_2, radius
        )

        lcc_h_min, lcc_h_max = np.amin(lcc[:, 0]), np.amax(lcc[:, 0])
        lcc_k_min, lcc_k_max = np.amin(lcc[:, 1]), np.amax(lcc[:, 1])
        lcc_l_min, lcc_l_max = np.amin(lcc[:, 2]), np.amax(lcc[:, 2])

        line_cut_bounds = [
            [lcc_h_min, lcc_h_max],
            [lcc_k_min, lcc_k_max],
            [lcc_l_min, lcc_l_max],
        ]

        line_cuts.append(line_cut_data)
        z_values.append(scan.data[z_val][0])

        if plot_3d:
            ax.plot(
                range(line_cut_data.shape[0]),
                line_cut_data,
                scan.data[z_val][0],
                zdir="y",
                color=colors[idx],
                alpha=0.75,
            )
        else:
            if line_cut_bounds[0][0] != line_cut_bounds[0][-1]:
                ax.plot(
                    np.linspace(
                        line_cut_bounds[0][0],
                        line_cut_bounds[0][-1],
                        line_cut_data.shape[0],
                    ),
                    line_cut_data,
                    label=str(round(scan.data[z_val][0], 5)),
                    color=colors[idx],
                    alpha=0.75,
                )
            elif line_cut_bounds[1][0] != line_cut_coords[1][-1]:
                ax.plot(
                    np.linspace(
                        line_cut_bounds[1][0],
                        line_cut_bounds[1][-1],
                        line_cut_data.shape[0],
                    ),
                    line_cut_data,
                    label=str(round(scan.data[z_val][0], 5)),
                    color=colors[idx],
                    alpha=0.75,
                )
            else:
                ax.plot(
                    np.linspace(
                        line_cut_bounds[2][0],
                        line_cut_bounds[2][-1],
                        line_cut_data.shape[0],
                    ),
                    line_cut_data,
                    label=str(round(scan.data[z_val][0], 5)),
                    color=colors[idx],
                    alpha=0.75,
                )

        if set_bounds is None:
            set_bounds = line_cut_bounds
        else:
            for i in range(3):
                if set_bounds[i][0] > line_cut_bounds[i][0]:
                    set_bounds[i][0] = line_cut_bounds[i][0]
                if set_bounds[i][1] < line_cut_bounds[i][1]:
                    set_bounds[i][1] = line_cut_bounds[i][1]

        print(f"{s_n} plotted...")
        print(line_cut_bounds)

    if plot_3d:
        ax.set_xlabel("Line Cut Index")
        ax.set_ylabel("Intensity")
        ax.set_zlabel(z_val)
    else:
        ax.set_xlabel("H")
        if point_1[0] != point_2[0]:
            ax.set_xlim(point_1[0], point_2[0])
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"H = {round(point_2[0], 5)}")
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 36))
        ax2.set_xlabel("K")
        if point_1[1] != point_2[1]:
            ax2.set_xlim(point_1[1], point_2[1])
        else:
            ax2.set_xticks([])
            ax2.set_xlabel(f"K = {round(point_2[1], 5)}")
        ax3 = ax2.twiny()
        ax3.xaxis.set_ticks_position("bottom")
        ax3.xaxis.set_label_position("bottom")
        ax3.spines["bottom"].set_position(("outward", 72))
        ax3.set_xlabel("L")
        if point_1[2] != point_2[2]:
            ax3.set_xlim(point_1[2], point_2[2])
        else:
            ax3.set_xticks([])
            ax3.set_xlabel(f"L = {round(point_2[2], 5)}")

        ax3.set
        ax.set_ylabel("Intensity")

    if log_scale:
        plt.yscale("log")
    fig.tight_layout()
    fig.show()


# DATA RETRIEVAL --------------------------------------------------------------


def get_raw_image_data(spec_scan, instr_config_path, image_dir) -> np.ndarray:
    raw_data = []
    instrument_reader = InstForXrayutilitiesReader(instr_config_path)

    for i, file in enumerate(sorted(os.listdir(image_dir))):
        if file.endswith("tif") and "alignment" not in file:
            image = Image.open(image_dir + "/" + file)
            image_array = np.array(image, dtype=np.int64).T
            raw_data.append(image_array)

    monitor_norm_factor = (
        np.array(spec_scan.data["Ion_Ch_2"])
        * instrument_reader.getMonitorScaleFactor()
    )

    filter_norm_factor = (
        np.array(spec_scan.data["transm"])
        * instrument_reader.getFilterScaleFactor()
    )

    norm_factor = monitor_norm_factor * filter_norm_factor

    return np.array(raw_data[:]) * norm_factor.reshape(norm_factor.size, 1, 1)


def get_2d_orthogonal_slice(data, coords, x=None, y=None, z=None):
    ...


def get_1d_line_cut(data, coords, point_1, point_2, radius):
    offsets = np.arange(-radius, radius + 1)
    offsets_grid = np.meshgrid(offsets, offsets, offsets)
    offsets_array = np.stack(offsets_grid, axis=-1).reshape(-1, 3)
    distances = np.linalg.norm(offsets_array, axis=1)
    valid_offsets = offsets_array[distances <= radius]

    x1, y1, z1 = (np.searchsorted(coords[i], point_1[i]) for i in range(3))
    x2, y2, z2 = (np.searchsorted(coords[i], point_2[i]) for i in range(3))

    points = np.transpose(line_nd((x1, y1, z1), (x2, y2, z2)))
    points_to_average = np.repeat(
        points, offsets_array.shape[0], axis=0
    ) + np.tile(offsets_array, (points.shape[0], 1))
    points_to_average = np.reshape(points_to_average, (points.shape[0], -1, 3))

    smoothed_line_cut_data = []

    for idx, pt in enumerate(points_to_average):
        valid = np.all(
            (points_to_average[idx] >= 0)
            & (points_to_average[idx] < data.shape),
            axis=1,
        )
        valid_points = points_to_average[idx][valid]
        smoothed_data_point = np.mean(
            data[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]]
        )
        smoothed_line_cut_data.append(smoothed_data_point)

    line_cut_data = np.array(smoothed_line_cut_data)
    line_cut_coords = [
        coords[0][valid_points[:, 0]],
        coords[1][valid_points[:, 1]],
        coords[2][valid_points[:, 2]],
    ]

    return line_cut_data, line_cut_coords


# DATA CONVERSION -------------------------------------------------------------


def create_rsm(
    spec_scan: spec.SpecDataFileScan,
    instr_config_path: str,
    det_config_path: str,
) -> np.ndarray:
    """Creates a reciprocal space map for each point in a scan."""

    point_rsm_list = []
    angle_names = []
    rsm_params = {"Energy": 0, "UB_Matrix": None}

    # rsMap3D XML readers
    instrument_reader = InstForXrayutilitiesReader(instr_config_path)
    detector_reader = DetectorGeometryForXrayutilitiesReader(det_config_path)

    # Names of angles used in instrument geometry
    # xrayutilities expects sample circles then detector circles
    sample_circle_names = instrument_reader.getSampleCircleNames()
    detector_circle_names = instrument_reader.getDetectorCircleNames()
    angle_names = sample_circle_names + detector_circle_names

    # Adds each angle to RSM parameters dictionary
    for angle in angle_names:
        rsm_params.update({angle: 0})

    # Checks for initial values of all RSM parameters in spec header
    # Updates parameter if found
    for param in rsm_params.keys():
        if param in spec_scan.positioner:
            rsm_params[param] = spec_scan.positioner[param]

    # Retrieves UB matrix
    ub_list = spec_scan.G["G3"].split(" ")

    # Adds UB matrix to parameters as a 3x3 array
    rsm_params["UB_Matrix"] = np.reshape(ub_list, (3, 3)).astype(np.float64)

    # Retrieves initial energy value
    # Hard-coded because this appears to be user-supplied
    for line in spec_scan.raw.split("\n"):
        if line.startswith("#U"):
            # Updates Energy parameter
            # Converted to eV from keV
            rsm_params["Energy"] = float(line.split(" ")[1]) * 1000
            break

    # Retrieve total number of scan points from spec
    point_count = len(spec_scan.data_lines)

    # Creates a reciprocal space map for every scan point
    for i in range(point_count):
        point_rsm = mapScanPoint(
            point=i,
            spec_scan=spec_scan,
            rsm_params=rsm_params,
            angle_names=angle_names,
            instrument_reader=instrument_reader,
            detector_reader=detector_reader,
        )
        point_rsm_list.append(point_rsm)

    # Convert RSM list to 3D array
    rsm = np.array(point_rsm_list)
    rsm = rsm.swapaxes(1, 3)
    rsm = rsm.swapaxes(1, 2)

    return rsm


def mapScanPoint(
    point: int,
    spec_scan: spec.SpecDataFileScan,
    rsm_params: dict,
    angle_names: list,
    instrument_reader: InstForXrayutilitiesReader,
    detector_reader: DetectorGeometryForXrayutilitiesReader,
) -> np.ndarray:
    """Creates a reciprocal space map for a single scan point."""

    # Gathers parameter values from SPEC data columns
    for i in range(len(spec_scan.L)):
        label = spec_scan.L[i]
        if label in rsm_params.keys():
            rsm_params[label] = spec_scan.data[label][point]

    # RSM process
    # See xrayutilities documentation for more info
    sample_circle_dir = instrument_reader.getSampleCircleDirections()
    det_circle_dir = instrument_reader.getDetectorCircleDirections()
    primary_beam_dir = instrument_reader.getPrimaryBeamDirection()
    q_conv = xu.experiment.QConversion(
        sample_circle_dir, det_circle_dir, primary_beam_dir
    )
    inplane_ref_dir = instrument_reader.getInplaneReferenceDirection()
    sample_norm_dir = instrument_reader.getSampleSurfaceNormalDirection()
    hxrd = xu.HXRD(
        inplane_ref_dir, sample_norm_dir, en=rsm_params["Energy"], qconv=q_conv
    )
    detector = detector_reader.getDetectors()[0]
    pixel_dir_1 = detector_reader.getPixelDirection1(detector)
    pixel_dir_2 = detector_reader.getPixelDirection2(detector)
    c_ch_1 = detector_reader.getCenterChannelPixel(detector)[0]
    c_ch_2 = detector_reader.getCenterChannelPixel(detector)[1]
    n_ch_1 = detector_reader.getNpixels(detector)[0]
    n_ch_2 = detector_reader.getNpixels(detector)[1]
    pixel_width_1 = detector_reader.getSize(detector)[0] / n_ch_1
    pixel_width_2 = detector_reader.getSize(detector)[1] / n_ch_2
    distance = detector_reader.getDistance(detector)
    roi = [0, n_ch_1, 0, n_ch_2]
    hxrd.Ang2Q.init_area(
        pixel_dir_1,
        pixel_dir_2,
        cch1=c_ch_1,
        cch2=c_ch_2,
        Nch1=n_ch_1,
        Nch2=n_ch_2,
        pwidth1=pixel_width_1,
        pwidth2=pixel_width_2,
        distance=distance,
        roi=roi,
    )

    # Retrieves angle values from parameter dictionary
    angle_values = [rsm_params[angle] for angle in angle_names]
    qx, qy, qz = hxrd.Ang2Q.area(*angle_values, UB=rsm_params["UB_Matrix"])

    # Converts list to array
    point_rsm = np.array([qx, qy, qz])

    return point_rsm


def grid_data(raw_image_data, rsm, shape, bounds):
    n_x, n_y, n_z = shape

    x_0, x_1 = bounds[0]
    y_0, y_1 = bounds[1]
    z_0, z_1 = bounds[2]

    gridder = xu.Gridder3D(nx=n_x, ny=n_y, nz=n_z)
    gridder.KeepData(True)
    gridder.dataRange(
        xmin=x_0, xmax=x_1, ymin=y_0, ymax=y_1, zmin=z_0, zmax=z_1, fixed=True
    )

    gridder(rsm[:, :, :, 0], rsm[:, :, :, 1], rsm[:, :, :, 2], raw_image_data)

    rsm_data = gridder.data

    rsm_data_coords = [gridder.xaxis, gridder.yaxis, gridder.zaxis]

    return rsm_data, rsm_data_coords


def cap_to_force(cap, T=290):
    p = np.array(
        [7.53621149e-10, 1.13610411e-06, 2.65618549e-04, 1.58396569e00]
    )
    return 1551 / (cap - np.polyval(p, T) + 1551 / 855) - 855
