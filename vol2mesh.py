import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import glob
import os
from skimage import measure
from plotly.offline import iplot
from plotly.tools import FigureFactory as FF
import vtk.

def decimate(mesh, multiplier=0.25):
    mesh = vtk.


def load_dicom_image_folder(folder):
    """
    Load a folder of 2D dicom images and return a 3d volumetric numpy object
    will ignore files that are not in proper format
    Args:
        folder(str):
            the folder where the dicom images are.

    Returns:
        numpy.ndarray of the volume
    """
    volume = np.array([])
    count = 0
    for file in glob.glob(os.path.join(folder, "*.dcm")):
        data = pydicom.dcmread(file)
        if hasattr(data, "pixel_array"):
            if count == 0:
                volume = np.array([data.pixel_array])
            else:
                volume = np.append(volume, np.array([data.pixel_array]), axis=0)
            count += 1
    return volume


def sample_stack(stack, rows=6, cols=6, start_with=0, show_every=1):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


def make_mesh(image, threshold=-300, step_size=1):
    p = image.transpose(2, 1, 0) #  why?

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    mesh = pymesh
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)


def plt_3d(verts, faces):
    print("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor([0,0,0])
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    # ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    print("showing")
    plt.show()


location = r"C:\Users\alpha\tmp\head-neck-pet-ct\Head-Neck-PET-CT\HN-CHUM-001\08-27-1885-PANC. avec C.A. SPHRE ORL   tte et cou  -TP-74220\3-StandardFull-07232"
d3 = load_dicom_image_folder(location)
v, f = make_mesh(d3, threshold=300, step_size=10)
print(v.shape) # TODO smplification
plt_3d(v,f)