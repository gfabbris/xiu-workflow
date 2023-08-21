import vtk
from vtk.util import numpy_support as npSup
import numpy as np

def LoadData(filename=None):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    data = reader.GetOutput()
    dim = data.GetDimensions()
#    print ("dim" + str(dim))

    vec = list(dim )
#    print ("vec: %s" % vec)
    vec = [i for i in dim]
    vec.reverse()
#    print vec
    u = npSup.vtk_to_numpy(data.GetPointData().GetArray('Scalars_'))
    maxVal = np.nanmax(u)
    minVal = np.nanmin(u)
#    print ("Data Min/Max Raw Values: %s/%s" % (str(minVal), str(maxVal)))
    u = u.reshape(vec)

    ctrdata = np.swapaxes(u, 0, 2)

    origin = data.GetOrigin()
#    print ("origin: %s" % (origin, ))
    spacing = data.GetSpacing()
#    print ("spacing: %s" % (spacing, ))
    extent = data.GetExtent()
#    print ("extent: %s" % (extent, ))

    x = []
    y = []
    z = []
    for point in range(extent[0], extent[1] + 1):
        x.append(origin[0] + point * spacing[0])
    for point in range(extent[2], extent[3] + 1):
        y.append(origin[1] + point * spacing[1])
    for point in range(extent[4], extent[5] + 1):
        z.append(origin[2] + point * spacing[2])
#    print ("H (" + str(x[0]) + ", " + str(x[-1]) + ")")
#    print ("K (" + str(y[0]) + ", " + str(y[-1]) + ")")
#    print ("L (" + str(z[0]) + ", " + str(z[-1]) + ")")
    axes = [x, y, z]

    return axes, ctrdata

