import os
import h5py
import argparse
import numpy as np
from numpy.core.fromnumeric import shape

skimage = None
mcubes = None

def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')
    tensor = h5f[key][()]
    h5f.close()

    return tensor


def exctract_colors_v(vertices, sdf):

    colors_v = np.empty_like(vertices)

    for i in range(len(vertices)):
        colors_v[i,:] = sdf[(int)(np.round(vertices[i,1])), np.round((int)(vertices[i,0])), np.round((int)(vertices[i,2])), 1:]

    return colors_v

def exctract_colors_f(colors_v, faces):

    colors_f = np.empty([len(faces), 3])

    for i in range(len(faces)):
        color_face = np.zeros(3)
        for j in range(len(faces[i])):
            color_face += colors_v[faces[i,j]]
        
        color_face /= len(faces[i])

        colors_f[i,:] = color_face

    return colors_f

def write_off(file, vertices, faces, colors_f):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face, color in zip(faces, colors_f):
            assert len(face) == 3, 'only triangular faces supported (%s)' % (file)
            # fp.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
            fp.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str((int)(color[0])) + ' ' + str((int)(color[1])) + ' ' + str((int)(color[2])) + '\n')

        # add empty line to be sure
        fp.write('\n')

try:
    from skimage import measure

    def marching_cubes(tensor):
        """
        Perform marching cubes using mcubes.

        :param tensor: input volume
        :type tensor: numpy.ndarray
        :return: vertices, faces
        :rtype: numpy.ndarray, numpy.ndarray
        """

        vertices, faces, normals, values = measure.marching_cubes_lewiner(tensor.transpose(1, 0, 2), 0)
        return vertices, faces

    print('Using skimage\'s marching cubes implementation.')
except ImportError:
    print('Could not find skimage, import skimage.measure failed.')
    print('If you use skimage, make sure to call voxelize with -mode=corner.')

    try:
        import mcubes

        def marching_cubes(tensor):
            """
            Perform marching cubes using mcubes.

            :param tensor: input volume
            :type tensor: numpy.ndarray
            :return: vertices, faces
            :rtype: numpy.ndarray, numpy.ndarray
            """

            return mcubes.marching_cubes(-tensor.transpose(1, 0, 2), 0)

        print('Using PyMCubes\'s marching cubes implementation.')
    except ImportError:
        print('Could not find PyMCubes, import mcubes failed.')
        print('You can use the version at https://github.com/davidstutz/PyMCubes.')
        print('If you use the voxel_centers branch, you can use -mode=center, otherwise use -mode=corner.')

if mcubes == None and measure == None:
    print('Could not find any marching cubes implementation; aborting.')
    exit(1);

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('input', type=str, help='The input HDF5 file.')
    parser.add_argument('output', type=str, help='Output directory for OFF files.')

    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print('Input file does not exist.')
        exit(1)

    tensor = read_hdf5(args.input)
    if len(tensor.shape) == 5:

        if not os.path.exists(args.output):
            os.makedirs(args.output)
            print('Created output directory.')
        else:
            print('Output directory exists; potentially overwriting contents.')

        for n in range(tensor.shape[0]):
            print('Minimum and maximum value: %f and %f. ' % (np.min(tensor[n,:,:,:,0]), np.max(tensor[n,:,:,:,0])))
            vertices, faces = marching_cubes(tensor[n,:,:,:,0])
            colors_v = exctract_colors_v(vertices, tensor[n,:,:,:,:])
            colors_f = exctract_colors_f(colors_v, faces)
            off_file = '%s/%d.off' % (args.output, n)
            write_off(off_file, vertices, faces, colors_f)
            print('Wrote %s.' % off_file)
    else:
        print('Minimum and maximum value: %f and %f. ' % (np.min(tensor[:,:,:,0]), np.max(tensor[:,:,:,0])))
        vertices, faces = marching_cubes(tensor[:,:,:,0])
        colors_v = exctract_colors_v(vertices, tensor[:,:,:,:])
        colors_f = exctract_colors_f(colors_v, faces)
        off_file = '%s' % (args.output)
        write_off(off_file, vertices, faces, colors_f)
        print('Wrote %s.' % off_file)


    print('Use MeshLab to visualize the created OFF files.')