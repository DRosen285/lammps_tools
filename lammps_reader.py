import numpy as np
import lammpsio

def read_lammps_trajectory_fast(filename, fields_to_extract=None, stride=1):
    """
    Reads a LAMMPS trajectory efficiently using NumPy, with optional stride.

    Parameters:
        filename: str
        fields_to_extract: list of fields to keep (e.g. ['id', 'xu', 'yu', 'zu']), or None to keep all
        stride: int, only every `stride`-th frame will be read (default = 1, read all)

    Returns:
        frames: list of NumPy structured arrays (1 per frame)
        timesteps: list of ints
        box_bounds: list of [(xlo,xhi), (ylo,yhi), (zlo,zhi)] per frame
    """
    frames = []
    timesteps = []
    box_bounds = []

    frame_idx = 0  # count total frames
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                f.readline()  # ITEM: NUMBER OF ATOMS
                num_atoms = int(f.readline().strip())

                f.readline()  # ITEM: BOX BOUNDS
                bounds = [tuple(map(float, f.readline().strip().split())) for _ in range(3)]

                header_line = f.readline().strip().split()
                assert header_line[:2] == ["ITEM:", "ATOMS"]
                all_fields = header_line[2:]

                # Decide what fields to extract
                if fields_to_extract is None:
                    fields = all_fields
                else:
                    fields = [f for f in all_fields if f in fields_to_extract]

                indices = [all_fields.index(f) for f in fields]
                dtype = [(f, 'f8') for f in fields]

                if frame_idx % stride == 0:
                    data = np.empty((num_atoms,), dtype=dtype)
                    for i in range(num_atoms):
                        values = f.readline().strip().split()
                        selected = [float(values[j]) for j in indices]
                        data[i] = tuple(selected)
                    frames.append(data)
                    timesteps.append(timestep)
                    box_bounds.append(bounds)
                else:
                    # Skip lines for this frame
                    for _ in range(num_atoms):
                        f.readline()

                frame_idx += 1

    return frames, np.array(timesteps), box_bounds


#function to check that molecule is not broken over pbc, otherwise com coordinates are wrong
def check_pbc(x_ref,y_ref,z_ref,x,y,z,box_array):
    dist=np.sqrt((x-x_ref)**2+(y-y_ref)**2+(z-z_ref)**2)
    x_max=box_array[0]*0.5
    y_max=box_array[1]*0.5
    z_max=box_array[2]*0.5
    if dist > box_array[0]*0.5:
            x=x-x_max*2*round((x-x_ref)/(x_max*2))
            y=y-y_max*2*round((y-y_ref)/(y_max*2))
            z=z-z_max*2*round((z-z_ref)/(z_max*2))
    else:
        x=x
        y=y
        z=z
#            print(dist_x,dist_y,dist_z)
    return(x,y,z)

