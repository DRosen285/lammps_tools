import numpy as np

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


def read_multiple_lammps_trajectories(filenames, fields_to_extract=None, stride=1):
    """
    Reads multiple LAMMPS trajectory files and combines the results.

    Parameters:
        filenames: list of str, paths to LAMMPS trajectory files
        fields_to_extract: list of fields to keep (e.g. ['id', 'xu', 'yu', 'zu']), or None to keep all
        stride: int, only every `stride`-th frame will be read from each file

    Returns:
        frames: list of NumPy structured arrays (1 per frame)
        timesteps: list of ints
        box_bounds: list of [(xlo,xhi), (ylo,yhi), (zlo,zhi)] per frame
    """
    all_frames = []
    all_timesteps = []
    all_box_bounds = []

    for filename in filenames:
        frames, timesteps, box_bounds = read_lammps_trajectory_fast(
            filename, fields_to_extract=fields_to_extract, stride=stride
        )
        all_frames.extend(frames)
        all_timesteps.extend(timesteps)
        all_box_bounds.extend(box_bounds)

    return all_frames, np.array(all_timesteps), all_box_bounds




#convert read trajectory information to numpy arrays

def convert_traj(frames,box_bounds,num_part,traj_size,x_clean,y_clean,z_clean,m_clean,box_array,t_array,unwrapped=False):
    for i in range(0,traj_size):
        for j in range(0,num_part):
            
            if unwrapped:
                x_clean[i][j][2]=frames[i][j]["xu"]#assign coordinate
                y_clean[i][j][2]=frames[i][j]["yu"]#assign coordinate
                z_clean[i][j][2]=frames[i][j]["zu"]#assign coordinate

            else:
                x_clean[i][j][2]=frames[i][j]["x"]#assign coordinate
                y_clean[i][j][2]=frames[i][j]["y"]#assign coordinate
                z_clean[i][j][2]=frames[i][j]["z"]#assign coordinate

            x_clean[i][j][0]=frames[i][j]["id"]#assign particle #
            x_clean[i][j][1]=frames[i][j]["type"]#assign particle type

            y_clean[i][j][0]=frames[i][j]["id"]#assign particle #
            y_clean[i][j][1]=frames[i][j]["type"]#assign particle type

            z_clean[i][j][0]=frames[i][j]["id"]#assign particle #
            z_clean[i][j][1]=frames[i][j]["type"]#assign particle type

            m_clean[i][j][0]=frames[i][j]["id"]#assign particle #
            m_clean[i][j][1]=frames[i][j]["type"]#assign particle type
            m_clean[i][j][2]=frames[i][j]["mass"]#assign mass

        box_array[i][0]= np.sqrt((box_bounds[i][0][1]-box_bounds[0][0][0])**2)
        box_array[i][1]= np.sqrt((box_bounds[i][1][1]-box_bounds[0][1][0])**2)
        box_array[i][2]= np.sqrt((box_bounds[i][2][1]-box_bounds[0][2][0])**2)
        t_array[i]=i
    
    return (x_clean, y_clean, z_clean, m_clean, box_array, t_array)    
 
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

