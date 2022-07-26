from copy import deepcopy
import argparse
import subprocess
import os
from time import sleep

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import rosbag
import numpy as np
import matplotlib.pyplot as plt

import pickle
from mpl_toolkits.mplot3d import axes3d

from gp_mpc.gp_wrapper import gp_model
from gp_mpc.control import start_node
from gp_mpc.helper_fns import yaml_load, force_comp_to_world


def bag_loader(path, map_and_append_msg, topic_name = 'robot_state', normalize = ''):
    '''
    Load a specified topic from a rosbag, returning the an array ordered to timestamp
    '''
    bag = rosbag.Bag(path)
    num_obs = bag.get_message_count(topic_name)
    if num_obs is 0:
        topic_name = '/'+topic_name
        num_obs = bag.get_message_count(topic_name)
    print('Loading ros bag {}  with {} msgs on topic {}'.format(path, num_obs, topic_name))

    msgs = {}
    t = []
    for _, msg, t_ros in bag.read_messages(topics=[topic_name]):
        t.append(t_ros.to_sec())
        map_and_append_msg(msg, msgs)
    t = [tt-t[0] for tt in t]

    msgs_in_order = {}
    for key in msgs.keys():
        t_in_order, el_in_order = zip(*sorted(zip(t,msgs[key])))
        msgs_in_order[key] = np.array(el_in_order).T
    msgs_in_order['t'] = t_in_order

    if normalize is not '':
        msgs_in_order[normalize] = (msgs_in_order[normalize].T-msgs_in_order[normlize][:,0]).T

    return msgs_in_order

def get_aligned_msgs(msgs1, msgs2):
    ''' 
    Select entries from msgs2 which occured most recently before msgs1
    '''
    aligned_msgs2 = {key:[] for key in msgs2.keys()}
    t2 = np.array(msgs2['t'])
    for t1 in msgs1['t']:
        last_before_t1 = np.where(t2<=t1)[0][-1] # find last time in t which is 
        for key in msgs2.keys():
            if key == 't': continue
            aligned_msgs2[key].append(msgs2[key][:,last_before_t1])

    for key in msgs2.keys():
        aligned_msgs2[key] = np.array(aligned_msgs2[key]).T

    return aligned_msgs2

def map_robot_state(msg, prev_msgs):
    if len(prev_msgs) is 0:
        for el in ('pos', 'vel', 'force'):
            prev_msgs[el] = []
    prev_msgs['pos'].append(msg.position)
    prev_msgs['vel'].append(msg.velocity)
    prev_msgs['force'].append(msg.effort)
    return prev_msgs

def map_impedance_gains(msg, prev_msgs):
    if len(prev_msgs) is 0:
        for el in ('K', 'B', 'M', 'Fd'):
            prev_msgs[el] = []
    prev_msgs['K'].append(msg.position)
    prev_msgs['B'].append(msg.velocity)
    prev_msgs['M'].append(msg.effort[:len(msg.position)])
    prev_msgs['Fd'].append(msg.effort[len(msg.position):])

def map_delta_impedance_gains(msg, prev_msgs):
    if len(prev_msgs) is 0:
        for el in ('dK', 'dB', 'dM', 'Fd'):
            prev_msgs[el] = []
    prev_msgs['dK'].append(msg.position)
    prev_msgs['dB'].append(msg.velocity)
    prev_msgs['dM'].append(msg.effort[:len(msg.position)])
    prev_msgs['Fd'].append(msg.effort[len(msg.position):])
    
def plot_model_cov(model_path, ext = 0.05):

    
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
        modes = list(models.keys())
        print('Loaded models for {}'.format(modes))

    state_bounds = [np.full(3,1e10), np.full(3,-1e10)]
    for mode in modes:
        current_range = models[mode].get_xrange()
        state_bounds = [np.minimum(state_bounds[0],current_range[0]),
                        np.maximum(state_bounds[1],current_range[1])]

    fig2 = plt.figure(dpi=200)
    ax2 = fig2.gca(projection='3d')
    max_force = np.zeros((3,1))
    min_cov = np.full((3,1), 1e10)
    max_cov = np.zeros((3,1))
    exp = ext
    off = 0
    x, y, z = np.meshgrid(np.linspace(state_bounds[0][0+off]-exp, state_bounds[1][0+off]+exp, 5),
                          np.linspace(state_bounds[0][1+off]-exp, state_bounds[1][1+off]+exp, 5),
                          np.linspace(state_bounds[0][2+off]-exp, state_bounds[1][2+off]+exp, 5))
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    offset = 0.0
    colors = ['r','g','b']
    for mode in modes:
        means = []
        covs   = []
        c = colors.pop()
        test_pt = models[mode].get_mean_state()
        for xi, yi, zi in zip(x,y,z):
            test_pt[off:off+3] = np.array([xi,yi,zi])
            mu, cov = models[mode].predict(test_pt)
            covs.append(0.5*np.min(np.diag(cov)))
        ax2.scatter(x+offset, y+offset, z+offset, s=covs, color = c)
        offset += 0.02
    #print(covs)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('GP for {}'.format(mode))
    return fig2, ax2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="data/rail/", help="Root folder for data & config")
    parser.add_argument("--rebuild_gp", default=False, action='store_true',
                        help="Force a new Gaussian process to build")
    parser.add_argument("--skip_validate", default=False, action='store_true',
                        help="Skip the evaluation step, use ready .bags to plot")
    parser.add_argument("-f","--files", nargs='+', help='Files to validate on', required=False)
    args = parser.parse_args()

    model_path = args.path+"GP_models.pkl"

    if args.rebuild_gp:
        if os.path.exists(model_path): os.remove(model_path)
        #subprocess.Popen(["python3", "-m" "gp_mpc.control", "--path", args.path])
        mpc_params = yaml_load(args.path, 'mpc_params.yaml')
        models = gp_model(args.path, rotation = mpc_params['enable_rotation'])
        models.load_models(rebuild = True)

    files = args.files
    if not args.skip_validate:
        for fi in files:
            print("Validating with file {}".format(fi))
            subprocess.Popen(["python3", "-m" "gp_mpc.control", "--path", args.path])
            sleep(1.0)
            subprocess.Popen(["rosbag", "record", "-a","-O","".join([args.path, "validate_", fi])])
            os.system("".join(["rosbag play -r 1.0 ", args.path, fi, " && rosnode kill -a"]))

    fig = plt.figure(dpi=200)
    ax = fig.gca(projection='3d')
    # fig, ax =  plot_model_cov(model_path)

    scale_B = 0.00004
    scale_M = 0.002
    num_plot_pts = 30

    for bag in ["".join([args.path, "validate_", fi]) for fi in files]:
        imp_msgs = bag_loader(bag, map_impedance_gains, topic_name = 'impedance_gains')
        state_msgs = bag_loader(bag, map_robot_state, topic_name = 'robot_state')

        state_msgs_aligned = get_aligned_msgs(imp_msgs, state_msgs)

        subsample_rate = int(state_msgs_aligned['pos'].shape[1]/num_plot_pts)
        skipcnt = 0
        for p, B, M  in zip(state_msgs_aligned['pos'].T, imp_msgs['B'].T, imp_msgs['M'].T):
            if skipcnt > subsample_rate:
                skipcnt = 0
            else:
                skipcnt += 1
                continue
            B = force_comp_to_world(p, B)
            M = force_comp_to_world(p, M)
            rot = 0.15
            line_damp = ax.plot([p[0]-scale_B*B[1], p[0]+scale_B*B[1]],
                    [p[1]-scale_B*rot*B[1], p[1]+scale_B*rot*B[1]],
                    [p[2], p[2]],'r', label = 'Damping')[0]
            ax.plot([p[0]+scale_B*rot*B[2], p[0]-scale_B*rot*B[2]],
                    [p[1]-scale_B*B[2], p[1]+scale_B*B[2]],
                    [p[2], p[2]],'r')
            ax.plot([p[0], p[0]],
                    [p[1], p[1]],
                    [p[2]-scale_B*B[2], p[2]+scale_B*B[2]],'r')
            p = p + 0.0005*np.ones(6)
            line_mass = ax.plot([p[0]-scale_M*M[1], p[0]+scale_M*M[1]],
                    [p[1]-scale_M*rot*M[1], p[1]+scale_M*rot*M[1]],
                    [p[2], p[2]],'b', label = 'Mass')[0]
            ax.plot([p[0]+scale_M*rot*M[2], p[0]-scale_M*rot*M[2]],
                    [p[1]-scale_M*M[2], p[1]+scale_M*M[2]],
                    [p[2], p[2]],'b')
            ax.plot([p[0], p[0]],
                    [p[1], p[1]],
                    [p[2]-scale_M*M[2], p[2]+scale_M*M[2]],'b')
        ax.view_init(elev=90, azim=0)
    plt.xlabel('X (meter)')
    plt.ylabel('Y (meter)')
    plt.legend(handles=[line_damp, line_mass],labels=['Damping', 'Mass'])
    plt.show()
