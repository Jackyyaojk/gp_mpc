import yaml
import casadi as ca
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

####################################################################################
#### ROS functions

def get_empty_jointstate():
    msg = JointState()
    msg.position = np.zeros(6)
    msg.velocity = np.zeros(6)
    msg.effort = np.zeros(12)
    msg.header.stamp = rospy.Time.now()
    return msg

def get_empty_pose():
    msg = PoseStamped()
    msg.header.frame_id = "panda_link0"
    msg.header.stamp = rospy.Time.now()
    return msg


def get_empty_marker():
    msg = Marker()
    msg.header.frame_id = "panda_link0"
    msg.header.stamp = rospy.Time.now()
    msg.type = 2
    msg.color.r = 1.0
    msg.color.a = 1.0
    return msg

# Define the mapping from ROS msg to the state
def msg_to_state(msg):
    q = np.array([msg.transform.rotation.w,
         msg.transform.rotation.x,
         msg.transform.rotation.y,
         msg.transform.rotation.z])
    r = quat_to_rotvec(q)
    p = np.array([msg.transform.translation.x,
         msg.transform.translation.y,
         msg.transform.translation.z])
    return np.hstack((p.T,np.squeeze(r)))

# Define the mapping from ROS msg to observation
def msg_to_obs(msg):
    return np.array([msg.wrench.force.x,
                     msg.wrench.force.y,
                     msg.wrench.force.z])

def compliance_to_world(init_pose, x):
    # Translate x from being in init_pose frame to world frame.
    #R = rotvec_to_rotation(init_pose[3:])
    #x_w = R@x[:3]+init_pose[:3]
    #if self.mpc_params['enable_rotation']:
    #    x_w = ca.vertcat(x_w, R.T@x[3:])
    #return x_w
    # Old method!
    q0 = rotvec_to_quat(init_pose[3:])           # Initial robot orientation, quaternion
    x_w = quat_vec_mult(q0, x[:3])+init_pose[:3] # Linear position in world coords
    if x.size()[0] == 6:
        x_w = ca.vertcat(x_w, quat_to_rotvec(quat_quat_mult(xyz_to_quat(x[3:]), q0)))
    return x_w

####################################################################################
#### Welcome to the orientation zone ####
#### Functions are grouped by the first representation in the argument list.


def cross(a,b):
    return ca.vertcat(a[1]*b[2]-a[2]*b[1],
                      a[2]*b[0]-a[0]*b[2],
                      a[0]*b[1]-a[1]*b[0])

#### Rotation vectors ####

def rotvec_to_rotation(vec):
    ty = ca.SX if type(vec) is ca.SX else ca.DM
    rot = ty.zeros(3,3)
    phi_ = ca.sqrt(ca.sumsqr(vec))
    phi = ca.if_else(phi_<1e-9, 1e-9, phi_)
    kx = vec[0]/phi
    ky = vec[1]/phi
    kz = vec[2]/phi

    cp = ca.cos(phi)
    sp = ca.sin(phi)
    vp = 1-cp
    rot[0,0] =  kx*kx*vp   +cp
    rot[0,1] =  kx*ky*vp-kz*sp
    rot[0,2] =  kx*kz*vp+ky*sp
    rot[1,0] =  kx*ky*vp+kz*sp
    rot[1,1] =  ky*ky*vp   +cp
    rot[1,2] =  ky*kz*vp-kx*sp
    rot[2,0] =  kx*kz*vp-ky*sp
    rot[2,1] =  ky*kz*vp+kx*sp
    rot[2,2] =  kz*kz*vp   +cp
    return rot

def rotvec_to_quat(r):
    norm_r = ca.norm_2(r)
    th_2 = norm_r/2.0
    return ca.vertcat(ca.cos(th_2),
                      ca.sin(th_2)*r[0]/norm_r,
                      ca.sin(th_2)*r[1]/norm_r,
                      ca.sin(th_2)*r[2]/norm_r)

def rotvec_rotvec_mult(r1, r2):
    a1 = ca.norm_2(r1)
    a2 = ca.norm_2(r2)
    diff = (a2-a1)/2.0
    add  = (a2+a1)/2.0
    r1 *= 1/a1
    r2 *= 1/a2

    a = 2*ca.acos((1-r1.T@r2)*ca.cos(diff)-(1+r1.T@r2)*ca.cos(add))
    r = (ca.sin(add)+ca.sin(diff))*r1+(ca.sin(add)-ca.sin(diff))*r2+(ca.cos(diff)-ca.cos(add))*cross(r2,r1)
    #a = 2*ca.acos(ca.cos(a2/2)*ca.cos(a1/2)-ca.sin(a2/2)*ca.sin(a1/2)*(r1.T@r2))
    #r = ca.sin(a2/2)*ca.cos(a1/2)*r2+ca.sin(a1/2)*ca.cos(a2/2)*r1+ca.sin(a2/2)*ca.sin(a1/2)*cross(r2,r1)

    return a*r/(ca.sin(a/2))

def rotvec_vec_mult(r,v):
    a = ca.norm_2(r)
    r *= 1/a
    return ca.cos(a)*v+ca.sin(a)*cross(r,v)+(1-ca.cos(a))*(r.T@v)*r

#### ZYZ Euler angles ####
# Convert to quaternion from intrinsic ZYZ euler angles
# https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
def euler_to_quat(eu):
    return ca.vertcat(ca.cos(eu[1]/2)*ca.cos(eu[2]/2+eu[0]/2),
                      ca.sin(eu[1]/2)*ca.sin(eu[2]/2-eu[0]/2),
                      ca.sin(eu[1]/2)*ca.cos(eu[2]/2-eu[0]/2),
                      ca.cos(eu[1]/2)*ca.sin(eu[2]/2+eu[0]/2))

# Transform ZYZ Euler angles into a rotation matrix
def euler_to_rotation(eu):
    rot = ca.SX.zeros(3,3)
    rot[0,0] =  ca.cos(eu[0])*ca.cos(eu[1])*ca.cos(eu[2]) - ca.sin(eu[0])*ca.sin(eu[2])
    rot[0,1] = -ca.cos(eu[0])*ca.cos(eu[1])*ca.sin(eu[2]) - ca.sin(eu[0])*ca.cos(eu[2])
    rot[0,2] =  ca.cos(eu[0])*ca.sin(eu[1])
    rot[1,0] =  ca.sin(eu[0])*ca.cos(eu[1])*ca.cos(eu[2]) + ca.cos(eu[0])*ca.sin(eu[2])
    rot[1,1] = -ca.sin(eu[0])*ca.cos(eu[1])*ca.sin(eu[2]) + ca.cos(eu[0])*ca.cos(eu[2])
    rot[1,2] =  ca.sin(eu[0])*ca.sin(eu[1])
    rot[2,0] = -ca.sin(eu[1])*ca.cos(eu[2])
    rot[2,1] =  ca.sin(eu[1])*ca.sin(eu[2])
    rot[2,2] =  ca.cos(eu[1])
    return rot

# Convert a rotation matrix to ZYZ euler coordinates
def rotation_to_euler(rot):
    eu = ca.SX.zeros(3)
    epsilon = 0.999991
    eu[1] = ca.if_else(rot[2,2]>epsilon,
                       0,
                       ca.if_else(rot[2,2]<-epsilon, np.pi, ca.acos(rot[2,2])))
    eu[0] = ca.if_else(ca.fabs(rot[2,2]) >= epsilon,
                       0,
                       ca.atan2(rot[1,2], rot[0,2]))
    eu[2] = ca.if_else(ca.fabs(rot[2,2]) >= epsilon,
                       ca.atan2(rot[1,0], rot[1,1]),
                       ca.atan2(rot[2,1], -rot[2,0]))
    return eu

def eulerpose_to_rotpose(eu):
    ret = np.zeros(6)
    ret[:3] = eu[:3]
    ret[3:] = np.squeeze(quat_to_rotvec(euler_to_quat(eu[3:])))
    return ret

def eulerpose_to_quatpose(x):
    ret = np.zeros(7)
    ret[:3] = x[:3]
    ret[3:] = np.squeeze(euler_to_quat(x[3:]))
    return ret

#### XYZ Euler angles ####
# Convert to quaternion from extrinsic xyz euler angles
def xyz_to_quat(xyz): # Possible to optimize?
    ty = ca.SX if type(xyz) is ca.SX else ca.DM
    q0 = ca.horzcat(ca.cos(xyz[0]/2), ca.sin(xyz[0]/2), ty(0.0), ty(0.0))
    q1 = ca.horzcat(ca.cos(xyz[1]/2), ty(0.0), ca.sin(xyz[1]/2), ty(0.0))
    q2 = ca.horzcat(ca.cos(xyz[2]/2), ty(0.0), ty(0.0), ca.sin(xyz[2]/2))
    # extrinsic = intrinsic with reversed order
    return quat_quat_mult(q0,quat_quat_mult(q1,q2))

def xyz_to_rotation(xyz):
    return quat_to_rotation(xyz_to_quat(xyz))

#### Quaternions ####
#### Mostly following tutorial from Weizmann
def quat_to_rotvec(q):
    q *= ca.sign(q[0])  # multiplying all quat elements by negative 1 keeps same rotation, but only q0 > 0 works here
    th_2 = ca.acos(q[0])
    th = th_2*2.0
    rotvec = ca.vertcat(q[1]/ca.sin(th_2)*th, q[2]/ca.sin(th_2)*th, q[3]/ca.sin(th_2)*th)
    return rotvec

def quat_vec_mult(q,v):
    if type(q) is ca.SX:
        ret = ca.SX.zeros(3)
    elif type(q) is ca.MX:
        ret = ca.MX.zeros(3)
    else:
        ret = ca.DM.zeros(3)
    ret[0] =    v[0]*(q[0]**2+q[1]**2-q[2]**2-q[3]**2)\
             +2*v[1]*(q[1]*q[2]-q[0]*q[3])\
             +2*v[2]*(q[0]*q[2]+q[1]*q[3])
    ret[1] =  2*v[0]*(q[0]*q[3]+q[1]*q[2])\
             +  v[1]*(q[0]**2-q[1]**2+q[2]**2-q[3]**2)\
             +2*v[2]*(q[2]*q[3]-q[0]*q[1])
    ret[2] =  2*v[0]*(q[1]*q[3]-q[0]*q[2])\
             +2*v[1]*(q[0]*q[1]+q[2]*q[3])\
             +  v[2]*(q[0]**2-q[1]**2-q[2]**2+q[3]**2)
    return ret

def quat_quat_mult(q,p):
    if type(q) is ca.SX:
        ret = ca.SX.zeros(4)
    else:
        ret = ca.DM.zeros(4)
    ret[0] = q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3]
    ret[1] = q[0]*p[1]+q[1]*p[0]-q[2]*p[3]+q[3]*p[2]
    ret[2] = q[0]*p[2]+q[1]*p[3]+q[2]*p[0]-q[3]*p[1]
    ret[3] = q[0]*p[3]-q[1]*p[2]+q[2]*p[1]+q[3]*p[0]
    return ret

def quaternion_inv(q):
    r = deepcopy(q)
    r[1:] *= -1.0
    return r

def quat_to_rotation(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    if type(q) is ca.SX:
        r = ca.SX.zeros(3,3)
    else:
        r = ca.DM.zeros(3,3)
    r[0,0] = 2 * (q0 * q0 + q1 * q1) - 1
    r[0,1] = 2 * (q1 * q2 - q0 * q3)
    r[0,2] = 2 * (q1 * q3 + q0 * q2)
     
    r[1,0] = 2 * (q1 * q2 + q0 * q3)
    r[1,1] = 2 * (q0 * q0 + q2 * q2) - 1
    r[1,2] = 2 * (q2 * q3 - q0 * q1)
     
    r[2,0] = 2 * (q1 * q3 - q0 * q2)
    r[2,1] = 2 * (q2 * q3 + q0 * q1)
    r[2,2] = 2 * (q0 * q0 + q3 * q3) - 1
    return r

#### CONVENIENCE FOR TRANSFORMATIONS ####
# Transform from a relative pose in compliance frame x_c to world, based on init_pose
def comp_to_world(init_pose, x_c):
    q0 = rotvec_to_quat(init_pose[3:])
    x_w = quat_vec_mult(q0, x_c[:3])+init_pose[:3]
    r_w = quat_to_rotvec(quat_quat_mult(xyz_to_quat(x_c[3:]), q0))
    return ca.vertcat(x_w, r_w)

# Transform a full trajectory from compliance frame to world
def comp_traj_to_world(init_pose, traj):
    traj_world = np.zeros((6,traj.shape[1]))
    for i in range(traj.shape[1]):
        traj_world[:,i] = np.squeeze(comp_to_world(init_pose, traj[:,i]))
    return traj_world

def force_comp_to_world(rotvec, force_comp):
    rot = rotvec_to_rotation(rotvec)
    return rot.T @ force_comp[0:3]


###############################################################################################

def yaml_load(path, fi, default_path = 'config/'):
    try:
        yaml_file = open(path+fi, 'r')
        print("File {} loaded from {}".format(fi, path))
    except FileNotFoundError:
        print("File {} not found in {} -> loading default in {}".format(fi, path, default_path))
        yaml_file = open(default_path+fi, 'r')

    yaml_content = yaml.load(yaml_file, Loader=yaml.UnsafeLoader)
    local_list = []
    for key, value in yaml_content.items():
        local_list.append((key, value))
    return dict(local_list)

#Split Data in test and training data:
def split(x, y, ratio):
    # training datalength = ratio * datalength
    # test datalength = (1-ratio) * datalength
    if x.shape[1] != y.shape[1]:
        print("Can not split data: x amd y have size {} and {} but must be equal!".format(x.shape, y.shape))
        return
    if 0 >= ratio or 1 <= ratio:
        print("Can not split data: ratio is {} but must be between 0 and 1!".format(ratio))
        return
    N = x.shape[1]
    indices = np.arange(0, N, 1)
    np.random.shuffle(indices)
    x_train = x[:, np.sort(indices[:int(ratio*N)], axis=None)]
    x_test = x[:, np.sort(indices[int(ratio*N)+1:], axis=None)]
    y_train = y[:, np.sort(indices[:int(ratio*N)], axis=None)]
    y_test = y[:, np.sort(indices[int(ratio*N)+1:], axis=None)]
    return x_train, y_train, x_test, y_test

# Build constraints from limits
def constraints(N_p, mpc_params, pos_max = 100.0, vel_max = 10.0, cov_max = 1.0e3,
                u_M_max = 40, u_B_max = 5000, u_K_max = 2000,
                u_dM_max = 0.1, u_dB_max = 1, u_dK_max = 0.5):
    u_lin_max = mpc_params['u_lin_max'],
    u_rot_max = mpc_params['u_rot_max'],
    opti_MBK = np.array(mpc_params['opti_MBK']),
    state_cov = mpc_params['state_cov']

    # constraints for positions an velocities
    if state_cov:
        xlb = list(np.concatenate((np.full(N_p, -pos_max), np.full(N_p, -vel_max), np.full(2*N_p, 1e-6))))
        xub = list(np.concatenate((np.full(N_p, pos_max), np.full(N_p, vel_max), np.full(2*N_p, cov_max))))
    else:
        xlb = list(np.concatenate((np.full(N_p, -pos_max), np.full(N_p, -vel_max))))
        xub = list(np.concatenate((np.full(N_p, pos_max), np.full(N_p, vel_max))))

    # constraints for input
    ulb = [-u_lin_max[0]]*3
    uub = [u_lin_max[0]]*3
    if N_p is 6:
        ulb.extend([-u_rot_max[0]]*3)
        uub.extend([u_rot_max[0]]*3)

    return xlb, xub, ulb, uub
