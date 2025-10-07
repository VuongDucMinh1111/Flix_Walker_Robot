import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import yaml                      # Làm việc với file YAML
import matplotlib.pyplot as plt   # Vẽ đồ thị
import numpy                     # Các phép toán ma trận
import math                      # Các phép toán toán học
import yaml                      # Làm việc với file YAML
import mediapy as media          # Thư viện cho media (video, ảnh)

# import xml.etree.ElementTree as ET

# tree = ET.parse('TestBeamTest3.xml')
# root = tree.getroot()

# for j in root.iter('joint'):
#     if j.attrib['name'] == 'Y2':
#         springref = float(j.attrib.get('springref', 0))
#         print("Springref Y2 from XML:", springref)

xml_path = 'TestBeam1.xml' #xml file (assumes this is in the same folder as this file)
simend = 500 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

_overlay = {}
def add_overlay(gridpos, text1, text2):

    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text1 + "\n"
    _overlay[gridpos][1] += text2 + "\n"

#HINT1: add the overlay here
def create_overlay(model,data):
    global F_friction
    topleft = mj.mjtGridPos.mjGRID_TOPLEFT
    topright = mj.mjtGridPos.mjGRID_TOPRIGHT
    bottomleft = mj.mjtGridPos.mjGRID_BOTTOMLEFT
    bottomright = mj.mjtGridPos.mjGRID_BOTTOMRIGHT

    add_overlay(bottomleft,"Restart",'r' ,)
    add_overlay(bottomleft, "Sim_Time", f"{data.time:.3f} s")


    # Hiển thị overlay lực
    add_overlay(bottomleft,"Applied_Force",f"Fx={applied_force_world[0]:.4f}, Fy={applied_force_world[1]:.4f}, Fz={applied_force_world[2]:.4f}")

    # Góc joint Y2 (deg)
    y2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "Y2")
    angle_y2_deg = np.rad2deg(data.qpos[y2_id])
    add_overlay(bottomleft, "Y2_angle", f"{angle_y2_deg:.2f} deg")

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass


applied_force_world = np.zeros(3)  # Lực apply thực tế
def controller_all(model, data):
    global applied_force_world
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "O")
    y1_dof_adr = model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "Y1")]
    y2_dof_adr = model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "Y2")]
    if 7 < data.time < 7.15:
        force_world = np.array([-0.001, -0.0, 0]) #X cho Test thong thuong / Y cho Test 3
        torque = np.array([0, 0, 0])


        # Cập nhật biến hiển thị (nếu cần)
        applied_force_world = force_world.copy()
        point = data.xipos[body_id] # vị trí tâm body trong world
        # Áp lực
        mj.mj_applyFT(model, data, force_world, torque, point, body_id, data.qfrc_applied)
    else:
        data.xfrc_applied[body_id] = np.zeros(6)
        applied_force_world[:] = 0
        data.qfrc_applied[y1_dof_adr] = 0
        data.qfrc_applied[y2_dof_adr] = 0




def keyboard_ball(window, key, scancode, act, mods):

    #Reset
    if key == glfw.KEY_R:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)



def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

# def quat2euler(q):
#     w, x, y, z = q
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll = np.arctan2(t0, t1)

#     t2 = +2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     pitch = np.arcsin(t2)

#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw = np.arctan2(t3, t4)

#     return np.array([roll, pitch, yaw])

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data

# <-- CHÈN VỊ TRÍ BAN ĐẦU CHO Y2 -->
y2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "Y2")
data.qpos[y2_id] = np.deg2rad(90)  # đặt góc ban đầu
mj.mj_forward(model, data)

cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

quat_sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "framequat_S1")
time_log = []
TrucX = []
TrucY = []
TrucZ = []

#joint_names = ["Y1","Y2"]
joint_names = [ "X2", "Y2"]
#joint_names = [ "Y2"] #TestBeamTest2
joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
joint_logs = {name: [] for name in joint_names}

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
if not window:
    raise Exception("Failed to create GLFW window")
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard_ball)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 31.651059535822398 ; cam.elevation = -51.58324924318847 ; cam.distance =  0.43763355205046667
cam.lookat =np.array([ 0.023957818772395004 , -0.02064014509067261 , 0.7731286189560816 ])

#initialize the controller
init_controller(model,data)



# đăng ký controller 
mj.set_mjcb_control(controller_all)

glfw.set_key_callback(window, keyboard_ball)

#site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "S1")
#point = data.site_xpos[site_id].copy()   # vị trí world
#body_id = model.site_bodyid[site_id]


while not glfw.window_should_close(window):
    time_prev = data.time
    # In ra giá trị ctrl để debug
    print(f"Time: {data.time:.2f}, Ctrl: {data.ctrl[0]}")

    # Lấy dof (degree of freedom) address của joint Y2 một lần
    y1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "Y1")
    y2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "Y2")
    # Mỗi joint có thể có nhiều DoF, nhưng hinge joint chỉ có 1
    # Chúng ta cần địa chỉ của DoF đó trong mảng qpos_spring
    qpos_adr = model.jnt_qposadr[y2_id] 
    dof_adr = model.jnt_dofadr[y2_id]
    dof_adr1 = model.jnt_dofadr[y1_id]
    # In ra springref của joint Y2 theo cách mới
    current_springref_rad = model.qpos_spring[qpos_adr]
    current_springref_deg = np.rad2deg(current_springref_rad) 
    print(f"Time: {data.time:.2f}, Y2 springref: {current_springref_deg:.2f} deg")

    qfrc_val = data.qfrc_applied[dof_adr]
    qfrc_val1 = data.qfrc_applied[dof_adr1]
    print(f"Time: {data.time:.3f}, qfrc_applied[Y2]: {qfrc_val:.6f}, qfrc_applied[Y1]: {qfrc_val1:.6f}")


    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)


    if (data.time>=simend):
        break;

    # quat = data.sensordata[quat_sensor_id*4 : quat_sensor_id*4 + 4]
    # euler_rad = quat2euler(quat)
    # X_deg = np.rad2deg(euler_rad[0])
    # Y_deg = np.rad2deg(euler_rad[1])                
    # Z_deg = np.rad2deg(euler_rad[2])                 
    time_log.append(data.time)
    # TrucX.append(X_deg)   
    # TrucY.append(Y_deg)
    # TrucZ.append(Z_deg)

    # Lưu góc từng joint (chuyển sang độ)
    for name, jid in zip(joint_names, joint_ids):
        joint_logs[name].append(np.rad2deg(data.qpos[jid]))

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #create overlay
    create_overlay(model,data)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

     # overlay items
    for gridpos, [t1, t2] in _overlay.items():

        mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150,gridpos,viewport,t1,t2,context)

    # clear overlay
    _overlay.clear()

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

# plt.plot(time_log, TrucX, label="TrucX")
# plt.plot(time_log, TrucY, label="TrucY")
# plt.plot(time_log, TrucZ, label="TrucZ")
# plt.xlabel("Time [s]")
# plt.ylabel("Goc [do]")
# plt.title("Góc quay chân (site S1)")
# plt.legend()
# plt.show()

for name in joint_names:
    plt.plot(time_log, joint_logs[name], label=name)

plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Góc quay các joint X1, X2, Y1, Y2")
plt.legend()
plt.show()

glfw.terminate()
