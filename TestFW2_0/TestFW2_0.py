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
import xml. etree . ElementTree as ET


xml_path = 'Bai2_0test.xml' #xml file (assumes this is in the same folder as this file)
#xml_path = 'Idle(No_Rotor).xml' #xml file (assumes this is in the same folder as this file)

simend = 500 #simulation time
print_camera_config = 0 #set to 1 to print camera config. This is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

tree = ET.parse("Bai2_0test.xml")
root = tree.getroot()

# <!-- friction mac dinh: friction="0.6 0.002 0.002"-->
for elem in root.iter("geom"):
    elem.set("friction", "0.8 0.1 0.1")

# Tìm thẻ <option>   
options = root.findall("option")

if len(options) > 0:
    option = options[0]   # lấy option có sẵn
else:
    option = ET.SubElement(root, "option")  # nếu chưa có thì mới tạo

# Set lại timestep & iterations (0.001 - 0.002 / >100)
option.set("timestep", "0.001")   # nhỏ hơn thì mô phỏng mượt hơn nhưng chậm
option.set("iterations", "200")  # số vòng lặp solver mỗi bước

# mac dinh stiffness="1.062072" damping="2.21209"
for elem in root.iter("joint"):
    if "X" in elem.get("name", ""):
        elem.set("stiffness", "1.062072")
        elem.set("damping", "2.21209")

#mac dinh stiffness="0.0024" damping="0.00183"
for elem in root.iter("joint"):
    if "Y" in elem.get("name", ""):
        elem.set("stiffness", "0.183")
        elem.set("damping", "0.24")

# de 2 rotor ko va nhau, min = 2* .00125 = 0.0025
for elem in root.iter("body"):
    if "A1" in elem.get("name", ""):
        elem.set("pos", "0 -.014 .005")
    if "A2" in elem.get("name", ""):
        elem.set("pos", "0 -.014 -.005")



# Lưu file mới tree.write("Bai2_0test_modified.xml")
#Lưu đè lên file cũ
tree.write("Bai2_0test.xml")




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

def get_center_of_mass(model, data):
    masses = model.body_mass
    positions = data.xipos
    total_mass = np.sum(masses)
    com = np.sum(masses[:, None] * positions, axis=0) / total_mass
    return com


def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass


m = 0.012   # khối lượng rotor (kg)
r = 0.014   # bán kính lệch tâm (m)
f = -30     # tần số quay mong muốn (Hz)
omega_d = 2 * numpy.pi * f  # tốc độ góc mục tiêu (rad/s)
I = m * r**2  # mômen quán tính xấp xỉ của rotor (kg·m²)
	
Kp = 0.0001    # hệ số tỉ lệ (tạo torque khi có sai số tốc độ)
Kd = 0.00  # hệ số dập tắt (tạo torque chống rung, chống overshoot

moving_dict = {}
def controller_all(model, data):
    global f_friction,target_ball, kp, b_act
    #mô hình ma sát Stribeck/Columb
    try:
        data.qfrc_applied[:] = 0
        for name in sensor_names:
            sensor_id = model.sensor(name).id
            objtype   = model.sensor(name).objtype
            objid     = model.sensor(name).objid #model co sensor

            # Lực đo bởi sensor (fx, fy, fz)
            # ---- 1. Lực local tại sensor ----
            f_local = np.array(data.sensordata[sensor_id : sensor_id+3])

            # ---- 2. Đổi sang world frame ----
            site_xmat = data.site_xmat[objid].reshape(3,3)   # rotation matrix của site
            f_world = site_xmat @ f_local

            fx, fy, fz = f_world

            f_horizontal = np.sqrt(fx**2 + fy**2)
            fz_abs = abs(fz)
            f_max_static = mu_s * fz_abs  # Ngưỡng ma sát nghỉ
            f_friction_x = 0.0
            f_friction_y = 0.0

            if f_horizontal < f_max_static:
        #         dính  
                f_friction_x = -fx
                f_friction_y = -fy
                f_friction = np.sqrt(fx**2 + fy**2)
                moving = False
            else:
        #         trượt
        # Lực ma sát ngược chiều với VẬN TỐC của điểm tiếp xúc
                # Lấy vector vận tốc (tịnh tiến + quay) của site trong world frame
                vel = np.zeros(6, dtype=np.float64)
                objid = int(np.ravel(objid)[0])   # đảm bảo là int Python, không phải np.array
                #print("objid raw:", objid, "type:", type(objid))
                if objid < 0:
                    raise ValueError("Site không tồn tại, name2id trả về -1")
                mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_SITE, objid, vel, 0)
                #print("Velocity:", vel)
                # Chỉ cần vận tốc tịnh tiến (vx, vy, vz)
                site_vel_linear = vel[3:6]
                #print("site_vel_linear:", site_vel_linear)
            
                # Tính tốc độ trên mặt phẳng ngang (xy)
                v_horizontal_speed = np.linalg.norm(site_vel_linear[:2])
                fz_abs = max(fz_abs, 1e-6)
                vel_s = 0.1   # tốc độ chuẩn hoá nhỏ
                eps = 1e-6
                mu_eff = mu_k + (mu_s - mu_k) * np.exp(-(v_horizontal_speed/vel_s)**2)  # smooth from mu_s->mu_k
                f_mag = mu_eff * fz_abs

                #print("v_horizontal_speed:", v_horizontal_speed)

                if v_horizontal_speed > 1e-6: # Thêm một ngưỡng nhỏ để tránh chia cho 0
                    # Vector đơn vị chỉ hướng của vận tốc
                    v_direction = site_vel_linear[:2] / v_horizontal_speed
                    #print("v_direction:", v_direction)
                
                    # Lực ma sát có độ lớn không đổi (mu_k * fz_abs)
                    # và ngược hướng với vector vận tốc
                    f_friction_vec = -mu_k * fz_abs * v_direction
                    #f_friction_vec = -f_mag * v_direction
                
                    f_friction_x = f_friction_vec[0]
                    f_friction_y = f_friction_vec[1]
                    f_friction   = 0.0
                    moving = True
                else:
                    f_friction_x = 0
                    f_friction_y = 0
                    f_friction   = 0.0
                    moving = False
                # Nếu vận tốc quá nhỏ, coi như không có ma sát trượt để tránh bất ổn
                # f_friction_x và f_friction_y sẽ giữ giá trị 0.
                # angle = np.arctan2(fy, fx)
                # f_friction_x = -mu_k * fz_abs * np.cos(angle)
                # f_friction_y = -mu_k * fz_abs * np.sin(angle)
                # f_friction = np.sqrt(f_friction_x**2 + f_friction_y**2)
            # ---- 2. Apply lực ma sát tại sensor ----
            # Lấy vị trí sensor để gán lực 
            if objtype == mj.mjtObj.mjOBJ_SITE:
                pos_site = data.site_xpos[objid]          # world pos của site
                body_id = model.site_bodyid[objid].item()      # body chứa site
            elif objtype == mj.mjtObj.mjOBJ_BODY:
                pos_site = data.xipos[objid]              # world pos khối tâm body
                body_id  = objid
            else:
                pos_site = np.zeros(3)
                body_id  = 0

            # ---- 3. Apply lực ma sát ----
            force  = np.array([f_friction_x, f_friction_y, 0.0], dtype=np.float64).reshape(3,)
            torque = np.zeros(3, dtype=np.float64)
            pos_site = data.site_xpos[objid].astype(np.float64).ravel()
            #pos_site = data.xipos[body_id]
            #body_id = int(body_id)
            mj.mj_applyFT(model, data, force, torque, pos_site, body_id, data.qfrc_applied) #bat/tat friction gia lap
            #print("Torque applied:", torque)
    except Exception as e:
        print("\n=== Exception in controller_all ===")
        traceback.print_exc()
        raise  

    #     # lưu lực ma sát của sensor này
    for name in sensor_names:
        try:
            F_friction_dict[name] = f_friction
            moving_dict[name] = moving
        except Exception as e:
            print(f"Error at sensor {name}:", e)

    #print(f"{name}: fx={fx:.6f}, fy={fy:.6f}, fz={fz:.6f}, "f"f_horizontal={f_horizontal:.6f}, "f"f_max_static={f_max_static:.6f}, "f"f_friction={f_friction:.6f}, moving={moving}")
    #print(f"World force: ({fx:.2f}, {fy:.2f})")
    #print(f"Friction force: ({f_friction_x:.2f}, {f_friction_y:.2f})")

    # Lấy actuator index cho ball

    # Gán torque cho 2 rotor quay ngược chiều nhau
    def get_qvel(model, data, joint_name):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        dof_id = model.jnt_dofadr[jid]
        return data.qvel[dof_id]
    w1 = get_qvel(model, data, "R1")
    w2 = get_qvel(model, data, "R2")
    t = data.time          # thời gian mô phỏng hiện tại
    def mycontroller(model, data):
        w1 = data.qvel[1]   # tốc độ góc joint R1
        w2 = data.qvel[2]   # tốc độ góc joint R2
        err1 = omega_d - w1
        err2 = omega_d - w2

        #công thức PD: t = Kp * (W_desired-W_sim) - Kd*W_sim
        torque1 = Kp * err1 - Kd * w1
        torque2 = Kp * err2 - Kd * w2
        data.ctrl[0] = torque1*1
        data.ctrl[1] = torque2*1

    mycontroller(model, data) #bat/tat rotor





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
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

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
cam.azimuth = 31.651059535822398 ; cam.elevation = -51.58324924318847 ; cam.distance =  1.469714557439742
cam.lookat =np.array([ 0.016898883570088997 , -0.009188887376575292 , 0.003225280529476309 ])

com_history = []
time_history = []

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

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # lưu lại trọng tâm tại mỗi bước
    com = get_center_of_mass(model, data)
    com_history.append(com.copy())
    time_history.append(data.time)

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

    _overlay.clear()


    glfw.swap_buffers(window)

    glfw.poll_events()

# sau khi kết thúc mô phỏng, vẽ quỹ đạo trọng tâm
com_history = np.array(com_history)

plt.figure()
plt.plot(time_history, com_history[:, 0], label='CoM X')
plt.plot(time_history, com_history[:, 1], label='CoM Y')
plt.plot(time_history, com_history[:, 2], label='CoM Z')
plt.xlabel("Time (s)")
plt.ylabel("Center of Mass Position (m)")
plt.legend()
plt.title("Center of Mass trajectory over time")
plt.show()

glfw.terminate()

